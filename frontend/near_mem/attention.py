"""Static sharded KV and fused MHA/GQA for prefill and one decode step.

KV placement is decided once per workload and never remapped during decode:
each KV head owns a contiguous group of cores and every request's context is
split into contiguous near-equal shards inside that head's group. Heads act
as independent parallel operators; there is no runtime re-layout cost.

The fused attention compiler streams every (request, head) KV shard from its
owner core's DRAM. ``query_parts`` trades context parallelism for query
parallelism through a temporal rotation that reuses streamed KV tiles across
query groups held in SRAM.
"""
from dataclasses import dataclass, replace
from math import ceil, gcd

from .graph import Region, window
from .search import divisors


@dataclass(frozen=True)
class Attention:
    batch: int
    queries: int
    context: int  # Includes the newly appended query tokens.
    query_heads: int
    kv_heads: int
    head_dim: int
    causal: bool = True

    def __post_init__(self):
        if min(self.batch, self.queries, self.context, self.query_heads, self.kv_heads, self.head_dim) < 1:
            raise ValueError('Attention dimensions must be positive')
        if self.context < self.queries or self.query_heads % self.kv_heads:
            raise ValueError('Invalid context length or GQA head ratio')

    @property
    def group(self):
        return self.query_heads//self.kv_heads

    @property
    def hidden(self):
        return self.query_heads*self.head_dim


class KVLayout:
    """Static head-partitioned KV sharding.

    ``head_group(head)`` returns the contiguous core range owning ``head``;
    ``shard(request, head, core)`` returns that request's contiguous token
    range inside the group. Attention reads these ranges and the QKV append
    writes the newest tokens into them; the physical map is identical on
    both sides.
    """
    def __init__(self, problem, cores):
        if cores % problem.kv_heads:
            raise ValueError('Head-partitioned KV requires cores divisible by kv_heads')
        self.problem = problem
        self.cores = cores
        self.group_cores = cores//problem.kv_heads
        self.shard_cores = min(self.group_cores, problem.context)
        # Number of shard cores that hold one extra context token.
        self.extra = problem.context % self.shard_cores
        self.base = problem.context//self.shard_cores
        # Per-core KV rows: a core serves one head and ``base`` or ``base+1``
        # tokens of every request's context.
        self.capacity = problem.batch*(self.base + bool(self.extra))

    def head_group(self, head):
        """Contiguous core range ``[start, end)`` owning ``head``."""
        start = head*self.group_cores
        return start, start+self.group_cores

    def shard_index(self, request, head, core):
        """Rank of ``core`` inside ``(request, head)``'s shard set, or None."""
        start, end = self.head_group(head)
        if not start <= core < end:
            return None
        # Rotate the shard start by request so different requests do not all
        # begin on the same group core.
        offset = (core-start - request*self.shard_cores) % self.group_cores
        return offset if offset < self.shard_cores else None

    def shard(self, request, head, core):
        """(start, length) of this request's context owned by ``core``."""
        s = self.shard_index(request, head, core)
        if s is None:
            return None
        return s*self.base + min(s, self.extra), self.base + (s < self.extra)

    def token_owner(self, request, head, token):
        """Core that owns one context token under contiguous sharding."""
        head_start, _ = self.head_group(head)
        lead = self.extra*(self.base+1)
        s = token//(self.base+1) if token < lead else self.extra + (token-lead)//self.base
        return head_start + (request*self.shard_cores + s) % self.group_cores

    def token_slot(self, request, head, token):
        """Physical KV row of one token inside its owner's partition."""
        core = self.token_owner(request, head, token)
        start, _ = self.shard(request, head, core)
        return request*(self.base + bool(self.extra)) + token - start

    def allocate(self, graph, name='kv'):
        """Create the physical KV tensor on every core and return its regions.

        Each region describes where one (head, request, shard) context slice
        lives. A core serves a single KV head; K and V are two column halves
        of width ``head_dim`` in one row per (request, token).
        """
        p = self.problem
        for core in range(self.cores):
            graph.tensor(core, name, [self.capacity, 2*p.head_dim])
        regions = []
        for head in range(p.kv_heads):
            for request in range(p.batch):
                for core in range(*self.head_group(head)):
                    piece = self.shard(request, head, core)
                    if piece is None:
                        continue
                    start, length = piece
                    slot = request*(self.base + bool(self.extra))
                    regions.append(Region(core, name, request*p.context+start,
                                          head*2*p.head_dim, length, 2*p.head_dim, slot, 0))
        return regions

    def append(self, graph, qkv, cache, name='kv_append'):
        """Route only the newest K/V projection rows into their cache slots."""
        p = self.problem
        source = []
        target = []
        # Historical KV is already covered by the cache; only the last
        # ``queries`` rows of each request's context are appended.
        for request in range(p.batch):
            box = Region(0, '', request*p.context+p.context-p.queries, 0,
                         p.queries, 2*p.kv_heads*p.head_dim)
            target.extend(x for r in cache if (x := r.intersect(box)) is not None)
            for head in range(p.kv_heads):
                for half in range(2):
                    columns = p.hidden + (half*p.kv_heads+head)*p.head_dim
                    pieces = window(qkv, request*p.queries, columns, p.queries, p.head_dim)
                    source.extend(replace(r, row=r.row+request*p.context+p.context-p.queries,
                                          col=r.col+(2*head+half)*p.head_dim) for r in pieces)
        graph.convert(name, source, target)
        return target


@dataclass(frozen=True)
class AttentionPlan:
    query_parts: int = 1
    query_tile: int = 1

    def sram_bytes(self, problem):
        r = min(self.query_tile, ceil(problem.queries/self.query_parts))*problem.group
        d = problem.head_dim
        return 14*r*d + 16*r

    def validate(self, layout, sram):
        if min(self.query_parts, self.query_tile) < 1:
            raise ValueError('Attention tile/partition sizes must be positive')
        if self.query_parts > layout.problem.queries or layout.group_cores % self.query_parts:
            raise ValueError('Query partition must divide the per-head core count and query length')
        if self.sram_bytes(layout.problem) > sram:
            raise ValueError('Fused attention live SRAM exceeds capacity')


def plans(layout, sram, rotation=True):
    p = layout.problem
    for pq in divisors(layout.group_cores) if rotation else (1,):
        if pq > p.queries:
            continue
        local = ceil(p.queries/pq)
        tiles = sorted({min(local, x) for x in (1, 4, 8, 16, 32, 64, 128)})
        for tq in tiles:
            plan = AttentionPlan(pq, tq)
            if plan.sram_bytes(p) <= sram:
                yield plan


def compile_attention(graph, layout, plan, query, cache, name='attention'):
    """Fused attention over statically sharded KV.

    Every KV shard streams from its owner core. When ``query_parts > 1`` a
    streamed tile is reused for all query groups before advancing, so a KV
    tile crosses the NoC once per tile instead of once per query group.
    """
    p = layout.problem
    plan.validate(layout, graph.sram)
    pq = plan.query_parts
    g = p.group
    d = p.head_dim
    groups = ceil(p.queries/pq)

    # Distribute query rows inside each head's core group: ``pq`` contiguous
    # query chunks of one (request, head) are spread across the head's
    # ``group_cores`` so heads run as independent parallel operators.
    inputs = []
    outputs = []
    query_views = {}
    for head in range(p.kv_heads):
        hstart, _ = layout.head_group(head)
        for request in range(p.batch):
            for s in range(pq):
                core = hstart + (request*pq+s) % layout.group_cores
                rows = min(groups, p.queries-s*groups)
                if rows <= 0:
                    continue
                qname = f'{name}_q_h{head}_b{request}_s{s}'
                expected = Region(core, qname, request*p.queries+s*groups,
                                  head*g*d, rows, g*d)
                local = next((x for r in query if r.core == core and
                              (x := r.intersect(expected)) is not None
                              and x.area == expected.area), None)
                if local is None:
                    graph.tensor(core, qname, [expected.rows, g*d])
                    local = expected
                inputs.append(local)
                query_views[head, request, s] = local
                oname = f'{name}_o_h{head}_b{request}_s{s}'
                graph.tensor(core, oname, [expected.rows, g*d])
                outputs.append(Region(core, oname, expected.row, head*g*d,
                                      expected.rows, g*d))
    graph.convert(name+'_query_layout', query, inputs)

    timeline = graph.timeline()
    t = 0
    coverage = 0
    tiles = 0
    for head in range(p.kv_heads):
        for request in range(p.batch):
            kv_regions = sorted(
                (r for r in cache if head*2*d <= r.col < (head+1)*2*d
                 and request*p.context <= r.row < (request+1)*p.context),
                key=lambda r: (r.core, r.row))
            for region in kv_regions:
                timeline.memory(t, region)
                # The owner keeps its streamed tile; other group owners pull
                # it through the NoC in the same wave.
                hstart, _ = layout.head_group(head)
                owners = {hstart + (request*pq+s) % layout.group_cores for s in range(pq)
                          if (head, request, s) in query_views}
                for core in owners - {region.core}:
                    timeline.send(t+1, core, region.core, 1, credit=True)
                    timeline.send(t+1, region.core, core, region.area*2)
                for s in range(pq):
                    if (head, request, s) not in query_views:
                        continue
                    core = hstart + (request*pq+s) % layout.group_cores
                    q = query_views[head, request, s]
                    for first in range(0, q.rows, plan.query_tile):
                        count = min(plan.query_tile, q.rows-first)
                        global_first = q.row+first-request*p.queries
                        end = p.context-p.queries+global_first+count if p.causal else p.context
                        tokens = max(0, min(region.rows, request*p.context+end-region.row))
                        if not tokens:
                            continue
                        rows = count*g
                        scores = rows*tokens
                        output = rows*d
                        timeline.work(core, t+2, macs=output*tokens,
                                      read=(output+tokens*d)*2, write=scores*4)
                        timeline.work(core, t+3, vectors=9*scores,
                                      read=9*scores*4, write=scores*4)
                        timeline.work(core, t+4, macs=output*tokens,
                                      read=scores*4+tokens*d*2, write=output*4)
                        timeline.work(core, t+5, vectors=3*output,
                                      read=output*8, write=output*4)
                        coverage += scores
                        tiles += 1
                t += 6
    for head in range(p.kv_heads):
        hstart, _ = layout.head_group(head)
        for request in range(p.batch):
            for s in range(pq):
                if (head, request, s) not in query_views:
                    continue
                core = hstart + (request*pq+s) % layout.group_cores
                region = next(r for r in outputs if r.core == core
                              and r.name == f'{name}_o_h{head}_b{request}_s{s}')
                timeline.work(core, t, vectors=region.area, read=region.area*4,
                              write=region.area*2)
                timeline.memory(t+1, region, True)
    graph.add_timeline(name, timeline, category='attention',
                       sram_bytes=plan.sram_bytes(p),
                       score_elements_issued=coverage, compute_tiles=tiles,
                       score_dram_bytes=0, historical_kv_write_bytes=0,
                       query_parts=pq, query_tile=plan.query_tile)
    graph.release(inputs, keep=query)
    return outputs
