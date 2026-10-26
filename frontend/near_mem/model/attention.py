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

from ..tasks import Region, window
from ..gemm.space import divisors


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

    def regions(self, name='kv'):
        """Where each (head, request, shard) context slice lives.

        A core serves a single KV head; K and V are two column halves of width
        ``head_dim`` in one row per (request, token).
        """
        p = self.problem; regions = []
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

    def allocate(self, graph, name='kv'):
        """Create the physical KV tensor on every core and return its regions."""
        for core in range(self.cores):
            graph.tensor(core, name, [self.capacity, 2*self.problem.head_dim])
        return self.regions(name)

    def appended(self, qkv, cache):
        """(source, target) regions moving the newest K/V projection rows into the cache."""
        p = self.problem; source = []; target = []
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
        return source, target

    def append(self, graph, qkv, cache, name='kv_append'):
        source, target = self.appended(qkv, cache)
        graph.convert(name, source, target)
        return target


@dataclass(frozen=True)
class AttentionPlan:
    """``query_parts × context_parts`` spatial split of one head's fused
    attention inside its core group. ``query_parts`` rotates a shared KV
    tile across query groups; ``context_parts`` splits the context into
    per-core flash-attention segments that merge softmax state + PV
    partials once (a context split-K reduction)."""
    query_parts: int = 1
    context_parts: int = 1
    query_tile: int = 1
    kv_tile: int = 0  # tokens per streamed KV tile; 0 streams a whole owner shard

    def sram_bytes(self, problem):
        qrows = min(self.query_tile, ceil(problem.queries/self.query_parts))*problem.group
        d = problem.head_dim
        # Q + online-softmax state + one streamed KV tile per context stage.
        kvrows = ceil(problem.context/(self.query_parts*self.context_parts))
        if self.kv_tile:
            kvrows = min(kvrows, self.kv_tile)
        return 14*qrows*d + 16*qrows + 2*kvrows*d

    def validate(self, layout, sram):
        if min(self.query_parts, self.context_parts, self.query_tile) < 1 or self.kv_tile < 0:
            raise ValueError('Attention tile/partition sizes must be positive')
        if self.query_parts*self.context_parts > layout.group_cores:
            raise ValueError('query_parts*context_parts must fit inside the per-head core group')
        if self.query_parts > layout.problem.queries:
            raise ValueError('Query partition must not exceed the query length')
        if self.sram_bytes(layout.problem) > sram:
            raise ValueError('Fused attention live SRAM exceeds capacity')


def plans(layout, sram, rotation=True):
    p = layout.problem
    for pq in divisors(layout.group_cores) if rotation else (1,):
        if pq > p.queries:
            continue
        for pc in divisors(layout.group_cores//pq):
            local = ceil(p.queries/pq)
            tiles = sorted({min(local, x) for x in (1, 4, 8, 16, 32, 64, 128)})
            for tq in tiles:
                # Shared rule for every method: the largest KV tile that fits.
                for kv in (0, *(1 << i for i in range(p.context.bit_length(), 3, -1))):
                    plan = AttentionPlan(pq, pc, tq, kv)
                    if plan.sram_bytes(p) <= sram:
                        yield plan
                        break


def assignments(layout, plan):
    """Query blocks of each (head, request): ``(head, request, s, c, region)``.

    Query group ``s`` of one (request, head) is replicated on ``pc`` context-
    segment cores; ``region`` is its global Q/output window on that core. The
    segment ``c == pc-1`` also holds the merged output.
    """
    p = layout.problem; pq, pc = plan.query_parts, plan.context_parts
    groups = ceil(p.queries/pq); width = p.group*p.head_dim
    for head in range(p.kv_heads):
        hstart, _ = layout.head_group(head)
        for request in range(p.batch):
            for s in range(pq):
                rows = min(groups, p.queries-s*groups)
                if rows <= 0:
                    continue
                for c in range(pc):
                    core = hstart + ((request*pq+s)*pc + c) % layout.group_cores
                    yield head, request, s, c, Region(core, '', request*p.queries+s*groups, head*width, rows, width)


def endpoints(layout, plan):
    """Logical Q input and output layouts of one attention plan."""
    blocks = list(assignments(layout, plan))
    return [r for *_, r in blocks], [r for _, _, _, c, r in blocks if c == plan.context_parts-1]


def compile_attention(graph, layout, plan, query, cache, name='attention'):
    """Fused attention over statically sharded KV.

    Every KV shard streams from its owner core. When ``query_parts > 1`` a
    streamed tile is reused for all query groups before advancing, so a KV
    tile crosses the NoC once per tile instead of once per query group.
    """
    p = layout.problem
    plan.validate(layout, graph.sram)
    pq = plan.query_parts
    pc = plan.context_parts
    g = p.group
    d = p.head_dim
    groups = ceil(p.queries/pq)

    # Distribute query rows across the head group's ``pq*pc`` cores: the
    # ``pq`` query chunks of one (request, head) are replicated on ``pc``
    # context-segment cores, each of which keeps a local flash-attention
    # output before the split-K merge.
    inputs = []
    outputs = []
    partials = {}
    query_views = {}
    for head, request, s, c, block in assignments(layout, plan):
        core = block.core
        qname = f'{name}_q_h{head}_b{request}_s{s}_c{c}'
        expected = replace(block, name=qname)
        local = next((x for r in query if r.core == core and
                      (x := r.intersect(expected)) is not None
                      and x.area == expected.area), None)
        if local is None:
            graph.tensor(core, qname, [block.rows, block.cols])
            local = expected
        inputs.append(local)
        query_views[head, request, s, c] = local
        # The last context segment also holds the merged output.
        pname = f'{name}_o_h{head}_b{request}_s{s}' if c == pc-1 else f'{name}_p_h{head}_b{request}_s{s}_c{c}'
        graph.tensor(core, pname, [block.rows, block.cols])
        partials[head, request, s, c] = replace(block, name=pname)
        if c == pc-1:
            outputs.append(partials[head, request, s, c])
    graph.convert(name+'_query_layout', query, inputs)

    # Per-core pipelines. Within one (request, head) task, the ``pc`` context
    # segments are a spatial split over the group's cores; each segment
    # streams only its own KV shard and computes a local flash-attention.
    # A KV tile owned by one segment core is rotated once to the other
    # ``pq-1`` query-group cores of the same segment before advancing.
    times = [0]*graph.cores
    coverage = 0
    tiles = 0
    timeline = graph.timeline()
    for head in range(p.kv_heads):
        hstart, _ = layout.head_group(head)
        seg = ceil(p.context/pc)
        for request in range(p.batch):
            for c in range(pc):
                cstart = request*p.context + c*seg
                cend = min(request*p.context+p.context, cstart+seg)
                kv_regions = sorted(
                    (r for r in cache if head*2*d <= r.col < (head+1)*2*d
                     and cstart <= r.row < cend),
                    key=lambda r: (r.core, r.row))
                step = plan.kv_tile or max((r.rows for r in kv_regions), default=1)
                kv_regions = [r.crop(row, r.col, min(step, r.row+r.rows-row), r.cols)
                              for r in kv_regions for row in range(r.row, r.row+r.rows, step)]
                for region in kv_regions:
                    # Cores of this segment across the pq query groups.
                    segcores = [hstart + ((request*pq+s)*pc + c) % layout.group_cores
                                for s in range(pq) if (head, request, s, c) in query_views]
                    slot = min(times[x] for x in {region.core, *segcores})
                    timeline.memory(slot, region)
                    for s in range(pq):
                        if (head, request, s, c) not in query_views:
                            continue
                        core = hstart + ((request*pq+s)*pc + c) % layout.group_cores
                        if core != region.core:
                            timeline.send(slot+1, region.core, core, region.area*2)
                        q = query_views[head, request, s, c]
                        for first in range(0, q.rows, plan.query_tile):
                            count = min(plan.query_tile, q.rows-first)
                            global_first = q.row+first-request*p.queries
                            end = p.context-p.queries+global_first+count if p.causal else p.context
                            tokens = max(0, min(region.rows,
                                                request*p.context+end-region.row))
                            if tokens <= 0:
                                continue
                            rows = count*g
                            scores = rows*tokens
                            output = rows*d
                            base = times[core] if core != region.core else slot
                            timeline.work(core, base+2, macs=output*tokens,
                                          read=(output+tokens*d)*2, write=scores*4)
                            timeline.work(core, base+3, vectors=9*scores,
                                          read=9*scores*4, write=scores*4)
                            timeline.work(core, base+4, macs=output*tokens,
                                          read=scores*4+tokens*d*2, write=output*4)
                            timeline.work(core, base+5, vectors=3*output,
                                          read=output*8, write=output*4)
                            coverage += scores
                            tiles += 1
                            times[core] = base+6
                    times[region.core] = max(times[region.core], slot+2)
            # Split-K merge: each query group's partials from the pc context
            # segments reduce onto the last segment's core. The merge carries
            # softmax rescale state plus the PV partial, then finalizes.
            for s in range(pq):
                if pc == 1 or (head, request, s, pc-1) not in partials:
                    continue
                dst = partials[head, request, s, pc-1]
                for c in range(pc-1):
                    if (head, request, s, c) not in partials:
                        continue
                    src = partials[head, request, s, c]
                    size = src.area*4 + src.rows*8  # PV partial + (m,l) state
                    t = max(times[src.core], times[dst.core])
                    timeline.send(t, src.core, dst.core, size)
                    times[src.core] = t+1
                    times[dst.core] = t+1
                    timeline.work(dst.core, t+2, vectors=6*src.area+3*src.rows,
                                  read=size*2, write=size)
                    times[dst.core] = t+3
    # Finalize each merged output: normalize by l and write to DRAM.
    for head in range(p.kv_heads):
        hstart, _ = layout.head_group(head)
        for request in range(p.batch):
            for s in range(pq):
                if pc == 1:
                    if (head, request, s, 0) not in partials:
                        continue
                    region = partials[head, request, s, 0]
                    core = region.core
                else:
                    if (head, request, s, pc-1) not in partials:
                        continue
                    region = partials[head, request, s, pc-1]
                    core = region.core
                t = times[core]
                timeline.work(core, t, vectors=region.area, read=region.area*4,
                              write=region.area*2)
                timeline.memory(t+1, region, True)
                times[core] = t+2
    graph.add_timeline(name, timeline, category='attention',
                       sram_bytes=plan.sram_bytes(p),
                       score_elements_issued=coverage, compute_tiles=tiles,
                       score_dram_bytes=0, historical_kv_write_bytes=0,
                       query_parts=pq, context_parts=pc,
                       query_tile=plan.query_tile, kv_tile=plan.kv_tile)
    graph.release(inputs, keep=query)
    return outputs
