"""Static sharded KV and fused MHA/GQA for prefill and one decode step.

KV placement is decided once per workload and never remapped during decode:
each KV head owns a contiguous group of cores and every request's context is
split into contiguous near-equal shards inside that head's group. Heads act
as independent parallel operators; there is no runtime re-layout cost.

The fused compiler streams every KV shard from its owner. Query/context
partitions determine spatial work and replicated Q; independent packet shifts
rotate Q with online state, paired K/V, or both. The common schedule provides
numeric replay and ATLAS lowering, including the resulting output layout.
"""
from dataclasses import dataclass, replace
from math import ceil, gcd

from ..tasks import Region, window
from ..gemm.space import divisors
from ..output import root_index, destinations


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
    """Spatial query/context partitions and temporal packet shifts.

    A query packet carries Q and its online (m,l,U) state. A KV packet carries
    paired K/V. Shifts are logical lanes inside a context segment, never DRAM
    remappings. A zero query shift reproduces the original search domain.
    """
    query_parts: int = 1
    context_parts: int = 1
    query_tile: int = 1
    kv_tile: int = 0
    query_shift: int = 0
    kv_shift: int = 1
    output: str = 'sharded'

    def sram_bytes(self, problem):
        q = min(self.query_tile, ceil(problem.queries/self.query_parts))*problem.group
        k = self.kv_tile or ceil(problem.context/(self.query_parts*self.context_parts))
        d = problem.head_dim
        # Two Q/state banks, score scratch, output scratch and four KV banks
        # (local owner staging plus incoming/outgoing packets at ragged edges).
        return 16*q*d + 16*q + 16*k*d + 4*q*k

    def validate(self, layout, sram):
        root_index(self.output,self.context_parts,self.context_parts-1)
        p = self.query_parts
        if min(p, self.context_parts, self.query_tile) < 1 or self.kv_tile < 0:
            raise ValueError('Attention tile/partition sizes must be positive')
        if p*self.context_parts > layout.group_cores or p > layout.problem.queries:
            raise ValueError('Attention partitions exceed cores or query length')
        if not 0 <= self.query_shift < p or not 0 <= self.kv_shift <= max(1,p-1):
            raise ValueError('Invalid rotation shift')
        if gcd(self.query_shift-self.kv_shift,p) != 1:
            raise ValueError('Relative rotation must cover every Q/KV pair')
        if self.sram_bytes(layout.problem) > sram:
            raise ValueError('Fused attention live SRAM exceeds capacity')


def plans(layout, sram, rotation=True, joint=True):
    """Nested spaces: context parallelism, KV rotation, then Q/joint rotation.

    Search relative unit shifts in both orientations and their single-side
    limits; this is a finite subset, not every coprime rotation schedule.
    """
    p = layout.problem
    for pq in divisors(layout.group_cores) if rotation else (1,):
        if pq > p.queries:
            continue
        shifts = [(0,1)]
        if joint and rotation and pq > 1:
            shifts = list(dict.fromkeys(shifts+[(1,0)]+[(a,(a+d)%pq) for a in range(1,pq) for d in (-1,1)]))
        for pc in divisors(layout.group_cores//pq):
            local = ceil(p.queries/pq)
            for tq in sorted({min(local,x) for x in (1,4,8,16,32,64,128)}):
                # Tile rule and SRAM accounting are identical for every method.
                for kv in (0, *(1 << i for i in range(p.context.bit_length(), -1, -1))):
                    plan = AttentionPlan(pq,pc,tq,kv)
                    if plan.sram_bytes(p) <= sram:
                        for a,b in shifts:
                            yield replace(plan,query_shift=a,kv_shift=b)
                        break


def assignments(layout, plan):
    """Initial query regions, replicated over independent context segments."""
    p = layout.problem; pq,pc = plan.query_parts,plan.context_parts
    groups = ceil(p.queries/pq); width = p.group*p.head_dim
    for head in range(p.kv_heads):
        hstart,_ = layout.head_group(head)
        for request in range(p.batch):
            for s in range(pq):
                rows = min(groups,p.queries-s*groups)
                if rows <= 0:
                    continue
                for c in range(pc):
                    core = hstart+((request*pq+s)*pc+c)%layout.group_cores
                    yield head,request,s,c,Region(core,'',request*p.queries+s*groups,head*width,rows,width)


def endpoints(layout, plan):
    from .attention_schedule import packets, lane_core
    blocks = list(assignments(layout,plan)); output = []
    for h,b,s,c,r in blocks:
        if c != 0:
            continue
        cores=[]
        for segment in range(plan.context_parts):
            rounds = max(map(len,packets(layout,plan,h,b,segment)))
            lane = (s+rounds*(plan.query_parts-1)*plan.query_shift)%plan.query_parts
            cores.append(lane_core(layout,plan,h,b,lane,segment))
        output.extend(replace(r,core=core) for core in destinations(plan.output,cores,plan.context_parts-1))
    return [r for *_,r in blocks],output


def compile_attention(graph, layout, plan, query, cache, name='attention'):
    from .attention_schedule import Schedule
    plan.validate(layout,graph.sram)
    return Schedule(layout,plan,cache).lower(graph,query,name)


def build_attention(problem, chip, plan, placed=False):
    """Single-operator graph shared by proxy and simulation.

    Q starts with one row-sharded copy, or already in the consumer layout
    for whole-graph search (whose edges account for distribution separately).
    """
    from ..tasks import Graph
    graph = Graph(chip); layout = KVLayout(problem,graph.cores)
    cache = layout.allocate(graph); rows = problem.batch*problem.queries
    regions = endpoints(layout,plan)[0] if placed else [
        Region(c,'',rows*c//graph.cores,0,rows*(c+1)//graph.cores-rows*c//graph.cores,problem.hidden)
        for c in range(graph.cores) if rows*(c+1)//graph.cores>rows*c//graph.cores]
    query = []
    for i,r in enumerate(regions):
        name = f'query_{i}'
        graph.tensor(r.core,name,[r.rows,r.cols]); query.append(replace(r,name=name))
    compile_attention(graph,layout,plan,query,cache)
    return graph
