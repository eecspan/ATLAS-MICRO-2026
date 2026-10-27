"""One steady-state Transformer layer, evaluated entirely by native ATLAS.

The original GEMM/Q/KV/FFN collectives remain a real incumbent. Shared additions
are statically sharded KV, causal prefill, fused attention, RMSNorm/residual/RoPE,
and explicit layout edges. Every layer uses the same plan, so the layer input
is the previous layer's FFN output in that plan's ``contract`` layout; one
simulated layer therefore stands for any layer of the model, including the
relayout between layers. Measurement and search live in ``experiments``.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isqrt

from .attention import Attention, AttentionPlan, KVLayout, compile_attention, endpoints
from ..sim import hardware
from ..tasks import Graph, Region
from .kernels import Gemm, pointwise, residual, rmsnorm
from ..gemm.native import best_tile
from ..gemm.plan import Plan, Problem


@dataclass(frozen=True)
class Workload:
    batch: int = 1
    queries: int = 128
    context: int = 128
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    ffn: int = 1536
    rope: bool = True

    @property
    def attention(self):
        return Attention(self.batch,self.queries,self.context,self.query_heads,self.kv_heads,self.head_dim)

    @property
    def problems(self):
        p = self.attention; rows = self.batch*self.queries
        return dict(qkv=Problem(rows,p.hidden+2*p.kv_heads*p.head_dim,p.hidden),
                    output=Problem(rows,p.hidden,p.hidden),
                    expand=Problem(rows,2*self.ffn,p.hidden),
                    contract=Problem(rows,p.hidden,self.ffn))


def configuration(kernels, attention):
    return dict(kernels={k:dict(plan=asdict(v.plan),native=v.native,direction=v.direction,gated=v.gated)
                         for k,v in kernels.items()}, attention=asdict(attention))


def decode_configuration(workload, data):
    def plan(obj):
        return Plan(**{k:tuple(v) if isinstance(v,list) else v for k,v in obj.items()})
    kernels = {k:Gemm(workload.problems[k],plan(v['plan']),v['native'],v['direction'],v['gated'])
               for k,v in data['kernels'].items()}
    return kernels,AttentionPlan(**data['attention'])


def materialize(graph, name, regions):
    """Per-core tensors holding ``regions`` at their local offsets (renamed to ``name``)."""
    extent = {}
    for r in regions:
        rows, cols = extent.get(r.core, (0, 0))
        extent[r.core] = max(rows, r.local_row+r.rows), max(cols, r.local_col+r.cols)
    for core, shape in sorted(extent.items()):
        graph.tensor(core, name, list(shape))
    return [replace(r, name=name) for r in regions]


def port(workload, cores, name, choice=None):
    """Logical layout of one operator boundary; ``choice`` is its bound Gemm or AttentionPlan."""
    p = workload.attention
    if name == 'kv':
        return KVLayout(p, cores).regions()
    op, end = name.split('.')
    if op == 'attention':
        return endpoints(KVLayout(p, cores), choice)[end == 'O']
    return choice.input_layout(cores)[0] if end == 'A' else choice.output_layout(cores)


def edges(workload):
    """Layout conversions ``build`` performs: ``(source, target, alias)``.

    Ports are ``op.A``/``op.C`` of a GEMM, ``attention.Q``/``attention.O``,
    and the KV cache ``kv``. The layer input is ``contract.C`` of the previous
    layer. ``alias`` is the ``tasks.traffic`` rule for same-core pieces.
    """
    return [('contract.C', 'qkv.A', 'bind'), ('qkv.C', 'kv', 'none'), ('qkv.C', 'attention.Q', 'cover'),
            ('attention.O', 'output.A', 'bind'), ('contract.C', 'output.C', 'none'),
            ('output.C', 'expand.A', 'bind'), ('expand.C', 'contract.A', 'bind'), ('output.C', 'contract.C', 'none')]


def edge_regions(workload, cores, target, source, destination):
    """(source, target) regions of an edge; the KV append moves only new K/V columns."""
    if target == 'kv':
        return KVLayout(workload.attention, cores).appended(source, destination)
    return source, destination


def build(workload, chip, config):
    if workload.ffn < 1:
        raise ValueError('Positive FFN width required')
    graph = Graph(chip); p = workload.attention
    if workload.rope and p.head_dim % 2:
        raise ValueError('RoPE requires an even head dimension')
    kernels, attention_plan = decode_configuration(workload,config)
    layout = KVLayout(p,graph.cores)
    kernels = {name:kernels[name].prepare(graph,name) for name in workload.problems}
    cache = layout.allocate(graph,'kv')
    rows = workload.batch*workload.queries
    # Steady state: the input is the previous layer's output, in this plan's contract layout.
    value = materialize(graph,'input',kernels['contract'].output_layout(graph.cores))
    normalized = rmsnorm(graph,'norm_attention',value,rows,p.hidden)
    query,temporary = kernels['qkv'].qkv(graph,normalized,layout,cache,rope=workload.rope)
    graph.release(normalized)
    if workload.rope:
        new_keys = []
        for r in cache:
            request = r.row//p.context
            start = max(r.row,request*p.context+p.context-p.queries)
            end = min(r.row+r.rows,(request+1)*p.context)
            if start < end:
                new_keys.append(r.crop(start,r.col,end-start,p.head_dim))
        pointwise(graph,'rope_k',new_keys,6)
    attended = compile_attention(graph,layout,attention_plan,query,cache,'attention')
    graph.release(temporary)
    projected = kernels['output'].compile(graph,attended)
    graph.release(attended)
    projected = residual(graph,'attention_residual',projected,value)
    graph.release(value,keep=projected); value = projected
    normalized = rmsnorm(graph,'norm_ffn',value,rows,p.hidden)
    expanded = kernels['expand'].compile(graph,normalized)
    graph.release(normalized)
    contracted = kernels['contract'].compile(graph,expanded)
    graph.release(expanded)
    contracted = residual(graph,'ffn_residual',contracted,value)
    graph.release(value,keep=contracted)
    return graph


def tune_native(workload, chip, out, workers=4):
    """Native K/N-grid kernels with the best native tile of each local shape.

    Fixed TP axes: each all-reduce leaves its output where the next GEMM reads
    it. Upstream splits QKV's K over mesh rows, so its layer input (the FFN
    output, split over columns) needs a transpose that upstream never pays;
    QKV uses the matching axis instead. ATLAS's packed single-KV-head
    protocol only supports the upstream axis.
    """
    arch,_,_ = hardware(chip); side = isqrt(arch['core_num'])
    if side*side != arch['core_num'] or side % 2:
        raise ValueError('Native cloud baseline needs an even square mesh')
    kernels = {}
    for name,problem in workload.problems.items():
        m,n,k = problem.shape
        if n % side or k % side:
            raise ValueError('Native K/N shape must divide mesh side')
        tm,tk,tn = best_tile((m,n//side,k//side),chip,out/name,workers)['tiling_factors']
        direction = dict(expand=1, qkv=int(workload.kv_heads > 1)).get(name, 0)
        kernels[name] = Gemm(problem,Plan((1,side,side),(tm,tn,tk)),True,direction,name=='expand')
    return kernels
