"""Fixed-workload Transformer graphs, evaluated entirely by native ATLAS.

The original GEMM/Q/KV/FFN collectives remain a real incumbent. Shared additions
are statically sharded KV, causal prefill, fused attention, RMSNorm/residual/RoPE,
and explicit layout edges. Measurement and search live in ``experiment.py``.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isqrt
from pathlib import Path

import yaml

from .attention import Attention, AttentionPlan, KVLayout, compile_attention
from .backend import hardware
from .graph import Graph, Region
from .kernels import Gemm, pointwise, residual, rmsnorm
from .native import matrix_placement
from .plan import Plan, Problem


@dataclass(frozen=True)
class Workload:
    batch: int = 1
    queries: int = 128
    context: int = 128
    query_heads: int = 8
    kv_heads: int = 2
    head_dim: int = 64
    ffn: int = 1536
    layers: int = 1
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


def row_layout(graph, name, rows, width):
    result = []
    for core in range(graph.cores):
        start = rows*core//graph.cores; end = rows*(core+1)//graph.cores
        if end > start:
            graph.tensor(core,name,[end-start,width])
            result.append(Region(core,name,start,0,end-start,width))
    return result


def build(workload, chip, config):
    if workload.layers < 1 or workload.ffn < 1:
        raise ValueError('Positive layer count and FFN width required')
    graph = Graph(chip); p = workload.attention
    if workload.rope and p.head_dim % 2:
        raise ValueError('RoPE requires an even head dimension')
    selected, attention_plan = decode_configuration(workload,config)
    layout = KVLayout(p,graph.cores)
    layers = []
    for layer in range(workload.layers):
        kernels = {name:deepcopy(selected[name]).prepare(graph,f'l{layer}_{name}') for name in workload.problems}
        cache = layout.allocate(graph,f'l{layer}_kv')
        layers.append((kernels,cache))
    rows = workload.batch*workload.queries
    value = row_layout(graph,'input',rows,p.hidden)
    for layer,(kernels,cache) in enumerate(layers):
        prefix = f'l{layer}_'
        normalized = rmsnorm(graph,prefix+'norm_attention',value,rows,p.hidden)
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
            pointwise(graph,prefix+'rope_k',new_keys,6)
        attended = compile_attention(graph,layout,attention_plan,query,cache,prefix+'attention')
        graph.release(temporary)
        projected = kernels['output'].compile(graph,attended)
        graph.release(attended)
        projected = residual(graph,prefix+'attention_residual',projected,value)
        graph.release(value,keep=projected); value = projected
        normalized = rmsnorm(graph,prefix+'norm_ffn',value,rows,p.hidden)
        expanded = kernels['expand'].compile(graph,normalized)
        graph.release(normalized)
        contracted = kernels['contract'].compile(graph,expanded)
        graph.release(expanded)
        contracted = residual(graph,prefix+'ffn_residual',contracted,value)
        graph.release(value,keep=contracted); value = contracted
    target = row_layout(graph,'output',rows,p.hidden)
    graph.convert('output_layout',value,target)
    graph.release(value,keep=target)
    return graph


def tune_native(workload, chip, out, workers=4):
    from frontend.auto.tiling_explorer_cloud import explore_gemm_comp
    from frontend.hardware_parser import load_chip_config_from_yaml
    from frontend.model_parser import Operator, OperatorType
    arch,capacity,_ = hardware(chip); side = isqrt(arch['core_num'])
    if side*side != arch['core_num'] or side % 2:
        raise ValueError('Native cloud baseline needs an even square mesh')
    cfg = yaml.safe_load(Path(chip).read_text()); single = deepcopy(cfg)
    single['architecture']['core_num'] = 1; single['architecture'].pop('noc',None)
    out.mkdir(parents=True,exist_ok=True)
    tuning = out/'single.yaml'; tuning.write_text(yaml.safe_dump(single))
    kernels = {}
    for name,problem in workload.problems.items():
        m,n,k = problem.shape
        if n % side or k % side:
            raise ValueError('Native K/N shape must divide mesh side')
        local = (m,n//side,k//side)
        a,b,c = matrix_placement(local,2,capacity)['tensor']
        op = Operator(name=name,op_type=OperatorType.GEMM,B=1,M=m,N=local[1],K=local[2])
        best = explore_gemm_comp(a,c,[b],op,2,load_chip_config_from_yaml(cfg),str(tuning),
                                8,512,8,workers,str(out/name))
        if best is None:
            raise ValueError('Native tile domain has no SRAM-feasible point')
        tm,tk,tn = best['tiling_factors']
        kernels[name] = Gemm(problem,Plan((1,side,side),(tm,tn,tk)),True,int(name=='expand'),name=='expand')
    return kernels
