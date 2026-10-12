"""Lazy candidate space, replaceable proposal policy, proxy ranking, native rerank.

Only divisibility/capacity and exact duplicate elimination are safe pruning.
Tile sampling and top-k proxy selection are explicit heuristics. --audit-all
measures their ranking loss within the declared candidate domain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import heapq
from itertools import permutations, product
import json
from math import prod, isqrt, ceil, log2, gcd
from pathlib import Path
import re
import time

import yaml

from .backend import fingerprint, hardware, simulate
from .plan import Plan, Problem, DEPENDENCIES


@lru_cache(None)
def divisors(n):
    small=[i for i in range(1,isqrt(n)+1) if n%i==0]
    return tuple(sorted(set(small+[n//i for i in small])))


def spatial_parts(cores):
    for m in divisors(cores):
        for n in divisors(cores//m):yield m,n,cores//(m*n)


def temporal_parts(replicas):
    return tuple((a,b) for a in divisors(replicas) for b in divisors(replicas//a))


def groups(base,loop):
    seen=set();g=[1,1,1]
    for axis in reversed(loop):
        d='mnk'.index(axis)
        for n in divisors(base[d]):
            g[d]=n;key=tuple(g)
            if key not in seen:seen.add(key);yield key
        g[d]=base[d]


def resident_bytes(p,dtype=2):
    return (2*sum((count if prod(parts)>1 else 1)*prod(p.tile[d] for d in DEPENDENCIES[v])*dtype
                 for (v,parts),count in zip(p.temporal.items(),p.payload_tiles))
            + (2*p.tile[0]*p.tile[1]*dtype if prod(p.c)==1 else 0))


def candidates(problem,cores,capacity,tile_min=8,tile_policy='powers',layer_policy='elk',dram_capacity=float('inf'),rotation=True):
    """Proposal interface: yields immutable Plans; future annealing can replace it."""
    for ps in spatial_parts(cores):
        if any(n%s for n,s in zip(problem.shape,ps)):continue
        periods=product(temporal_parts(ps[1]),temporal_parts(ps[0]),temporal_parts(ps[2])) if rotation else [((1,1),)*3]
        for a,b,c in periods:
            shell=Plan(ps,(1,1,1),a,b,c)
            try:
                shell.validate(problem)
            except ValueError:continue
            lm,ln,lk=(n//s for n,s in zip(problem.shape,ps))
            address=0
            for size in (lm*lk//prod(a),lk*ln//prod(b),lm*ln//prod(c)):
                address=(address+127)//128*128+size*problem.dtype_bytes
            if address>dram_capacity:continue
            base=tuple(n//(s*t) for n,s,t in zip(problem.shape,ps,shell.periods))
            def tiles(n):
                all_tiles=[v for v in divisors(n) if v>=min(n,tile_min)]
                return all_tiles if tile_policy=='all' else [v for v in all_tiles if v&(v-1)==0 or v==n]
            for tile in product(*(tiles(n) for n in base)):
                basic=replace(shell,tile=tile)
                if resident_bytes(basic,problem.dtype_bytes)>capacity:continue
                counts=tuple(n//t for n,t in zip(base,tile))
                for loop in permutations('mnk'):
                    loop=''.join(loop)
                    for group in (groups(counts,loop) if shell.layers else [(1,1,1)]):
                        p=replace(basic,loop=loop,group=group)
                        if resident_bytes(p,problem.dtype_bytes)>capacity:continue
                        if layer_policy=='all':orders=p.orders()
                        else:
                            sizes={v:n*prod(p.tile[d] for d in DEPENDENCIES[v]) for v,n in zip(DEPENDENCIES,p.payload_tiles)}
                            orders=[tuple(sorted(range(len(p.layers)),key=lambda i:-sum(sizes[v] for v in p.layers[i].tensors)))]
                        for order in orders:yield replace(p,layer_order=order)


class Proxy:
    """Roofline/traffic proposal score only; never presented as measured cycles."""
    def __init__(self,chip):
        arch,_,paths=hardware(chip);self.arch=arch
        dram=yaml.safe_load(paths[1].read_text())['MemorySystem']['DRAM']
        pins=re.search(r'_(\d+)pin',dram['org']['preset'])
        rate=re.search(r'_(\d+)Mbps',dram['timing']['preset'])
        if not pins or not rate:raise ValueError('Proxy requires explicit HBDRAM pin/rate presets')
        self.dram_bw=int(pins[1])*int(rate[1])*dram['org']['channel']/8/arch['frequency']
        self.burst=dram['org'].get('channel_width',int(pins[1]))*dram['org'].get('internal_prefetch_size',1)//8

    @staticmethod
    @lru_cache(None)
    def line_factor(length,total,burst):
        count=total//length;period=burst//gcd(length,burst)
        amounts=[ceil(((i*length)%burst+length)/burst)*burst for i in range(period)]
        cycles,remainder=divmod(count,period)
        return (cycles*sum(amounts)+sum(amounts[:remainder]))/(count*length)

    def __call__(self,problem,p):
        lm,ln,lk=(n//s for n,s in zip(problem.shape,p.ps));tm,tn,tk=p.tile;dtype=problem.dtype_bytes
        nm,nn,nk=lm//tm,ln//tn,lk//tk;inner=p.loop[-1]
        reads_a=(nm*nk if inner=='n' else nm*nn*nk)*tm*tk*dtype/prod(p.a)
        reads_b=(nn*nk if inner=='m' else nm*nn*nk)*tk*tn*dtype/prod(p.b)
        if p.layers and p.b==(1,1) and p.c==(1,1):
            epochs=tuple(n//(s*t*g*period) for n,s,t,g,period in zip(problem.shape,p.ps,p.tile,p.group,p.periods))
            varying=[d for d in p.loop if epochs['mnk'.index(d)]>1]
            repeats=1 if not varying or varying[-1]=='n' else epochs[1]
            reads_a=lm*lk*dtype/prod(p.a)*repeats
        c=lm*ln*dtype*(1 if inner=='k' else 2*nk-1)/prod(p.c)
        # Explicit final materialization/reload plus coarse C-rotation spill
        # pressure. This remains a proxy: SRAM/DRAM banking and phase costs are
        # deliberately left to the independent backend rerank.
        if p.ps[2]//prod(p.c)>1:c+=2*lm*ln*dtype/prod(p.c)
        payload={v:n*prod(p.tile[d] for d in DEPENDENCIES[v])*dtype for v,n in zip(DEPENDENCIES,p.payload_tiles)}
        epochs=prod(n//(s*t*g*period) for n,s,t,g,period in zip(problem.shape,p.ps,p.tile,p.group,p.periods))
        volume=0;outer=1;events=0
        for i in (p.layer_order or tuple(range(len(p.layers)))):
            layer=p.layers[i]
            volume+=outer*(layer.radix-1)*sum(payload[v] for v in layer.tensors)
            events+=outer*(layer.radix-1)
            outer*=layer.radix
        volume*=epochs
        residual=p.ps[2]//prod(p.c)
        volume+=lm*ln*dtype*2*(residual-1)/residual
        events=events*epochs+nm*nn*ceil(log2(residual))
        micro_steps=nm*nn*nk
        mac=micro_steps*ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        dram=(reads_a+reads_b+c)/self.dram_bw
        noc=volume/self.arch['noc']['flit_size']
        # Native tasks cannot issue fractional micro-iterations. Count their
        # phase overhead and the credit/data phases of residual reduction.
        # This is a ranking heuristic, never a safe latency lower bound.
        return max(mac,dram,noc)+micro_steps+2*events


def shortlist(problem,plans,score,budget):
    heap=[];count=0;best_by_family={}
    for count,p in enumerate(plans,1):
        value=score(problem,p);row=(value,count,p)
        if len(heap)<budget:heapq.heappush(heap,(-value,-count,p))
        elif value < -heap[0][0]:heapq.heapreplace(heap,(-value,-count,p))
        # Preserve loop/rotation families, rather than let an approximate score
        # erase all unequal-period or C-rotation candidates from native rerank.
        family=(p.loop,p.a[1]!=p.b[0],prod(p.c)>1,p.ps[2]>1)
        if family not in best_by_family or value<best_by_family[family][0]:best_by_family[family]=row
    ranked=sorted([(-s,-i,p) for s,i,p in heap]+list(best_by_family.values()),key=lambda x:x[:2])
    unique={}
    for score_value,index,p in ranked:unique.setdefault(p,score_value)
    return list(unique.items()),count


def evaluate(args):
    problem,plan,chip,cache,provenance=args
    signature=dict(problem=asdict(problem),plan=asdict(plan),backend=provenance,overlap=True)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    out=cache/key
    if (out/'result.json').exists() and (out/'signature.json').exists():
        if json.loads((out/'signature.json').read_text())==json.loads(json.dumps(signature)):
            return dict(key=key,plan=asdict(plan),result=json.loads((out/'result.json').read_text()),cache_hit=True)
    run=simulate(problem,plan,chip,out,provenance=provenance)
    (out/'signature.json').write_text(json.dumps(signature,sort_keys=True))
    return dict(key=key,plan=asdict(plan),result=run['result'],cache_hit=False)


def main():
    ap=argparse.ArgumentParser()
    for name,value in [('m',128),('n',6144),('k',4096),('tile-min',16),('budget',16),('workers',4)]:
        ap.add_argument('--'+name,type=int,default=value)
    ap.add_argument('--chip',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--tile-policy',choices=['powers','all'],default='powers')
    ap.add_argument('--layer-policy',choices=['elk','all'],default='elk')
    ap.add_argument('--audit-all',action='store_true')
    args=ap.parse_args()
    if min(args.m,args.n,args.k,args.tile_min,args.budget,args.workers)<=0:ap.error('Counts must be positive')
    problem=Problem(args.m,args.n,args.k);arch,dram_capacity,_=hardware(args.chip)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    def domain():return candidates(problem,arch['core_num'],arch['core']['buffer']['buffer_size']*1024,args.tile_min,args.tile_policy,args.layer_policy,dram_capacity)
    start=time.monotonic();selected,count=shortlist(problem,domain(),Proxy(args.chip),args.budget)
    if not selected:raise ValueError('No legal resident-panel candidates')
    proposed={p for p,_ in selected}
    plans=list(dict.fromkeys(domain())) if args.audit_all else list(proposed)
    plans.sort(key=lambda p:json.dumps(asdict(p),sort_keys=True))
    generation=time.monotonic()-start;provenance=fingerprint(args.chip)
    (out/'search.json').write_text(json.dumps(dict(problem=asdict(problem),generated=count,selected=len(selected),evaluated=len(plans),generation_seconds=generation,
         tile_policy=args.tile_policy,layer_policy=args.layer_policy,shortlist=[dict(plan=asdict(p),score=s) for p,s in selected],
         pruning='divisibility/capacity exact; tile/layer sampling and proxy shortlist heuristic',sha256=provenance),indent=2))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(evaluate,[(problem,p,args.chip,out/'cache',provenance) for p in plans]):
            rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
            print(len(rows),len(plans),row['result']['e2e_cycles'],flush=True)
    best=min(rows,key=lambda r:r['result']['e2e_cycles'])
    summary=dict(best=best,candidates=count,evaluations=len(rows),elapsed_seconds=time.monotonic()-start,scope='single GEMM resident-panel space; not full-cloud speedup')
    if args.audit_all:
        keys={json.dumps(asdict(p),sort_keys=True) for p in proposed}
        short_best=min((r for r in rows if json.dumps(r['plan'],sort_keys=True) in keys),key=lambda r:r['result']['e2e_cycles'])
        summary.update(shortlist_best=short_best,shortlist_loss=short_best['result']['e2e_cycles']/best['result']['e2e_cycles']-1)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
