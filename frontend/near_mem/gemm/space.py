"""The GEMM mapping space and its analytical proxy score.

``candidates`` lazily yields every legal Plan; divisibility, SRAM/DRAM capacity
and exact duplicates are removed, and a selector may ``skip`` whole subtrees.
``Proxy`` is a roofline-style ranking score (compute, DRAM, NoC, phase
overhead, input replication), never a cycle count: final choices are always
measured by ATLAS. ``Proxy.lower`` bounds that score from below over every
completion of a partial plan, so selectors prune without changing their result.
"""
from dataclasses import replace
from functools import lru_cache
from itertools import permutations, product
from math import prod, isqrt, ceil, log2, gcd
import re

import yaml

from ..sim import hardware

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


def candidates(problem,cores,capacity,tile_min=8,tile_policy='powers',layer_policy='elk',dram_capacity=float('inf'),
               rotation=True,skip=None,single_b=False):
    """Yields immutable Plans, one ``(ps,a,b,c)`` shell at a time.

    ``single_b`` keeps one DRAM copy of B (a graph's weights): an M split
    must then rotate B over its cores instead of replicating it.

    ``skip(prefix)`` is asked for every shell (``tile=None``) and every tile
    before loops, groups and layer orders are expanded; ``True`` drops all
    completions of that prefix.
    """
    for ps in spatial_parts(cores):
        if any(n%s for n,s in zip(problem.shape,ps)):continue
        periods=product(temporal_parts(ps[1]),temporal_parts(ps[0]),temporal_parts(ps[2])) if rotation else [((1,1),)*3]
        for a,b,c in periods:
            if single_b and prod(b)!=ps[0]:continue
            shell=Plan(ps,(1,1,1),a,b,c)
            try:
                shell.validate(problem)
            except ValueError:continue
            lm,ln,lk=(n//s for n,s in zip(problem.shape,ps))
            address=0
            for size in (lm*lk//prod(a),lk*ln//prod(b),lm*ln//prod(c)):
                address=(address+127)//128*128+size*problem.dtype_bytes
            if address>dram_capacity:continue
            if skip and skip(replace(shell,tile=None)):continue
            base=tuple(n//(s*t) for n,s,t in zip(problem.shape,ps,shell.periods))
            def tiles(n):
                all_tiles=[v for v in divisors(n) if v>=min(n,tile_min)]
                return all_tiles if tile_policy=='all' else [v for v in all_tiles if v&(v-1)==0 or v==n]
            for tile in product(*(tiles(n) for n in base)):
                basic=replace(shell,tile=tile)
                if resident_bytes(basic,problem.dtype_bytes)>capacity:continue
                if skip and skip(basic):continue
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
        """Standalone score: the kernel plus single-copy input replication."""
        return self.kernel(problem,p)+self.distribution(problem,p)

    def edge(self,load,dtype=2):
        """Score of a relayout from ``tasks.traffic``: its busiest core's DRAM or NoC port."""
        flit=self.arch['noc']['flit_size']
        return max((max((r+w)*dtype/self.dram_bw,s*dtype/flit,x*dtype/flit) for r,w,s,x in load.values()),default=0)

    def distribution(self,problem,p):
        # Initial single-copy distribution: each replica receives (r-1)/r.
        lm,ln,lk=(n//s for n,s in zip(problem.shape,p.ps))
        received=(lm*lk/prod(p.a)*(1-prod(p.a)/p.ps[1])+lk*ln/prod(p.b)*(1-prod(p.b)/p.ps[0]))*problem.dtype_bytes
        return max(received/self.arch['noc']['flit_size'],2*received/self.dram_bw)

    def kernel(self,problem,p):
        """Kernel score with inputs already in place (graph edges are scored apart)."""
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
        # A/B shifts are prefetched under compute by the compiler's pipeline.
        # A C shift waits for the panel's last update and the next compute
        # waits for it, and residual split-K runs after all compute: both are
        # serial, so they are added rather than overlapped.
        volume=0;serial=0;outer=1;events=0
        for i in (p.layer_order or tuple(range(len(p.layers)))):
            layer=p.layers[i]
            volume+=outer*(layer.radix-1)*sum(payload[v] for v in layer.tensors if v!='C')
            serial+=outer*(layer.radix-1)*payload['C']*('C' in layer.tensors)
            events+=outer*(layer.radix-1)
            outer*=layer.radix
        volume*=epochs;serial*=epochs
        residual=p.ps[2]//prod(p.c)
        serial+=lm*ln*dtype*2*(residual-1)/residual
        events=events*epochs+nm*nn*ceil(log2(residual))
        micro_steps=nm*nn*nk
        step=ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        mac=micro_steps*step
        dram=(reads_a+reads_b+c)/self.dram_bw
        noc=volume/self.arch['noc']['flit_size'];serial/=self.arch['noc']['flit_size']
        # Native tasks cannot issue fractional micro-iterations. Count their
        # phase overhead and one phase per shift or reduction step.
        # This is a ranking heuristic, never a safe latency lower bound.
        return max(mac,dram,noc)+serial+micro_steps+events


    def lower(self,problem,p):
        """Lower bound of ``kernel`` over all loops, groups and layer orders of
        ``p``; with ``p.tile=None`` also over all tiles.

        Per tensor, payload x epochs >= its local panel / temporal parts, and the
        layer order minimizing sum(outer*(radix-1)*weight) sorts layers by
        weight, largest first (exchange argument).
        """
        lm,ln,lk=local=tuple(n//s for n,s in zip(problem.shape,p.ps));dtype=problem.dtype_bytes
        weight={v:dtype*prod(local[d]/parts[i] for i,d in enumerate(DEPENDENCIES[v])) for v,parts in p.temporal.items()}
        def least(w):
            total=0;outer=1
            for layer in sorted(p.layers,key=w,reverse=True):total+=outer*(layer.radix-1)*w(layer);outer*=layer.radix
            return total
        volume=least(lambda l:sum(weight[v] for v in l.tensors if v!='C'))
        serial=least(lambda l:weight['C']*('C' in l.tensors))
        residual=p.ps[2]//prod(p.c)
        serial+=lm*ln*dtype*2*(residual-1)/residual
        if p.tile is None:
            micro_steps=rows=1;mac=lm*ln*lk/self.arch['core']['matrix']['mac_num']
        else:
            nm,nn,nk=(n//t for n,t in zip(local,p.tile));micro_steps=nm*nn*nk;rows=nm*nn
            mac=micro_steps*ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        events=prod(l.radix for l in p.layers)-1+rows*ceil(log2(residual))
        dram=(lm*lk/prod(p.a)+lk*ln/prod(p.b)+lm*ln/prod(p.c)*(1+2*(residual>1)))*dtype/self.dram_bw
        flit=self.arch['noc']['flit_size']
        return max(mac,dram,volume/flit)+serial/flit+micro_steps+events


def family(p):
    """Structural class kept by the shortlist: loop, rotation pattern, split-K."""
    return (p.loop,p.a[1]!=p.b[0],prod(p.c)>1,p.ps[2]>1)


def families(prefix):
    """Every ``family`` a completion of ``prefix`` can have (loops are chosen later)."""
    return {family(replace(prefix,loop=''.join(o))) for o in permutations('mnk')}


def structure(p):
    """Spatial and rotation decision of a plan, independent of tiles and loops."""
    return (p.ps,p.a,p.b,p.c)
