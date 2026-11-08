"""The GEMM mapping space and its analytical proxy score.

``candidates`` lazily yields every legal Plan; divisibility, SRAM/DRAM capacity
and exact duplicates are removed, and a selector may ``skip`` whole subtrees.
``Proxy`` is a roofline-style ranking score (compute, DRAM, NoC, phase
overhead, input replication), never a cycle count: final choices are always
measured by ATLAS. Its DRAM term charges the compiler's exact tile loads and
stores (``transfers``) through a calibrated channel model of the backend
(bursts and opened rows of the busiest channel, ``Proxy.access``). ``Proxy.lower`` bounds that score from below over every
completion of a partial plan, so selectors prune without changing their result.
"""
from dataclasses import replace
from functools import lru_cache
from itertools import permutations, product
from math import prod, isqrt, ceil, log2, gcd
import re

import yaml

from ..sim import hardware, strides

from .plan import Plan, Problem, DEPENDENCIES, execution_problem


LOOPS=tuple(''.join(o) for o in permutations('mnk'))


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
    return p.sram_bytes(dtype)


def tile_values(n,low,policy='all'):
    """Tile extents of a dimension ``n``: divisors of at least ``min(n, low)``
    (ATLAS's native rule); ``powers`` keeps powers of two and ``n``."""
    values=[v for v in divisors(n) if v>=min(n,low)]
    return values if policy=='all' else [v for v in values if v&(v-1)==0 or v==n]


def candidates(problem,cores,capacity,tile_min=8,tile_policy='all',layer_policy='elk',dram_capacity=float('inf'),
               rotation=True,skip=None,single_b=False,receive='sram_consume',layout='local',loops=LOOPS,parts=None,
               transpose=False):
    """Yields immutable Plans, one ``(ps,a,b,c)`` shell at a time.

    ``single_b`` keeps one DRAM copy of B (a graph's weights): an M split
    must then rotate B over its cores instead of replicating it.
    ``receive``/``layout`` select the baseline variants of ``Plan``: DRAM
    landing rotates only A/B in whole blocks; a global layout stores one
    plan-independent copy, so weights are never replicated. ``loops`` limits
    the micro loop orders; a single fixed loop also fixes the group to 1.
    ``tile_min`` is one minimum or per-axis ``(m, n, k)`` minima; ``parts``
    restricts the spatial partitions, numbered column-major if ``transpose``.

    ``skip(prefix)`` is asked for every shell (``tile=None``) and every tile
    before loops, groups and layer orders are expanded; ``True`` drops all
    completions of that prefix.
    """
    low=tile_min if isinstance(tile_min,tuple) else (tile_min,)*3
    for ps in parts or spatial_parts(cores):
        if any(n%s for n,s in zip(problem.shape,ps)):continue
        periods=product(temporal_parts(ps[1]),temporal_parts(ps[0]),temporal_parts(ps[2])) if rotation else [((1,1),)*3]
        for a,b,c in periods:
            if single_b and layout=='local' and prod(b)!=ps[0]:continue
            shell=Plan(ps,(1,1,1),a,b,c,receive=receive,layout=layout,transpose=transpose)
            try:
                shell.validate(problem)
            except ValueError:continue
            lm,ln,lk=(n//s for n,s in zip(problem.shape,ps))
            address=0;copies=3 if shell.land else 1
            for size in (lm*lk//prod(a)*copies,lk*ln//prod(b)*copies,lm*ln//prod(c)*(1+(shell.land and c!=(1,1)))):
                address=(address+127)//128*128+size*problem.dtype_bytes
            if address>dram_capacity:continue
            if skip and skip(replace(shell,tile=None)):continue
            base=tuple(n//(s*t) for n,s,t in zip(problem.shape,ps,shell.periods))
            if shell.land and not shell.layers:continue  # nothing rotates: a local plan
            for tile in product(*(tile_values(n,l,tile_policy) for n,l in zip(base,low))):
                basic=replace(shell,tile=tile)
                if resident_bytes(basic,problem.dtype_bytes)>capacity:continue
                if skip and skip(basic):continue
                counts=tuple(n//t for n,t in zip(base,tile))
                for loop in loops:
                    for group in (groups(counts,loop) if shell.layers and not shell.land and len(loops)>1 else [(1,1,1)]):
                        p=replace(basic,loop=loop,group=group)
                        if resident_bytes(p,problem.dtype_bytes)>capacity:continue
                        if layer_policy=='all':orders=p.orders()
                        else:
                            sizes={v:n*prod(p.tile[d] for d in DEPENDENCIES[v]) for v,n in zip(DEPENDENCIES,p.payload_tiles)}
                            orders=[tuple(sorted(range(len(p.layers)),key=lambda i:-sum(sizes[v] for v in p.layers[i].tensors)))]
                        for order in orders:yield replace(p,layer_order=order)


def _changes(extents,moves):
    """Key changes, the first access included, of a nested loop (outermost
    first) whose level ``i`` moves the key iff ``moves[i]``."""
    changes=1;outer=1
    for i,e in enumerate(extents):
        if e>1 and any(moves[j] and extents[j]>1 for j in range(i,len(extents))):changes+=outer*(e-1)
        outer*=e
    return changes


def transfers(problem,p):
    """Per-core DRAM tile loads and stores ``{tensor: (loads, stores)}`` of the
    main loop of ``compile_plan``, counted from the plan without compiling it.

    A tile (or a resident panel of ``payload`` tiles) is reloaded whenever the
    epoch / gcd-layer state / group loop nest moves its key; C's first touch
    is zeroed, not loaded. Split-K finalization is charged separately.
    """
    problem=execution_problem(problem,p)
    local=[n//s for n,s in zip(problem.shape,p.ps)];per=p.periods
    base=[l//(t*q) for l,t,q in zip(local,p.tile,per)]
    group=base if p.land else list(p.group)  # DRAM landing shifts whole blocks: one epoch
    order=['mnk'.index(c) for c in p.loop]
    epochs=[base[d]//group[d] for d in order];groups=[group[d] for d in order]
    layers=[p.layers[i] for i in (p.layer_order or range(len(p.layers)))]
    shifts=[];states=1
    for layer in layers:shifts.append(states*(layer.radix-1));states*=layer.radix
    out={}
    for v,deps in DEPENDENCIES.items():
        parts=p.temporal[v];distinct=prod(local[d]//p.tile[d] for d in deps)//prod(parts)
        epoch_changes=_changes(epochs,[d in deps for d in order])
        if prod(parts)>1 and not p.land:
            # A resident shifted panel is reloaded once per epoch-key change.
            payload=prod(per[d]//parts[i]*p.group[d] for i,d in enumerate(deps))
            changes=epoch_changes*payload
        else:
            inner=any(g>1 and d in deps for g,d in zip(groups,order))
            # A shift moves the key if the tensor depends on its axis (or the
            # group loop does anyway); a non-zero final state moves it again
            # at the next epoch.
            between=sum(n for n,l in zip(shifts,layers) if inner or l.axis in deps)
            reset=inner or any(n%l.radix and l.axis in deps for n,l in zip(shifts,layers))
            boundary=prod(epochs)-1 if reset else epoch_changes-1
            changes=1+prod(epochs)*(states*(_changes(groups,[d in deps for d in order])-1)+between)+boundary
        out[v]=(changes-distinct,changes) if v=='C' else (changes,0)
    return out


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
        # Address geometry of the channel model (``access``): channels, the
        # channel interleave bit above the burst offset, and bursts per row.
        mapper=yaml.safe_load(paths[1].read_text())['MemorySystem']['AddrMapper']
        self.channels=dram['org']['channel'];self.interleave=mapper.get('channel_lowest_bit',0)
        columns=re.search(r'_(\d+)col',dram['org']['preset'])
        self.columns=int(columns[1]) if columns else 128
        self.channel_bw=self.dram_bw/self.channels

    # Calibrated per DRAM timing preset with ``experiments.dram_calibration``
    # on single-bank HBDRAM channels (2-32 channels, 0.125-2 TB/s per core):
    # each row opened in the busiest channel costs ``ROW`` cycles beyond its
    # bursts, and a batch hides ``OVERLAP`` cycles of the first access.
    ROW,OVERLAP=37.1,17.2

    @lru_cache(None)
    def access(self,run,lines,pitch):
        """Proxy cycles of one batched DRAM access: ``lines`` runs of ``run``
        contiguous bytes, ``pitch`` bytes apart, from an interleave-aligned base.

        Requests are bursts; the busiest channel serves its bursts at the
        channel's share of the bandwidth and pays ``ROW`` per opened row
        (OneLevelInterleave mapping, open-row policy). Never below the bytes
        at peak bandwidth, so ``lower`` remains a bound.
        """
        bits=self.channels.bit_length()-1;low=self.interleave;bursts={};rows={}
        for line in range(lines):
            start=line*pitch
            for t in range(start//self.burst,(start+run+self.burst-1)//self.burst):
                channel=(t>>low)%self.channels;row=(((t>>(low+bits))<<low)|(t&((1<<low)-1)))//self.columns
                bursts[channel]=bursts.get(channel,0)+1;rows.setdefault(channel,set()).add(row)
        busiest=max(bursts[c]*self.burst/self.channel_bw+self.ROW*len(rows[c]) for c in bursts)
        return max(run*lines/self.dram_bw,busiest-self.OVERLAP)

    def tile(self,rows,cols,shape,column,dtype=2):
        """Proxy cycles of one ``rows x cols`` window of a 2-D DRAM tensor ``shape``
        stored row-major (column-major if ``column``) with the shared pitch rule."""
        pitch=max(strides(list(shape),dtype,column));run,lines=(rows,cols) if column else (cols,rows)
        if run==pitch:run,lines=run*lines,1  # whole rows of an unpadded tensor are one run
        return self.access(run*dtype,lines,pitch*dtype)

    def costs(self,problem,p):
        """Proxy cycles of one DRAM tile access per tensor of a plan with a tile
        (``problem`` already the execution problem): windows of the per-core
        A/C (row-major) and B (column-major) tensors; T10 reads windows of its
        fixed global blocks, one piece per home block a tile spans."""
        local=[n//s for n,s in zip(problem.shape,p.ps)]
        if p.layout=='global':
            from .layout import GlobalLayout
            shapes={v:GlobalLayout(problem,p.cores).block(v) for v in DEPENDENCIES}
        else:
            shapes={v:tuple(local[d]//parts[i] for i,d in enumerate(DEPENDENCIES[v])) for v,parts in p.temporal.items()}
        return {v:prod(ceil(p.tile[d]/n) for d,n in zip(axes,shapes[v]))
                  *self.tile(*(min(p.tile[d],n) for d,n in zip(axes,shapes[v])),shapes[v],v=='B',problem.dtype_bytes)
                for v,axes in DEPENDENCIES.items()}

    def __call__(self,problem,p):
        """Standalone score: the kernel plus single-copy input replication."""
        problem=execution_problem(problem,p)
        return self.kernel(problem,p)+self.distribution(problem,p)

    @property
    def mac(self):
        return self.arch['core']['matrix']['mac_num']

    def roofline(self,problem):
        """Whole-chip ideal: all MACs at peak, or every operand moved once at full DRAM bandwidth."""
        cores=self.arch['core_num'];m,n,k=problem.shape
        return max(m*n*k/(cores*self.mac),(m*k+k*n+m*n)*problem.dtype_bytes/(cores*self.dram_bw))

    def remote(self,p):
        """Share of a global-layout tile that lives on other cores (T10's NUMA fraction)."""
        return 1-1/p.cores if p.layout=='global' else 0

    @staticmethod
    @lru_cache(None)
    def hotspot(problem,shell):
        """Per input of a global-layout plan, the busiest home core's remote
        reads over the mean (>= 1), for the first compute state's tiles: when
        many readers want blocks of one home, its NoC port serializes them."""
        from .layout import GlobalLayout
        from .ownership import Ownership
        own=Ownership(problem,shell);layout=GlobalLayout(problem,shell.cores);state=(0,)*len(own.layers);out={}
        for v in 'AB':
            axes=DEPENDENCIES[v];load={};total=0
            for core in range(shell.cores):
                block=own.block(core,state)
                for home,_,extent in layout.pieces(v,*(block[d]*own.base[d] for d in axes),*(shell.tile[d] for d in axes)):
                    if home!=core:load[home]=load.get(home,0)+prod(extent);total+=prod(extent)
            out[v]=max(load.values())*shell.cores/total if total else 1
        return out

    def edge(self,phases,dtype=2):
        """Score of a relayout from ``tasks.traffic``: per phase, its busiest DRAM or NoC port or link."""
        flit=self.arch['noc']['flit_size']
        return sum(max((max((r+w)*dtype/self.dram_bw,s*dtype/flit,x*dtype/flit) for r,w,s,x in load.values()),default=0)
                   for load in phases)

    def distribution(self,problem,p):
        # Initial single-copy distribution: each replica receives (r-1)/r.
        problem=execution_problem(problem,p)
        if getattr(p,'kind',None)=='flash' or p.layout=='global':return 0
        lm,ln,lk=(n//s for n,s in zip(problem.shape,p.ps))
        received=(lm*lk/prod(p.a)*(1-prod(p.a)/p.ps[1])+lk*ln/prod(p.b)*(1-prod(p.b)/p.ps[0]))*problem.dtype_bytes
        return max(received/self.arch['noc']['flit_size'],2*received/self.dram_bw)

    def kernel(self,problem,p):
        """Kernel score with inputs already in place (graph edges are scored apart)."""
        problem=execution_problem(problem,p)
        if getattr(p,'kind',None)=='flash':
            from .flash import score
            return score(problem,p,self)
        lm,ln,lk=(n//s for n,s in zip(problem.shape,p.ps));tm,tn,tk=p.tile;dtype=problem.dtype_bytes
        nm,nn,nk=lm//tm,ln//tn,lk//tk
        # DRAM: the compiler's exact tile loads/stores, each charged by the
        # channel model for its window of the stored tensor (A/C row-major,
        # B column-major; T10 reads windows of its fixed global blocks).
        count=transfers(problem,p);residual=p.ps[2]//prod(p.c);cost=self.costs(problem,p)
        # Residual split-K reloads every partial C tile for the reduction; its
        # busiest rank then stores a 1/residual share of the outputs, or all of
        # them when the contract collects or replicates them.
        tiles=nm*nn//prod(p.c);final=tiles if p.output!='sharded' else ceil(tiles/residual)
        accesses=dict(A=count['A'][0],B=count['B'][0],C=sum(count['C'])+(tiles+final)*(residual>1))
        memory=sum(n*cost[v] for v,n in accesses.items())
        moved={v:n*prod(p.tile[d] for d in DEPENDENCIES[v])*dtype for v,n in accesses.items()}
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
        volume*=epochs;serial*=epochs;rotated_c=serial
        # Residual split-K: a line reduce-scatter per wave of ``residual`` C
        # tiles, r-1 tile hops each (both directions in parallel), r+1 phases.
        waves=ceil(max(1,nm*nn//prod(p.c))/residual)*(residual>1)
        serial+=waves*(residual-1)*tm*tn*dtype
        # DRAM-landing copies overlap compute and add no exchange phase.
        events=(0 if p.land else events*epochs)+waves*(residual+1)
        micro_steps=nm*nn*nk
        step=ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        mac=micro_steps*step
        # DRAM landing: every forwarded block is read by its sender and
        # written by its receiver, overlapped with compute (no shift phases).
        landed=2*(volume+rotated_c) if p.land else 0
        # Global layout: non-home tile reads and final writes cross the NoC.
        remote=self.remote(p)*(moved['A']+moved['B']+lm*ln*dtype/prod(p.c))
        if p.layout=='global':
            # Remote reads drain through their home cores' ports: the busiest
            # home serves ``hotspot`` times the mean (loops/groups do not change it).
            hot=self.hotspot(problem,replace(p,loop='mnk',group=(1,1,1),layer_order=()))
            remote=self.remote(p)*(moved['A']*hot['A']+moved['B']*hot['B']+lm*ln*dtype/prod(p.c))
        dram=memory+landed/self.dram_bw
        noc=(volume+remote)/self.arch['noc']['flit_size'];serial/=self.arch['noc']['flit_size']
        # Native tasks cannot issue fractional micro-iterations. Count their
        # phase overhead and one phase per shift or reduction step.
        # This is a ranking heuristic, never a safe latency lower bound.
        return max(mac,dram,noc)+serial+micro_steps+events


    def lower(self,problem,p):
        """Lower bound of ``kernel`` over all loops, groups and layer orders of
        ``p``; with ``p.tile=None`` also over all tiles. FlashGEMM spaces are
        small and scanned without pruning, so their bound is the score.

        Per tensor, payload x epochs >= its local panel / temporal parts, and the
        layer order minimizing sum(outer*(radix-1)*weight) sorts layers by
        weight, largest first (exchange argument).
        """
        problem=execution_problem(problem,p)
        if getattr(p,'kind',None)=='flash':return self.kernel(problem,p)
        lm,ln,lk=local=tuple(n//s for n,s in zip(problem.shape,p.ps));dtype=problem.dtype_bytes
        weight={v:dtype*prod(local[d]/parts[i] for i,d in enumerate(DEPENDENCIES[v])) for v,parts in p.temporal.items()}
        def least(w):
            total=0;outer=1
            for layer in sorted(p.layers,key=w,reverse=True):total+=outer*(layer.radix-1)*w(layer);outer*=layer.radix
            return total
        volume=least(lambda l:sum(weight[v] for v in l.tensors if v!='C'))
        serial=least(lambda l:weight['C']*('C' in l.tensors));rotated_c=serial
        residual=p.ps[2]//prod(p.c)
        serial+=lm*ln/prod(p.c)*dtype*(residual-1)/residual
        if p.tile is None:
            micro_steps=rows=1;mac=lm*ln*lk/self.arch['core']['matrix']['mac_num']
        else:
            nm,nn,nk=(n//t for n,t in zip(local,p.tile));micro_steps=nm*nn*nk;rows=nm*nn
            mac=micro_steps*ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        waves=ceil(max(1,rows//prod(p.c))/residual)*(residual>1)
        events=(0 if p.land else prod(l.radix for l in p.layers)-1)+waves*(residual+1)
        a=lm*lk/(1 if p.land else prod(p.a));b=lk*ln/(1 if p.land else prod(p.b))
        c=lm*ln/prod(p.c)*(1+(residual>1)/residual)  # stores, plus a split-K reload and final share
        if p.tile is None or p.land:
            memory=(a+b+c)*dtype/self.dram_bw  # landed blocks move between DRAM slots
        else:
            # Every A/B tile is loaded and every C tile stored at least once,
            # each at its channel cost (never below its bytes at peak bandwidth).
            cost=self.costs(problem,p);tm,tn,tk=p.tile
            memory=a/(tm*tk)*cost['A']+b/(tk*tn)*cost['B']+c/(tm*tn)*cost['C']
        dram=memory+2*(volume+rotated_c)*p.land/self.dram_bw  # volumes are in bytes
        remote=self.remote(p)*(lm*lk/prod(p.a)+lk*ln/prod(p.b)+lm*ln/prod(p.c))*dtype
        flit=self.arch['noc']['flit_size']
        return max(mac,dram,(volume+remote)/flit)+serial/flit+micro_steps+events


def family(p):
    """Structural class kept by the shortlist: loop, rotation pattern, split-K."""
    if getattr(p,'kind',None)=='flash':return (p.dataflow,p.b_resident)
    return (p.loop,p.a[1]!=p.b[0],prod(p.c)>1,p.ps[2]>1)


def families(prefix,loops=LOOPS):
    """Every ``family`` a completion of ``prefix`` can have (loops are chosen later)."""
    if getattr(prefix,'kind',None)=='flash':return {family(prefix)}
    return {family(replace(prefix,loop=o)) for o in loops}


def structure(p):
    """Spatial (incl. mesh orientation) and rotation decision of a plan, independent of tiles and loops."""
    if getattr(p,'kind',None)=='flash':return (p.dataflow,p.b_resident)
    return (p.ps,p.transpose,p.a,p.b,p.c)
