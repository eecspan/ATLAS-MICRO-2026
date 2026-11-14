"""The GEMM mapping space and its analytical proxy score.

``candidates`` lazily yields every legal Plan; divisibility, SRAM/DRAM capacity
and exact duplicates are removed, and a selector may ``skip`` whole subtrees.
``Proxy`` is a roofline-style ranking score (compute, DRAM, NoC, phase
overhead, input replication), never a cycle count: final choices are always
measured by ATLAS. Its DRAM term charges the compiler's exact tile loads and
stores (``transfers``) through a calibrated channel model of the backend
(bursts and opened rows of the busiest channel, ``Proxy.access``), each tile
at the burst offsets where the compiler places it. Its NoC
term is the maximum channel load of the compiler's transfers under BookSim's
XY routing (busiest injection/ejection port or directed link, ``gemm.noc``),
so ring hops that span several mesh links and colliding routes are charged.
Overlap is per step where the compiler fixes it: each shift against the
micro-steps that hide it, and the pipeline's first loads and last store,
which nothing hides. ``model.schedule_proxy`` re-ranks a shortlist on the
compiled schedule. ``Proxy.lower`` bounds that score from below over every completion of a
partial plan, so selectors prune without changing their result.
"""
from dataclasses import replace
from functools import lru_cache
from itertools import permutations, product
from math import prod, isqrt, ceil, log2, gcd
import re

import yaml

from ..sim import hardware, strides

from .plan import Plan, Problem, DEPENDENCIES, execution_problem
from . import noc as channel


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


def nest(loop,extents):
    """The loop order restricted to axes of more than one iteration."""
    return ''.join(a for a in loop if extents['mnk'.index(a)]>1)


def schedule_key(plan,counts):
    """What the compiler's loop nests depend on, given the per-axis tile counts
    ``counts`` of one epoch-group block: the group, and the order of the epoch
    and micro loops that iterate more than once. ``loop`` is used only to order
    those nests (``compiler.loop_points``, ``transfers``); a one-iteration loop
    contributes a single index wherever it is placed, so plans with equal keys
    compile to the same program."""
    group=counts if plan.land else plan.group  # DRAM landing: one epoch
    epochs=tuple(n//g for n,g in zip(counts,group))
    return plan.group,nest(plan.loop,epochs),nest(plan.loop,group)


def candidates(problem,cores,capacity,tile_min=8,tile_policy='all',layer_policy='elk',dram_capacity=float('inf'),
               rotation=True,skip=None,single_b=False,receive='sram_consume',layout='local',loops=LOOPS,parts=None,
               transpose=False,distinct=True):
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

    With ``distinct`` loop orders that differ only in where one-iteration
    loops sit are yielded once (the first in ``loops`` order): they compile
    to the same program (``schedule_key``). ``distinct=False`` yields them all.
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
                seen=set();fits={}
                for loop in loops:
                    for group in (groups(counts,loop) if shell.layers and not shell.land and len(loops)>1 else [(1,1,1)]):
                        p=replace(basic,loop=loop,group=group)
                        if distinct:
                            key=schedule_key(p,counts)
                            if key in seen:continue
                            seen.add(key)
                        # Live SRAM depends on the group, not on the loop order.
                        if group not in fits:fits[group]=resident_bytes(p,problem.dtype_bytes)<=capacity
                        if not fits[group]:continue
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


@lru_cache(None)
def offsets(rows,row_step,cols,col_step,burst):
    """``((offset, count), ...)``: offsets within a burst of the windows
    ``i*row_step + j*col_step`` (bytes) for ``i < rows``, ``j < cols``."""
    def residues(n,step):
        period=burst//gcd(step,burst);counts={}
        for i in range(min(n,period)):
            r=i*step%burst;counts[r]=counts.get(r,0)+n//period+(i<n%period)
        return counts
    out={}
    for a,x in residues(rows,row_step).items():
        for b,y in residues(cols,col_step).items():
            out[(a+b)%burst]=out.get((a+b)%burst,0)+x*y
    return tuple(sorted(out.items()))


@lru_cache(None)
def hbdram_columns(preset):
    """Bursts per DRAM row of an HBDRAM organization preset, read from the
    simulator's own preset table (names do not always encode it)."""
    from ..sim import ROOT
    source=(ROOT/'simulator/src/dram/ramulator2/src/dram/impl/HB-DRAM.cpp').read_text()
    found=re.search(r'\{"'+re.escape(preset)+r'",\s*\{[^{]*\{([^}]*)\}',source)
    if not found:raise ValueError(f'Unknown HBDRAM preset {preset}')
    levels=[eval(x) for x in found[1].split(',')]  # channel, bankgroup, bank, row, column
    if levels[1]*levels[2]!=1:raise ValueError('The DRAM channel model assumes one bank per channel')
    return levels[4]


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
        if mapper.get('impl')!='OneLevelInterleave':raise ValueError('The DRAM channel model assumes OneLevelInterleave')
        self.channels=dram['org']['channel'];self.interleave=mapper.get('channel_lowest_bit',0)
        self.columns=dram['org'].get('column') or hbdram_columns(dram['org']['preset'])
        self.channel_bw=self.dram_bw/self.channels
        self.timing=dram['timing']['preset']

    @property
    def row_cost(self):
        return self.calibration[0]

    @property
    def overlap(self):
        return self.calibration[1]

    @property
    def calibration(self):
        if self.timing not in self.CALIBRATION:
            raise ValueError(f'Calibrate the DRAM channel model for {self.timing} (experiments.dram_calibration)')
        return self.CALIBRATION[self.timing]

    # ``(ROW, OVERLAP)`` per DRAM timing preset, fitted by
    # ``experiments.dram_calibration`` on single-bank HBDRAM channels (2-32
    # channels, 0.125-2 TB/s per core): each row opened in the busiest channel
    # costs ROW cycles (at 1 GHz) beyond its bursts, and a batch hides OVERLAP
    # cycles of the first access. Channel count, bandwidth, interleave and row
    # width come from the configuration and need no recalibration.
    CALIBRATION={'HBDRAM_500Mbps':(37.1,17.2)}

    @lru_cache(None)
    def access(self,run,lines,pitch,start=0):
        """Proxy cycles of one batched DRAM access: ``lines`` runs of ``run``
        contiguous bytes, ``pitch`` bytes apart, from ``start`` bytes past an
        interleave-aligned base.

        Requests are bursts; the busiest channel serves its bursts at the
        channel's share of the bandwidth and pays ``row_cost`` per opened row
        (OneLevelInterleave mapping, open-row policy). Never below the bytes
        at peak bandwidth, so ``lower`` remains a bound.
        """
        bits=self.channels.bit_length()-1;low=self.interleave;bursts={};rows={}
        for line in range(lines):
            line_start=start+line*pitch
            for t in range(line_start//self.burst,(line_start+run+self.burst-1)//self.burst):
                channel=(t>>low)%self.channels;row=(((t>>(low+bits))<<low)|(t&((1<<low)-1)))//self.columns
                bursts[channel]=bursts.get(channel,0)+1;rows.setdefault(channel,set()).add(row)
        busiest=max(bursts[c]*self.burst/self.channel_bw+self.row_cost*len(rows[c]) for c in bursts)
        return max(run*lines/self.dram_bw,busiest-self.overlap)

    def tile(self,rows,cols,shape,column,dtype=2):
        """Mean proxy cycles of a ``rows x cols`` window of a 2-D DRAM tensor
        ``shape`` stored row-major (column-major if ``column``) with the shared
        pitch rule, over every window position of the tensor: a window whose
        byte offset is not a burst multiple touches one more burst per line,
        so each position is charged from its offset within a burst."""
        pitch=max(strides(list(shape),dtype,column));run,lines=(rows,cols) if column else (cols,rows)
        extent,depth=(shape[0],shape[1]) if column else (shape[1],shape[0])
        if run==pitch:return self.access(run*lines*dtype,1,pitch*dtype)  # whole rows of an unpadded tensor are one run
        starts=offsets(max(1,depth//lines),lines*pitch*dtype,max(1,extent//run),run*dtype,self.burst)
        return sum(n*self.access(run*dtype,lines,pitch*dtype,s) for s,n in starts)/sum(n for _,n in starts)

    def costs(self,problem,p):
        """Proxy cycles of one DRAM tile access per tensor of a plan with a tile
        (``problem`` already the execution problem): windows of the per-core
        A/C (row-major) and B (column-major) tensors; T10 reads windows of its
        fixed global blocks, one piece per home block a tile spans. Depends
        only on the tensor windows, so loop orders, groups and layer orders of
        one tile share one evaluation."""
        return self._costs(problem,p.ps,p.tile,p.a,p.b,p.c,p.layout)

    @lru_cache(maxsize=1<<16)
    def _costs(self,problem,ps,tile,a,b,c,layout):
        local=[n//s for n,s in zip(problem.shape,ps)];temporal={'A':a,'B':b,'C':c}
        if layout=='global':
            from .layout import GlobalLayout
            shapes={v:GlobalLayout(problem,prod(ps)).block(v) for v in DEPENDENCIES}
        else:
            shapes={v:tuple(local[d]//parts[i] for i,d in enumerate(DEPENDENCIES[v])) for v,parts in temporal.items()}
        return {v:prod(ceil(tile[d]/n) for d,n in zip(axes,shapes[v]))
                  *self.tile(*(min(tile[d],n) for d,n in zip(axes,shapes[v])),shapes[v],v=='B',problem.dtype_bytes)
                for v,axes in DEPENDENCIES.items()}

    def __call__(self,problem,p):
        """Standalone score: the kernel plus single-copy input replication."""
        problem=execution_problem(problem,p)
        return self.kernel(problem,p)+self.distribution(problem,p)

    @property
    def sram(self):
        return self.arch['core']['buffer']['buffer_size']*1024

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
        many readers want blocks of one home, its NoC port serializes them,
        and XY routes from the homes to their readers share mesh links."""
        from .layout import GlobalLayout
        from .ownership import Ownership
        own=Ownership(problem,shell);layout=GlobalLayout(problem,shell.cores);state=(0,)*len(own.layers);out={}
        for v in 'AB':
            axes=DEPENDENCIES[v];flows=[]
            for core in range(shell.cores):
                block=own.block(core,state)
                for home,_,extent in layout.pieces(v,*(block[d]*own.base[d] for d in axes),*(shell.tile[d] for d in axes)):
                    if home!=core:flows.append((home,core,prod(extent)))
            total=sum(n for _,_,n in flows);load=channel.load(flows,channel.mesh(shell.cores))
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
        if not received:return 0
        # The linear all-gather's busiest port or link, hop by hop (a middle
        # replica receives from both sides); the receive volume if the groups
        # are no mesh sub-grids.
        busiest=channel.gather(problem,channel.shell(p))
        busiest=received if busiest is None else max(received,busiest*problem.dtype_bytes)
        return max(busiest/self.arch['noc']['flit_size'],2*received/self.dram_bw)

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
        # NoC volumes are bytes on the busiest port or XY link of each shift
        # phase (``gemm.noc``); ``sent``/``rotated_c`` are the bytes every core sends.
        # The compiler moves a layer's A/B and C in separate phases. An A/B
        # shift leaves in equal shares with every micro-step of the state
        # before it: it overlaps the state's compute and the rest of it
        # (``exposed``) stalls the next; a phase of ATLAS lasts until its
        # slowest core and packet finish. A waiting C shift (``c_accumulate``)
        # leaves after the panel's last update and the next compute waits for
        # it: serial. A deferred one travels with the next state's first
        # micro-step, like an A/B shift. DRAM landing forwards A/B blocks
        # under compute; only C copies serialize.
        step=ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
        flit=self.arch['noc']['flit_size'];defer=p.defers(self.sram,dtype)
        volume=0;serial=0;outer=1;events=0;sent=0;rotated_c=0;exposed=0
        order=p.layer_order or tuple(range(len(p.layers)))
        shape=channel.shell(p);loads=channel.rotation(shape,channel.head(p.layers,order)) if p.layers else ()
        for i in order:
            layer=p.layers[i];shifts=outer*(layer.radix-1);front=loads[i][0]
            ab=channel.peak(front,dict(payload,C=0));c=channel.peak(front,dict(payload,A=0,B=0))
            volume+=shifts*ab
            # Spread over the state's micro-steps when there are several (``compiler``).
            if not p.land:exposed+=shifts*max(0,ab/flit-step*prod(p.group))
            if defer:
                volume+=shifts*c;exposed+=shifts*max(0,c/flit-step)
            else:serial+=shifts*c
            sent+=shifts*sum(payload[v] for v in layer.tensors if v!='C')
            rotated_c+=shifts*payload['C']*('C' in layer.tensors)
            events+=shifts
            outer*=layer.radix
        volume*=epochs;serial*=epochs;rotated_c*=epochs;sent*=epochs;exposed*=epochs
        # Residual split-K: a line reduce-scatter per wave of ``residual`` C
        # tiles, r-1 tile steps each at the busiest port/link load of a step
        # (a middle rank sends to and receives from both neighbours), r+1 phases.
        waves=ceil(max(1,nm*nn//prod(p.c))/residual)*(residual>1)
        serial+=waves*(residual-1)*tm*tn*dtype*(channel.reduction(shape) if residual>1 else 0)
        # DRAM-landing copies overlap compute and add no exchange phase.
        events=(0 if p.land else events*epochs)+waves*(residual+1)
        micro_steps=nm*nn*nk
        mac=micro_steps*step
        # DRAM landing: every forwarded block is read by its sender and
        # written by its receiver, overlapped with compute (no shift phases).
        landed=2*(sent+rotated_c) if p.land else 0
        # Global layout: non-home tile reads and final writes cross the NoC.
        remote=self.remote(p)*(moved['A']+moved['B']+lm*ln*dtype/prod(p.c))
        if p.layout=='global':
            # Remote reads drain through their home cores' ports: the busiest
            # home serves ``hotspot`` times the mean (loops/groups do not change it).
            hot=self.hotspot(problem,replace(p,loop='mnk',group=(1,1,1),layer_order=()))
            remote=self.remote(p)*(moved['A']*hot['A']+moved['B']*hot['B']+lm*ln*dtype/prod(p.c))
        dram=memory+landed/self.dram_bw
        noc=(volume+remote)/flit;serial/=flit
        # Pipeline prologue and epilogue: the first compute waits for its
        # operands (a rotating tensor's resident panel, else one tile; C
        # starts zeroed) and the last C write follows the last compute, so
        # neither hides under compute; a DRAM-bound pipeline ends one step
        # after its last access.
        shifted={v:prod(parts)>1 and not p.land for v,parts in p.temporal.items()}
        resident={v:(n if shifted[v] else 1)*cost[v] for v,n in zip(DEPENDENCIES,p.payload_tiles)}
        ends=resident['A']+resident['B']+resident['C']
        # Native tasks cannot issue fractional micro-iterations. Count their
        # phase overhead and one phase per shift or reduction step.
        # This is a ranking heuristic, never a safe latency lower bound.
        return max(mac+exposed+ends,dram+step,noc)+serial+micro_steps+events


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
        sent=least(lambda l:sum(weight[v] for v in l.tensors if v!='C'))
        rotated_c=least(lambda l:weight['C']*('C' in l.tensors))
        # Channel loads depend on the layer order only through its head
        # (``noc.head``); the least volume over every possible head bounds all
        # orders, and ``noc.peak`` is monotone and homogeneous in the per-core
        # volumes. Stalls of shifts beyond their step (>= 0) are omitted. A C
        # shift counts as overlapped unless the plan cannot defer it (a
        # prefix without tiles may still defer; overlapping never costs more).
        shape=channel.shell(p);volume=serial=float('inf')
        defer=(p.defers(self.sram,dtype) if p.tile is not None else
               prod(p.c)>1 and not p.land and p.c_accumulate!='wait')
        for pair in channel.heads(p.layers):
            loads=channel.rotation(shape,pair) if p.layers else ()
            # Per layer: (overlapped, serial) busiest volumes, as ``kernel`` splits them.
            def split(front,l):
                ab=channel.peak(front,dict(weight,C=0));c=channel.peak(front,dict(weight,A=0,B=0))
                return (ab+c,0) if defer else (ab,c)
            row={l:split(loads[i][0],l) for i,l in enumerate(p.layers)}
            volume=min(volume,least(lambda l:row[l][0]))
            serial=min(serial,least(lambda l:row[l][1]))
        residual=p.ps[2]//prod(p.c)
        if residual>1:serial+=lm*ln/prod(p.c)*dtype*(residual-1)/residual*channel.reduction(shape)
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
        dram=memory+2*(sent+rotated_c)*p.land/self.dram_bw  # volumes are in bytes
        remote=self.remote(p)*(lm*lk/prod(p.a)+lk*ln/prod(p.b)+lm*ln/prod(p.c))*dtype
        flit=self.arch['noc']['flit_size']
        ends=step=0
        if p.tile is not None:
            # The prologue/epilogue of ``kernel`` at the least resident panels
            # (groups only enlarge them) and the DRAM-bound pipeline's last step.
            step=ceil(prod(p.tile)/self.arch['core']['matrix']['mac_num'])
            cost=self.costs(problem,p)
            ends=sum((prod(per//parts[i] for i,per in enumerate(p.periods[d] for d in DEPENDENCIES[v]))
                      if prod(parts)>1 and not p.land else 1)*cost[v] for v,parts in p.temporal.items())
        return max(mac+ends,dram+step,(volume+remote)/flit)+serial/flit+micro_steps+events


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
