"""General multi-axis compiler with explicit ownership and live tensor panels.

Full resident shifted panels are supported; partial-panel spilling is
rejected, not approximated by analytical traffic. Six micro loop orders and
saturate-inner groups lower into actual loads/RMW/reuse. Two baseline
variants share this compiler: ``receive='dram_land'`` forwards each rotated
block DRAM to DRAM (Ring-Attention style), and ``layout='global'`` keeps
A/B/C in one plan-independent layout read remotely (T10 style).
"""
from itertools import product
from math import prod
from copy import deepcopy

from .layout import GlobalLayout
from .ownership import Ownership
from .plan import DEPENDENCIES
from .program import Program
from ..sim import span, strides
from ..output import reduce_steps, broadcast_steps, root_index


def compile_kernel(problem, plan, sram_bytes, dram_bytes):
    """The executable program of any GEMM plan: ``regions``, ``placements``, ``lower``."""
    if getattr(plan,'kind',None)=='flash':
        from .flash import FlashProgram
        return FlashProgram(problem,plan,sram_bytes,dram_bytes)
    return compile_plan(problem,plan,sram_bytes,dram_bytes)


def compile_plan(problem, plan, sram_bytes, dram_bytes, overlap=True):
    own=Ownership(problem,plan)
    p=plan;core_count=p.cores
    shifted={v:prod(parts)>1 for v,parts in p.temporal.items()}
    land={v:shifted[v] and p.receive=='dram_land' for v in DEPENDENCIES}
    counts={v:payload if shifted[v] and not land[v] else 1 for v,payload in zip(DEPENDENCIES,p.payload_tiles)}
    if p.panel is not None and tuple(counts.values())!=p.panel:
        raise ValueError('Partial-panel spilling requires a separate explicit spill schedule')
    tile_shapes={v:tuple(p.tile[d] for d in axes) for v,axes in DEPENDENCIES.items()}
    tile_bytes={v:prod(shape)*problem.dtype_bytes for v,shape in tile_shapes.items()}
    needed=p.sram_bytes(problem.dtype_bytes,overlap)
    if needed>sram_bytes:
        raise ValueError(f'SRAM capacity exceeded: {needed} > {sram_bytes}')
    glob=GlobalLayout(problem,core_count) if p.layout=='global' else None
    base_tiles=tuple(b//t for b,t in zip(own.base,p.tile))
    group=base_tiles if p.land else p.group  # DRAM landing shifts whole blocks: one epoch
    placements=[];address_maps=[]
    for core in range(core_count):
        tensors=[];maps={};addr=0
        def allocate(name,shape,column=False):
            nonlocal addr
            addr=(addr+127)//128*128
            t=dict(name=name,base_addr=addr,element_size=problem.dtype_bytes,shape=list(shape),
                   strides=strides(shape,problem.dtype_bytes,column))
            tensors.append(t);addr+=span(shape,t['strides'],problem.dtype_bytes)
        for v,axes in DEPENDENCIES.items():
            blocks=own.tensor_blocks(core,v)
            axis_blocks=[sorted({key[i] for key in blocks}) for i in range(2)]
            shape=[len(keys)*own.base[d] for keys,d in zip(axis_blocks,axes)]
            maps[v]=(axis_blocks,shape)
            if glob and v!='C':
                allocate(v,glob.block(v),v=='B')
                continue
            allocate(v,shape,v=='B')  # under a global layout, C is the local partial-sum scratch
            for slot in (('1',) if v=='C' else ('1','2')) if land[v] else ():
                allocate(v+slot,shape,v=='B')
        if glob:allocate('Cg',glob.block('C'))  # final outputs, written to their homes
        if addr>dram_bytes:raise ValueError('Aligned DRAM placement exceeds capacity')
        placements.append(dict(core_id=core,tensor=tensors));address_maps.append(maps)
    phases=[]
    cache=[{v:set() for v in DEPENDENCIES} for _ in range(core_count)]
    materialized=[set() for _ in range(core_count)]
    # DRAM landing: the DRAM slot holding each core's current rotated block,
    # and the core whose original layout that block keeps (copies are verbatim).
    # A landed C alternates between slots C and C1 and finally returns home
    # (one epoch: a whole number of turns per layer), so it starts in the
    # slot that makes its last one C, its output tensor.
    shifts=[0]*len(own.layers)
    for event,arg in own.events():
        if event=='shift':shifts[arg]+=1
    c_moves=sum(n+(-n)%l.radix for n,l in zip(shifts,own.layers) if 'C' in l.tensors)
    holder=[{v:('C1' if v=='C' and c_moves%2 else v,core) for v in DEPENDENCIES} for core in range(core_count)]
    forwards=[]

    def done(core):
        """C tiles already materialized in the DRAM block ``core`` now holds."""
        return materialized[holder[core]['C'][1] if land['C'] else core]

    def phase():
        actions=[[] for _ in range(core_count)];phases.append(actions);return actions

    def access(core,v,key,write=False,final=False,step=None):
        item=dict(op='store' if write else 'load',tensor=v,key=key,shape=tile_shapes[v],final=final)
        if p.land:item['step']=step  # places DRAM-landing copies
        if glob and v!='C':
            return item  # lowering reads the tile from its home cores
        name,origin=holder[core][v];axes=DEPENDENCIES[v];blocks,_=address_maps[origin][v]
        if any(key[i]//base_tiles[d] not in blocks[i] for i,d in enumerate(axes)):
            raise AssertionError(f'{v} tile {key} is not in DRAM slot {name} of core {core}')
        base=[(blocks[i].index(key[i]//base_tiles[d])*base_tiles[d]+key[i]%base_tiles[d])*p.tile[d]
              for i,d in enumerate(axes)]
        item.update(base=base,extent=list(tile_shapes[v]))
        if name!=v:item['name']=name  # a DRAM-landing slot
        return item

    def save_c(row,core,final=False,step=None):
        for key in sorted(cache[core]['C']):
            row[core].append(access(core,'C',key,True,final,step));done(core).add(key)

    def prepare(row,core,v,keys,step=None):
        keys=set(keys)
        if keys==cache[core][v]:return
        row[core].append(dict(op='clear',tensor=v))
        cache[core][v]=keys
        for key in sorted(keys):
            if v=='C' and key not in done(core):
                row[core].append(dict(op='zero',tensor='C',key=key,shape=tile_shapes[v]))
            else:row[core].append(access(core,v,key,step=step))

    def exchange(layer_id, tensors, direction=-1, step=None):
        # Like native ATLAS collectives, the arrival of a step's data is the
        # neighbour synchronisation; no separate ready-credit phase is issued.
        if land['C'] and 'C' in tensors:
            # A landed C is copied from DRAM: write back its live tiles first.
            flush=phase()
            for core in range(core_count):
                save_c(flush,core,step=step);cache[core]['C']=set();flush[core].append(dict(op='clear',tensor='C'))
        data=phase()
        for v in tensors:
            if land[v] and v=='C':
                # C is final only after the step's last update: copy it serially,
                # one row after its store lands; the next load waits one row.
                data[0].append(dict(op='nop'));copies=phase();phase()[0].append(dict(op='nop'))
                for core,ops in enumerate(forward(layer_id,v,direction)):copies[core].extend(ops)
                continue
            if land[v]:
                forwards.append((step,v,forward(layer_id,v,direction)))
                continue
            old=[dict(c) for c in cache]
            for src in range(core_count):
                dst=own.neighbor(src,layer_id,v,direction)
                data[src].append(dict(op='send',dst=dst,tensor=v,byte_count=len(old[src][v])*tile_bytes[v]))
                cache[dst][v]=old[src][v]

    def forward(layer_id,v,direction):
        """Tile-sized copies of every core's DRAM slot into its neighbour's other slot."""
        old=[dict(h) for h in holder];chunks=[]
        for src in range(core_count):
            dst=own.neighbor(src,layer_id,v,direction)
            name,origin=old[src][v]
            target=('C1' if name=='C' else 'C') if v=='C' else v+('2' if name==v+'1' else '1')
            holder[dst][v]=(target,origin)
            (blocks,shape),axes=address_maps[origin][v],DEPENDENCIES[v];th,tw=tile_shapes[v]
            # Global tile index of a slot offset (replays C's moves exactly).
            key=lambda x,i:(blocks[i][x//own.base[axes[i]]]*own.base[axes[i]]+x%own.base[axes[i]])//p.tile[axes[i]]
            chunks.append([dict(op='forward',tensor=v,name=name,dst=dst,dst_name=target,base=[r,c],
                                extent=list(tile_shapes[v]),byte_count=tile_bytes[v],key=(key(r,0),key(c,1)))
                           for r in range(0,shape[0],th) for c in range(0,shape[1],tw)
                           if v!='C' or (key(r,0),key(c,1)) in materialized[origin]])  # C: only written tiles
        return chunks

    def loop_points(extents):
        order=tuple('mnk'.index(c) for c in p.loop)
        for values in product(*(range(extents[d]) for d in order)):
            point=[0,0,0]
            for d,value in zip(order,values):point[d]=value
            yield tuple(point)

    def reduce_scatter(rows,cores,keys):
        """ATLAS's bidirectional line reduce-scatter over ``cores`` with whole
        tiles as chunks: tile ``c`` ends on rank ``r-1-c``; at step ``s`` rank
        ``j`` forwards its running sum of one tile to each neighbour. Each
        partial is loaded one row before its first use, received partials are
        added in the next row, before being forwarded, and owners store."""
        r=len(cores);first={};last={}
        def use(row,core,key):
            first[core,key]=min(first.get((core,key),row),row);last[core,key]=max(last.get((core,key),row),row)
        moves=[]
        for s in range(r-1):
            for j in range(r):
                for d,c in (((j+1,s-j),) if j<=s else ())+(((j-1,r-1-abs(j+s-(r-1))),) if j>=r-1-s else ()):
                    if 0<=d<r and c<len(keys):
                        moves.append((s,cores[j],cores[d],keys[c]))
                        use(1+s,cores[j],keys[c]);use(2+s,cores[d],keys[c])
        for core in cores:
            rows[0][core].append(dict(op='clear',tensor='C'));cache[core]['C']=set()
        for (core,key),row in sorted(first.items(),key=lambda x:x[1]):
            rows[row-1][core].append(access(core,'C',key));cache[core]['C'].add(key)
        for s,src,dst,key in moves:  # by step: a row's adds precede its sends
            rows[2+s][dst].append(dict(op='add',key=key,byte_count=tile_bytes['C']))
            rows[1+s][src].append(dict(op='send',dst=dst,tensor='C',target='partial',key=key,byte_count=tile_bytes['C']))
        for c,key in enumerate(keys):
            if p.output != 'replicated':
                rows[r+1][cores[r-1-c]].append(access(cores[r-1-c],'C',key,True,True))
        for (core,key),row in last.items():  # free a forwarded partial after its last send
            if core!=cores[r-1-keys.index(key)]:
                rows[row][core].append(dict(op='drop',tensor='C',key=key));cache[core]['C'].discard(key)

    epochs=list(loop_points(tuple(n//g for n,g in zip(base_tiles,group))))
    step=0
    for epoch_index,epoch in enumerate(epochs):
        init=phase()
        for core in range(core_count):
            for v,axes in DEPENDENCIES.items():
                if not shifted[v] or land[v]:continue
                keys=[]
                for block in own.tensor_blocks(core,v):
                    ranges=[range(b*base_tiles[d]+epoch[d]*group[d],b*base_tiles[d]+(epoch[d]+1)*group[d])
                            for b,d in zip(block,axes)]
                    keys.extend(product(*ranges))
                prepare(init,core,v,keys)
        state=(0,)*len(own.layers)
        for event,arg in own.events():
            if event=='shift':
                exchange(arg,own.layers[arg].tensors,step=step-1)
                continue
            state=arg
            for micro in loop_points(group):
                triples=[]
                for core in range(core_count):
                    block=own.block(core,state)
                    triples.append(tuple(b*bt+e*g+x for b,bt,e,g,x in zip(block,base_tiles,epoch,group,micro)))
                stores=phase();loads=phase()
                for core,(m,n,k) in enumerate(triples):
                    if (not shifted['C'] or land['C']) and cache[core]['C']!={(m,n)}:save_c(stores,core,step=step)
                    for v,key in (('A',(m,k)),('B',(k,n)),('C',(m,n))):
                        if not shifted[v] or land[v]:prepare(loads,core,v,[key],step)
                        elif key not in cache[core][v]:raise AssertionError(f'Missing resident {v} tile {key}')
                compute=phase()
                for core,index in enumerate(triples):
                    compute[core].append(dict(op='gemm',index=index,**({'step':step} if p.land else {})))
            step+=1
        next_epoch=epochs[epoch_index+1] if epoch_index+1<len(epochs) else None
        # Restore only live panels. Others will load the next epoch from local
        # homes. C must return to its declared owner before materialization.
        reuse={v:(v=='C' or next_epoch is not None and all(epoch[d]==next_epoch[d] for d in axes))
               for v,axes in DEPENDENCIES.items()}
        for i,offset in enumerate(state):
            tensors=tuple(v for v in own.layers[i].tensors if reuse[v])
            for _ in range((-offset)%own.layers[i].radix):
                if tensors:exchange(i,tensors)
        flush=phase()
        for core in range(core_count):
            for v,axes in DEPENDENCIES.items():
                if not shifted[v]:continue
                if v=='C':
                    keep=next_epoch is not None and all(epoch[d]==next_epoch[d] for d in axes)
                    if not keep:
                        save_c(flush,core)
                        cache[core][v]=set();flush[core].append(dict(op='clear',tensor=v))
                elif not reuse[v]:
                    # The next prepare clears/replaces the bank. A standalone
                    # metadata-clear phase would unnecessarily break prefetch.
                    cache[core][v]=set()
    flush=phase()
    for core in range(core_count):save_c(flush,core)
    # Finalize every C tile. A tile with partials on several cores (residual
    # split-K) is reduced like ATLAS's line collective: a group's tiles are
    # dealt to its ranks and run its bidirectional linear reduce-scatter.
    owners={}
    for core in range(core_count):
        for key in done(core):owners.setdefault(key,[]).append(core)
    groups={}
    for key,cores in sorted(owners.items()):
        if len(cores)==1:
            # No residual split-K: the last write (a store, or for a landed C
            # the copy that brings it home) already is final. Do not
            # reread/rewrite it merely to mark the output in the oracle.
            writes=[a for row in phases for ops in row for a in ops
                    if a.get('tensor')=='C' and a['op'] in ('store','forward') and tuple(a['key'])==key]
            writes[-1]['final']=True
            continue
        groups.setdefault(tuple(cores),[]).append(key)
    waves=[[(cores,keys[i:i+(1 if p.output in ('first','last') else len(cores))])
            for i in range(0,len(keys),1 if p.output in ('first','last') else len(cores))] for cores,keys in groups.items()]
    for w in range(max(map(len,waves),default=0)):
        batch=[g[w] for g in waves if w<len(g)]
        rows=[phase() for _ in range(max(len(c) for c,_ in batch)+2)]
        for cores,keys in batch:
            if p.output in ('first','last'):
                key=keys[0]; root=root_index(p.output,len(cores),0)
                for core in cores:
                    rows[0][core].extend([dict(op='clear',tensor='C'),access(core,'C',key)])
                for t,src,dst in reduce_steps(cores,root):
                    rows[1+t][src].append(dict(op='send',dst=dst,tensor='C',target='partial',key=key,byte_count=tile_bytes['C']))
                    rows[2+t][dst].insert(0,dict(op='add',key=key,byte_count=tile_bytes['C']))
                rows[-1][cores[root]].append(access(cores[root],'C',key,True,True))
            else:
                reduce_scatter(rows,cores,keys)
        if p.output == 'replicated':
            # Broadcast completed tiles directly from SRAM, one tile at a
            # time per group. No intermediate final store/reload is inserted.
            for c in range(max(len(keys) for _,keys in batch)):
                final=phase()
                send_rows=[phase() for _ in range(max(len(cores) for cores,_ in batch))]
                stores=phase()
                for cores,keys in batch:
                    if c>=len(keys):continue
                    key=keys[c]; root=len(cores)-1-c
                    final[cores[root]].append(dict(access(cores[root],'C',key,True,True),op='finalize'))
                    for t,src,dst in broadcast_steps(cores,root):
                        send_rows[t][src].append(dict(op='copy',dst=dst,tensor='C',key=key,byte_count=tile_bytes['C']))
                    for core in cores:
                        stores[core].append(access(core,'C',key,True,True))
                        stores[core].append(dict(op='drop',tensor='C',key=key))
    phases=[row for row in phases if any(row)]
    if overlap:
        phases=pipeline(phases)
    if forwards:
        phases=place_forwards(phases,forwards)
    return Program(problem,plan,phases,placements,needed,overlap)


def timed(row):
    return any(a['op'] not in ('clear','zero','drop') for ops in row for a in ops)


def place_forwards(phases,forwards):
    """Overlap each DRAM-landing copy with the compute of the step before it.

    A copy at (timed) phase t reads at t-1, sends at t and lands at t+1. It
    reads after the step's first compute starts and lands before the next
    step first loads the block; idle phases are inserted only when the step
    is too short.
    """
    for step,v,chunks in forwards:
        times=[i for i,row in enumerate(phases) if timed(row)]
        def first(match):
            return next(j for j,i in enumerate(times) if any(match(a) for ops in phases[i] for a in ops))
        start=first(lambda a:a['op']=='gemm' and a['step']==step)
        end=first(lambda a:a['op']=='load' and a['tensor']==v and a['step']==step+1)
        need=max(0,3-(end-start))
        for _ in range(need):
            phases.insert(times[end],[[dict(op='nop')]]+[[] for _ in phases[0][1:]])
        times=[i for i,row in enumerate(phases) if timed(row)]
        slots=times[start+1:end+need-1]
        for core,ops in enumerate(chunks):
            for j,op in enumerate(ops):
                phases[slots[j*len(slots)//len(ops)]][core].append(op)
    return phases


def pipeline(phases):
    """Local, dependency-checked overlap using the allocated second A/B banks."""
    def only(row,kinds):return all(a['op'] in kinds for ops in row for a in ops)
    def compute(row):return only(row,{'gemm','store','drop'}) and any(a['op']=='gemm' for ops in row for a in ops)
    # Coalesce independent initial panel/operand loads into one native batch.
    work=[]
    for row in phases:
        if work and only(row,{'clear','zero','load'}) and only(work[-1],{'clear','zero','load'}):
            work[-1]=[a+b for a,b in zip(work[-1],row)]
        else:work.append(deepcopy(row))
    phases=[];i=0
    while i<len(work):
        row=work[i]
        if i+3<len(work) and compute(row) and only(work[i+1],{'store'}) and only(work[i+2],{'clear','zero','load'}) and only(work[i+3],{'gemm'}):
            stores,loads,next_compute=work[i+1:i+4]
            valid=True
            for current,save,load in zip(row,stores,loads):
                pending={a['key'] for a in current if a['op']=='store'}
                valid &= len(save)==1 and save[0]['tensor']=='C'
                valid &= not any(a['op']=='load' and a['tensor']=='C' and a['key'] in pending for a in load)
                valid &= any(a['op']=='clear' and a['tensor']=='C' for a in load)
            if valid:
                for core in range(len(row)):
                    key=stores[core][0]['key']
                    for a in loads[core]:
                        if a['op']=='clear' and a['tensor']=='C':a['keep']=[key]
                    row[core]+=loads[core]
                    next_compute[core]=stores[core]+[dict(op='drop',tensor='C',key=key)]+next_compute[core]
                phases.append(row);i+=3;continue
        phases.append(row);i+=1
    out=[];i=0
    while i<len(phases):
        row=phases[i]
        if compute(row) and i+1<len(phases):
            following=phases[i+1]
            if only(following,{'clear','load'}) and all(a.get('tensor') in ('A','B') for ops in following for a in ops):
                out.append([a+b for a,b in zip(row,following)]);i+=2;continue
            if only(following,{'send'}) and all(a['tensor'] in ('A','B') for ops in following for a in ops):
                out.append([a+b for a,b in zip(row,following)]);i+=2;continue
        out.append(row);i+=1
    # When A rotates into its second bank, the independent second B bank can
    # still prefetch the next B tile (and conversely). Do not overwrite a bank
    # already reserved for the incoming tensor in this same phase.
    merged=[];i=0
    while i<len(out):
        row=out[i]
        if i+1<len(out) and only(row,{'gemm','send','store','drop'}) and only(out[i+1],{'clear','load'}):
            incoming={a['tensor'] for ops in row for a in ops if a['op']=='send'}
            loads={a['tensor'] for ops in out[i+1] for a in ops}
            if loads<= {'A','B'} and not loads & incoming:
                merged.append([a+b for a,b in zip(row,out[i+1])]);i+=2;continue
        merged.append(row);i+=1
    return merged
