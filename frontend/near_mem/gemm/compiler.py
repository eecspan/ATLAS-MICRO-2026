"""General multi-axis compiler with explicit ownership and live tensor panels.

Full resident shifted panels are supported. Partial-panel spilling and coarse
DRAM landing are rejected, not approximated by analytical traffic. Six micro
loop orders and saturate-inner groups lower into actual loads/RMW/reuse.
"""
from itertools import product
from math import prod
from copy import deepcopy

from .ownership import Ownership
from .plan import DEPENDENCIES
from .program import Program
from ..sim import span, strides


def compile_plan(problem, plan, sram_bytes, dram_bytes, overlap=True):
    own=Ownership(problem,plan)
    p=plan;core_count=p.cores
    shifted={v:prod(parts)>1 for v,parts in p.temporal.items()}
    counts={v:payload if shifted[v] else 1 for v,payload in zip(DEPENDENCIES,p.payload_tiles)}
    if p.receive!='sram_consume':
        raise ValueError('Coarse DRAM landing is not lowered by the resident-panel compiler')
    if p.panel is not None and tuple(counts.values())!=p.panel:
        raise ValueError('Partial-panel spilling requires a separate explicit spill schedule')
    tile_shapes={v:tuple(p.tile[d] for d in axes) for v,axes in DEPENDENCIES.items()}
    tile_bytes={v:prod(shape)*problem.dtype_bytes for v,shape in tile_shapes.items()}
    # C's partial-result scratch is reused as the receive bank. A/B two-bank
    # allocation admits prefetch or computation/shift overlap, never both future
    # payloads in one bank. Conservative capacity is explicit in the manifest.
    needed=sum(counts[v]*tile_bytes[v]*(2 if overlap or shifted[v] or v=='C' else 1) for v in DEPENDENCIES)
    if overlap and not shifted['C']:
        needed+=2*tile_bytes['C']  # Native double-buffered partial and running outputs.
    if needed>sram_bytes:
        raise ValueError(f'SRAM capacity exceeded: {needed} > {sram_bytes}')
    base_tiles=tuple(b//t for b,t in zip(own.base,p.tile))
    placements=[];address_maps=[]
    for core in range(core_count):
        tensors=[];maps={};addr=0
        for v,axes in DEPENDENCIES.items():
            blocks=own.tensor_blocks(core,v)
            axis_blocks=[sorted({key[i] for key in blocks}) for i in range(2)]
            shape=[len(keys)*own.base[d] for keys,d in zip(axis_blocks,axes)]
            order=(1,0) if v=='B' else (0,1)
            shape_order=[shape[i] for i in order];tile_order=[tile_shapes[v][i] for i in order]
            addr=(addr+127)//128*128
            t=dict(name=v,base_addr=addr,element_size=problem.dtype_bytes)
            t.update(shape=shape,strides=strides(shape,problem.dtype_bytes,v=='B'))
            tensors.append(t);addr+=span(shape,t['strides'],problem.dtype_bytes)
            maps[v]=(axis_blocks,shape,order)
        if addr>dram_bytes:raise ValueError('Aligned DRAM placement exceeds capacity')
        placements.append(dict(core_id=core,tensor=tensors));address_maps.append(maps)
    phases=[]
    cache=[{v:set() for v in DEPENDENCIES} for _ in range(core_count)]
    materialized=[set() for _ in range(core_count)]

    def phase():
        actions=[[] for _ in range(core_count)];phases.append(actions);return actions

    def access(core,v,key,write=False,final=False):
        axes=DEPENDENCIES[v];blocks,shape,order=address_maps[core][v]
        base=[(blocks[i].index(key[i]//base_tiles[d])*base_tiles[d]+key[i]%base_tiles[d])*p.tile[d]
              for i,d in enumerate(axes)]
        item=dict(op='store' if write else 'load',tensor=v,key=key,shape=tile_shapes[v],final=final)
        item.update(base=base,extent=list(tile_shapes[v]))
        return item

    def save_c(row,core,final=False):
        for key in sorted(cache[core]['C']):
            row[core].append(access(core,'C',key,True,final));materialized[core].add(key)

    def prepare(row,core,v,keys):
        keys=set(keys)
        if keys==cache[core][v]:return
        row[core].append(dict(op='clear',tensor=v))
        cache[core][v]=keys
        for key in sorted(keys):
            if v=='C' and key not in materialized[core]:
                row[core].append(dict(op='zero',tensor='C',key=key,shape=tile_shapes[v]))
            else:row[core].append(access(core,v,key))

    def exchange(layer_id, tensors, direction=-1):
        # Like native ATLAS collectives, the arrival of a step's data is the
        # neighbour synchronisation; no separate ready-credit phase is issued.
        data=phase()
        for v in tensors:
            old=[dict(c) for c in cache]
            for src in range(core_count):
                dst=own.neighbor(src,layer_id,v,direction)
                data[src].append(dict(op='send',dst=dst,tensor=v,byte_count=len(old[src][v])*tile_bytes[v]))
                cache[dst][v]=old[src][v]

    def loop_points(extents):
        order=tuple('mnk'.index(c) for c in p.loop)
        for values in product(*(range(extents[d]) for d in order)):
            point=[0,0,0]
            for d,value in zip(order,values):point[d]=value
            yield tuple(point)

    epochs=list(loop_points(tuple(n//g for n,g in zip(base_tiles,p.group))))
    for epoch_index,epoch in enumerate(epochs):
        init=phase()
        for core in range(core_count):
            for v,axes in DEPENDENCIES.items():
                if not shifted[v]:continue
                keys=[]
                for block in own.tensor_blocks(core,v):
                    ranges=[range(b*base_tiles[d]+epoch[d]*p.group[d],b*base_tiles[d]+(epoch[d]+1)*p.group[d])
                            for b,d in zip(block,axes)]
                    keys.extend(product(*ranges))
                prepare(init,core,v,keys)
        state=(0,)*len(own.layers)
        for event,arg in own.events():
            if event=='shift':
                exchange(arg,own.layers[arg].tensors)
                continue
            state=arg
            for micro in loop_points(p.group):
                triples=[]
                for core in range(core_count):
                    block=own.block(core,state)
                    triples.append(tuple(b*bt+e*g+x for b,bt,e,g,x in zip(block,base_tiles,epoch,p.group,micro)))
                stores=phase();loads=phase()
                for core,(m,n,k) in enumerate(triples):
                    if not shifted['C'] and cache[core]['C']!={(m,n)}:save_c(stores,core)
                    for v,key in (('A',(m,k)),('B',(k,n)),('C',(m,n))):
                        if not shifted[v]:prepare(loads,core,v,[key])
                        elif key not in cache[core][v]:raise AssertionError(f'Missing resident {v} tile {key}')
                compute=phase()
                for core,index in enumerate(triples):compute[core].append(dict(op='gemm',index=index))
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
    # Finalize every C tile, including residual split-K replicas after C rotation.
    owners={}
    for core in range(core_count):
        for key in materialized[core]:owners.setdefault(key,[]).append(core)
    for key,cores in sorted(owners.items()):
        if len(cores)==1:
            # No residual split-K: the last materialization already is final.
            # Do not reread/rewrite it merely to mark the output in the oracle.
            core=cores[0]
            for row in reversed(phases):
                stores=[a for a in row[core] if a['op']=='store' and a['key']==key]
                if stores:
                    stores[-1]['final']=True
                    break
            continue
        loads=phase()
        for core in cores:prepare(loads,core,'C',[key])
        stride=1
        while stride<len(cores):
            data=phase();adds=phase()
            for index in range(stride,len(cores),2*stride):
                src,dst=cores[index],cores[index-stride]
                data[src].append(dict(op='send',dst=dst,tensor='C',target='partial',byte_count=tile_bytes['C']))
                adds[dst].append(dict(op='add',byte_count=tile_bytes['C']))
            stride*=2
        store=phase();store[cores[0]].append(access(cores[0],'C',key,True,True))
    phases=[row for row in phases if any(row)]
    if overlap:
        phases=pipeline(phases)
    return Program(problem,plan,phases,placements,needed)


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
