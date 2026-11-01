"""Method-specific fused dataflows over the shared attention event IR.

The DRAM ring exchanges whole logical shards between local tiled computations.
Flash OS/WS use a fixed query/context grid and linear SRAM all-gathers; OS
keeps query state, WS keeps the gathered KV panel while streaming query waves.
Neither path writes the score matrix or changes persistent KV ownership.
"""
from math import ceil

from .attention import assignments
from .attention_schedule import Event, lane_core, packets
from ..tasks import Region


def query_waves(schedule, head, request):
    p,plan=schedule.layout.problem,schedule.plan
    blocks={(s,c):r for h,b,s,c,r in assignments(schedule.layout,plan) if (h,b)==(head,request)}
    for wave in range(ceil(ceil(p.queries/plan.query_parts)/plan.query_tile)):
        result={}
        for (s,c),r in blocks.items():
            first=wave*plan.query_tile
            if first<r.rows:
                result[head,request,s,wave,c]=r.crop(r.row+first,r.col,min(plan.query_tile,r.rows-first),r.cols)
        yield result


def slot(schedule, kind, core, bank, r, origin):
    """DRAM scratch for a full logical shard; micro-tiles are views into it."""
    p=schedule.layout.problem
    cols=r.cols if kind=='kv' else 3*r.cols+4*p.group
    name=f'work_{kind}_{bank}'
    region=Region(core,name,r.row,0,r.rows,cols,r.row-origin,0)
    old=schedule.scratch.get((core,name),(0,cols))
    schedule.scratch[core,name]=(max(old[0],region.local_row+r.rows),cols)
    return region


def copy_blocks(schedule, time, copies):
    """Issue at most one chunk per endpoint per step into read/send/store banks."""
    available={}; end=time
    for kind,key,r,src,dst in copies:
        t=max(time,available.get(src.core,time),available.get(dst.core,time))
        schedule.events.append(Event(t,kind,src.core,key,r,storage=dst,source=src))
        available[src.core]=available[dst.core]=t+1
        end=max(end,t+3)
    return end


def build_dram(schedule):
    """Rotate whole DRAM shards, independently of the SRAM micro-tile size."""
    layout,plan=schedule.layout,schedule.plan; p=layout.problem
    pq,pc=plan.query_parts,plan.context_parts; emit=schedule.events.append
    core=lambda h,b,s,c:lane_core(layout,plan,h,b,s,c)
    for h in range(p.kv_heads):
        time=0
        for b in range(p.batch):
            waves=list(query_waves(schedule,h,b)); qstorage={}
            held={(c,lane):[((h,b,c,j,lane),r,r) for j,r in enumerate(rs)]
                  for c in range(pc) for lane,rs in enumerate(packets(layout,plan,h,b,c,schedule.cache))}
            for stage in range(pq):
                # Read-only whole-block copies may overlap all computations
                # in this rotation stage, not just its final micro-tile.
                copies=[]; following={}
                if stage<pq-1 and plan.kv_shift:
                    for (c,lane),items in held.items():
                        dest=(lane+plan.kv_shift)%pq; next_items=[]
                        for key,r,src in items:
                            origin=b*p.context+p.context*(c*pq+key[-1])//(pq*pc)
                            dst=slot(schedule,'kv',core(h,b,dest,c),(stage+1)%2,r,origin)
                            copies.append(('kv_copy',key,r,src,dst)); next_items.append((key,r,dst))
                        following[c,dest]=next_items
                copy_end=copy_blocks(schedule,time,copies)
                for regions in waves:
                    positions={key:(key[2]+stage*plan.query_shift)%pq for key in regions}
                    for key,r in regions.items():
                        owner=core(h,b,positions[key],key[-1])
                        emit(Event(time,'q_load' if stage==0 else 'q_restore',owner,key,r,
                                   storage=qstorage.get(key)))
                    start=time+3
                    active=[(key,r,core(h,b,positions[key],key[-1]),held[key[-1],positions[key]])
                            for key,r in regions.items()]
                    count=max((len(items) for _,_,_,items in active),default=0)
                    for j in range(count):
                        load=start if j==0 else start+2+4*(j-1)
                        update=start+2+4*j
                        for q,r,owner,items in active:
                            if j>=len(items):continue
                            k,kv,src=items[j]
                            emit(Event(load,'kv_restore',owner,k,kv,storage=src))
                            emit(Event(update,'update',owner,q,r,other=k))
                            emit(Event(update+4,'kv_drop',owner,k))
                    time=start+2+4*count
                    if stage==pq-1:
                        time=schedule.finish_wave(time+2,positions,regions)
                    else:
                        for key,r in regions.items():
                            owner=core(h,b,positions[key],key[-1])
                            origin=b*p.queries+key[2]*ceil(p.queries/pq)
                            target=qstorage.get(key) or slot(schedule,'q',owner,0,r,origin)
                            emit(Event(time,'q_save',owner,key,r,storage=target)); qstorage[key]=target
                        time+=3
                time=max(time,copy_end)+1
                if following:held=following
                if stage<pq-1 and plan.query_shift:
                    copies=[]
                    for regions in waves:
                        for key,r in regions.items():
                            dest=(key[2]+(stage+1)*plan.query_shift)%pq
                            origin=b*p.queries+key[2]*ceil(p.queries/pq)
                            dst=slot(schedule,'q',core(h,b,dest,key[-1]),(stage+1)%2,r,origin)
                            copies.append(('q_copy',key,r,qstorage[key],dst)); qstorage[key]=dst
                    time=copy_blocks(schedule,time,copies)+1


def gather_kv(schedule, time, head, request, pieces, turn):
    """Read single-copy slabs, then line all-gather before any tile update."""
    layout,plan=schedule.layout,schedule.plan; emit=schedule.events.append
    gathered={}
    for c,lanes in enumerate(pieces):
        members=[lane_core(layout,plan,head,request,s,c) for s in range(plan.query_parts)]
        for origin,tiles in enumerate(lanes):
            if turn>=len(tiles):continue
            r=tiles[turn]; key=(head,request,c,turn,origin)
            emit(Event(time,'kv_load',r.core,key,r))
            if r.core!=members[origin]:
                emit(Event(time+1,'kv_move',r.core,key,r,members[origin]))
            for direction in (-1,1):
                for hop in range(1,len(members)):
                    dest=origin+direction*hop
                    if not 0<=dest<len(members):break
                    emit(Event(time+1+hop,'kv_send',members[dest-direction],key,r,members[dest]))
            gathered[key]=r
    return time+plan.query_parts+2,gathered


def build_flash(schedule):
    """Fixed gather dataflow, with Q-state or gathered-KV stationary loops."""
    layout,plan=schedule.layout,schedule.plan; p=layout.problem
    emit=schedule.events.append; pq,pc=plan.query_parts,plan.context_parts
    def queries(time,regions,storage,restore):
        for key,r in regions.items():
            emit(Event(time,'q_restore' if restore else 'q_load',r.core,key,r,storage=storage.get(key)))
        return time+(3 if restore else pc+3)
    def update(time,regions,gathered):
        for origin in range(pq):
            for q,r in regions.items():
                for k,kv in gathered.items():
                    if k[2]==q[-1] and k[-1]==origin:
                        emit(Event(time,'update',r.core,q,r,other=k))
            time+=4
        return time
    def drop(time,gathered):
        for key,r in gathered.items():
            h,b,c,_,_=key
            for lane in range(pq):
                emit(Event(time,'kv_drop',lane_core(layout,plan,h,b,lane,c),key))
    for h in range(p.kv_heads):
        time=0
        for b in range(p.batch):
            pieces=[packets(layout,plan,h,b,c,schedule.cache) for c in range(pc)]
            count=max(len(rs) for lanes in pieces for rs in lanes)
            waves=list(query_waves(schedule,h,b))
            if plan.dataflow=='flash_os':
                for regions in waves:
                    time=queries(time,regions,{},False)
                    ready,gathered=gather_kv(schedule,time,h,b,pieces,0)
                    for turn in range(count):
                        time=max(time,ready)
                        next_panel=gather_kv(schedule,time,h,b,pieces,turn+1) if turn+1<count else None
                        time=update(time,regions,gathered); drop(time,gathered)
                        if next_panel:ready,gathered=next_panel
                    time=schedule.finish_wave(time+2,{key:key[2] for key in regions},regions)
            else:
                storage={}; ready,gathered=gather_kv(schedule,time,h,b,pieces,0)
                for turn in range(count):
                    time=max(time,ready)
                    next_panel=gather_kv(schedule,time,h,b,pieces,turn+1) if turn+1<count else None
                    for regions in waves:
                        time=queries(time,regions,storage,turn>0)
                        time=update(time,regions,gathered)
                        if turn==count-1:
                            time=schedule.finish_wave(time+2,{key:key[2] for key in regions},regions)
                        else:
                            for key,r in regions.items():
                                origin=b*p.queries+key[2]*ceil(p.queries/pq)
                                dst=slot(schedule,'q',r.core,0,r,origin)
                                emit(Event(time,'q_save',r.core,key,r,storage=dst)); storage[key]=dst
                            time+=3
                    drop(time,gathered)
                    if next_panel:ready,gathered=next_panel
