"""Fused attention packet schedule, shared by value replay and ATLAS lowering.

Q and its online state are one mobile object; paired K/V are another. Each
context segment visits the Cartesian product of its Q and KV packets through
relative cyclic shifts. Context segments own disjoint tokens and merge once.
Read-only KV forwarding and next-round DRAM staging overlap current updates;
Q/state movement waits for its update. A shared output contract selects the
reduction owner and optional broadcast after normalization.
DRAM/WS dataflows may spill online state, but never the score matrix; KV
owners never change. The same events drive numeric replay and native tasks.
"""
from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil

from .attention import assignments, endpoints
from ..output import root_index, reduce_steps, broadcast_steps
from ..tasks import transfers


def lane_core(layout, plan, head, request, lane, segment):
    start,_ = layout.head_group(head)
    return start+((request*plan.query_parts+lane)*plan.context_parts+segment)%layout.group_cores


def packets(layout, plan, head, request, segment, cache=None):
    """Per-lane lists of owner-local tiles, clipping both shard and segment edges."""
    p = layout.problem; count = plan.query_parts*plan.context_parts
    step = plan.kv_tile or ceil(p.context/count)
    regions = layout.slices(head, request, cache)
    result = []
    for lane in range(plan.query_parts):
        part = segment*plan.query_parts+lane
        lo = request*p.context+p.context*part//count
        hi = request*p.context+p.context*(part+1)//count
        pieces = []
        for r in regions:
            a,b = max(lo,r.row),min(hi,r.row+r.rows)
            for row in range(a,b,step):
                pieces.append(r.crop(row,r.col,min(step,b-row),r.cols))
        result.append(sorted(pieces,key=lambda r:r.row))
    return result


@dataclass(frozen=True)
class Event:
    time: int
    kind: str
    core: int
    key: tuple
    region: object = None
    peer: object = None
    other: tuple = ()
    ready: int = None
    storage: object = None
    source: object = None


class Schedule:
    def __init__(self, layout, plan, cache=None):
        plan.validate(layout,plan.sram_bytes(layout.problem))
        self.layout = layout; self.plan = plan; self.events = []
        self.cache = layout.regions() if cache is None else cache
        self.scratch = {}
        if plan.dataflow == 'dram' or plan.flash:
            from .attention_flows import build_dram, build_flash
            (build_dram if plan.dataflow == 'dram' else build_flash)(self)
        else:
            self._build()
        self.events.sort(key=lambda e:e.time)
        # The first local consumer can be compute or a transfer to another
        # lane. Waiting there also orders every subsequent receiver's use.
        pending = {}
        for i,e in enumerate(self.events):
            if e.kind in ('kv_load','kv_restore'):
                pending[e.core,e.key] = i
            elif e.kind in ('kv_move','kv_send','update'):
                key = (e.core,e.other if e.kind == 'update' else e.key)
                if key in pending:
                    at = pending.pop(key)
                    self.events[at] = replace(self.events[at],ready=e.time)
        assert not pending, 'Every staged KV packet must have a consumer'

    def buffer_peak(self):
        """Audit packet lifetimes, including simultaneous receive and compute.

        Sends retain their source through the send step; KV forwarding keeps
        its source until PV completes. Add worst-case score/output scratch
        and one partial-state receive slot to the observed packet footprint.
        """
        p=self.layout.problem; actions=[]
        for e in self.events:
            r=e.region; qsize=(r.area*6+r.rows*p.group*8) if r else 0
            if e.kind in ('q_load','q_restore','kv_load','kv_restore'):
                kind='q' if e.kind.startswith('q_') else 'kv'
                actions.append((e.time,1,e.core,(kind,e.key),qsize if kind=='q' else r.area*2))
            elif e.kind in ('q_move','kv_move','kv_send'):
                kind='q' if e.kind=='q_move' else 'kv'; size=qsize if kind=='q' else r.area*2
                actions.append((e.time,1,e.peer,(kind,e.key),size))
                if e.kind!='kv_send':actions.append((e.time+1,-1,e.core,(kind,e.key),0))
            elif e.kind=='kv_drop':actions.append((e.time,-1,e.core,('kv',e.key),0))
            elif e.kind=='merge':actions.append((e.time+1,-1,e.core,('q',e.key),0))
            elif e.kind=='normalize':actions.append((e.time+1,-1,e.core,('q',e.key),0))
            elif e.kind=='q_save':actions.append((e.time+2,-1,e.core,('q',e.key),0))
            elif e.kind in ('q_copy','kv_copy'):
                size=e.storage.area*2; key=('dma',len(actions))
                actions.extend([(e.time,1,e.source.core,(key,'src'),size),(e.time+2,-1,e.source.core,(key,'src'),0),
                                (e.time+1,1,e.storage.core,(key,'dst'),size),(e.time+3,-1,e.storage.core,(key,'dst'),0)])
        live={}; used=[0]*self.layout.cores; peak=0
        for _,sign,core,key,size in sorted(actions,key=lambda a:(a[0],a[1])):
            if sign<0:used[core]-=live.pop((core,key))
            else:
                assert (core,key) not in live
                live[core,key]=size; used[core]+=size; peak=max(peak,used[core])
        assert not live
        q=min(self.plan.query_tile,ceil(p.queries/self.plan.query_parts))*p.group
        k=self.plan.kv_tile or ceil(p.context/(self.plan.query_parts*self.plan.context_parts))
        return peak+4*q*k+4*q*p.head_dim

    def _build(self):
        p,plan = self.layout.problem,self.plan
        pq,pc = plan.query_parts,plan.context_parts
        blocks = {(h,b,s,c):r for h,b,s,c,r in assignments(self.layout,plan)}
        for h in range(p.kv_heads):
            t = 0
            for b in range(p.batch):
                pieces = [packets(self.layout,plan,h,b,c,self.cache) for c in range(pc)]
                for wave in range(ceil(ceil(p.queries/pq)/plan.query_tile)):
                    qpos = {}; qregions = {}; ends = []
                    for c in range(pc):
                        for s in range(pq):
                            block = blocks.get((h,b,s,c)); first = wave*plan.query_tile
                            if block is None or first >= block.rows:
                                continue
                            key = (h,b,s,wave,c)
                            r = block.crop(block.row+first,block.col,min(plan.query_tile,block.rows-first),block.cols)
                            qpos[key] = s; qregions[key] = r
                            self.events.append(Event(t,'q_load',r.core,key,r))
                    for c in range(pc):
                        u = t+2
                        prefetched = set()
                        for turn in range(max(map(len,pieces[c]))):
                            kvpos = {}; kvregions = {}
                            for lane,tiles in enumerate(pieces[c]):
                                if turn >= len(tiles):
                                    continue
                                r = tiles[turn]; key = (h,b,c,turn,lane)
                                core = lane_core(self.layout,plan,h,b,lane,c)
                                # A prefetched tile is already in the independent
                                # staging bank when this turn starts.  This is
                                # the same phase convention as GEMM: the next
                                # DRAM load is issued with the current turn's
                                # first compute, not after the current turn.
                                if key not in prefetched:
                                    self.events.append(Event(u,'kv_load',r.core,key,r))
                                else:
                                    prefetched.remove(key)
                                if core != r.core:
                                    self.events.append(Event(u+1,'kv_move',r.core,key,r,core))
                                kvpos[key] = lane; kvregions[key] = r
                            u += 3
                            # Prefetch the next KV turn at the first QK update
                            # of this turn.  Later stages (softmax, PV and
                            # rotation) occupy the interval before next use.
                            next_turn = turn+1
                            if next_turn < max(map(len,pieces[c])):
                                for lane,tiles in enumerate(pieces[c]):
                                    if next_turn >= len(tiles):
                                        continue
                                    r = tiles[next_turn]
                                    key = (h,b,c,next_turn,lane)
                                    self.events.append(Event(u,'kv_load',r.core,key,r))
                                    prefetched.add(key)
                            for stage in range(pq):
                                for q,qlane in qpos.items():
                                    if q[-1] != c:
                                        continue
                                    for k,klane in kvpos.items():
                                        if qlane == klane:
                                            core = lane_core(self.layout,plan,h,b,qlane,c)
                                            self.events.append(Event(u,'update',core,q,qregions[q],other=k))
                                if stage == pq-1:
                                    u += 4
                                    break
                                for positions,regions,shift,kind in ((qpos,qregions,plan.query_shift,'q_move'),
                                                                    (kvpos,kvregions,plan.kv_shift,'kv_move')):
                                    for key,lane in list(positions.items()):
                                        if kind == 'q_move' and key[-1] != c:
                                            continue
                                        dest = (lane+shift)%pq
                                        src = lane_core(self.layout,plan,h,b,lane,c)
                                        dst = lane_core(self.layout,plan,h,b,dest,c)
                                        if src != dst:
                                            if kind == 'kv_move':
                                                # Read-only KV can serve compute
                                                # and send concurrently; retain
                                                # its source until PV consumes it.
                                                self.events.append(Event(u,'kv_send',src,key,regions[key],dst))
                                                self.events.append(Event(u+4,'kv_drop',src,key))
                                            else:
                                                self.events.append(Event(u+4,kind,src,key,regions[key],dst))
                                        positions[key] = dest
                                u += 6 if plan.query_shift else 4
                            for key,lane in kvpos.items():
                                self.events.append(Event(u,'kv_drop',lane_core(self.layout,plan,h,b,lane,c),key))
                        ends.append(u)
                    t = max(ends)+2
                    t = self.finish_wave(t,qpos,qregions)

    def finish_wave(self,t,qpos,qregions):
        plan=self.plan; pq,pc=plan.query_parts,plan.context_parts
        h,b=next(iter(qpos))[:2]; wave=next(iter(qpos))[3]
        for s in range(pq):
            root = root_index(plan.output,pc,pc-1)
            dst = (h,b,s,wave,root)
            if dst not in qpos:
                continue
            core = lane_core(self.layout,plan,h,b,qpos[dst],root)
            cores = [lane_core(self.layout,plan,h,b,qpos[h,b,s,wave,c],c) for c in range(pc)]
            for step,c,d in reduce_steps(list(range(pc)),root):
                src = (h,b,s,wave,c); target = (h,b,s,wave,d)
                self.events.append(Event(t+3*step,'merge',cores[c],src,qregions[src],cores[d],target))
            done = t+3*(pc-1)+1
            self.events.append(Event(done,'normalize',core,dst,qregions[dst]))
            if plan.output == 'replicated':
                for step,src,target in broadcast_steps(cores,root):
                    self.events.append(Event(done+1+step,'output_send',src,dst,qregions[dst],target))
                for target in cores:
                    self.events.append(Event(done+pc+1,'finish',target,dst,qregions[dst]))
            else:
                self.events.append(Event(done+1,'finish',core,dst,qregions[dst]))
        return t+3*(pc-1)+pc+4

    def lower(self, graph, query, name):
        p,plan = self.layout.problem,self.plan
        peak=self.buffer_peak()
        if peak>plan.sram_bytes(p):
            raise ValueError(f'Packet lifetime exceeds declared SRAM: {peak} > {plan.sram_bytes(p)}')
        inputs = []; outputs = []; outviews = {}
        for i,r in enumerate(endpoints(self.layout,plan)[0]):
            qname = f'{name}_q_{i}'
            local = next((x for src in query if src.core == r.core and
                          (x := src.intersect(r)) is not None and x.area == r.area),None)
            if local is None:
                graph.tensor(r.core,qname,[r.rows,r.cols]); local = replace(r,name=qname)
            inputs.append(local)
        for r in endpoints(self.layout,plan)[1]:
            h = r.col//(p.group*p.head_dim); b = r.row//p.queries
            s = (r.row%p.queries)//ceil(p.queries/plan.query_parts)
            out = replace(r,name=f'{name}_o_h{h}_b{b}_s{s}')
            graph.tensor(out.core,out.name,[out.rows,out.cols])
            outputs.append(out); outviews[h,b,s,out.core] = out
        graph.convert(name+'_query_layout',query,inputs)
        # Scratch is separate from the persistent KV cache and is released at
        # the operator boundary. Mixed Q/state packets use FP16-sized storage
        # units: FP16 Q, FP32 U/m/l, without rounding their byte counts away.
        scratch=[]
        for (core,label),shape in self.scratch.items():
            graph.tensor(core,name+'_'+label,shape)
            scratch.append((core,name+'_'+label))
        def physical(r):
            return replace(r,name=name+'_'+r.name) if (r.core,r.name) in self.scratch else r
        timeline = graph.timeline(); coverage = tiles = query_noc = 0
        kv = {e.key:e.region for e in self.events if e.kind in ('kv_load','kv_restore')}
        qloads={}
        for e in self.events:
            if e.kind=='q_load':qloads.setdefault((e.time,e.key[:-1]),[]).append(e)
        for (t,_),events in qloads.items():
            if plan.flash:
                # A fixed line all-gather of the unique Q slabs. No full Q
                # replica is written to DRAM before the fused computation.
                members=[e.core for e in events]
                for src,_ in transfers(inputs,[events[0].region]):
                    root=members.index(src.core) if src.core in members else 0
                    timeline.memory(t,src)
                    timeline.send(t+1,src.core,members[root],src.area*2)
                    query_noc+=src.area*2*(src.core!=members[root])
                    for step,a,b in broadcast_steps(members,root):
                        timeline.send(t+2+step,a,b,src.area*2)
                        query_noc+=src.area*2
            else:
                for e in events:
                    for src,_ in transfers(inputs,[replace(e.region,core=e.core)]):
                        timeline.memory(t,src)
                        timeline.send(t+1,src.core,e.core,src.area*2)
                        query_noc+=src.area*2*(src.core!=e.core)
        for e in self.events:
            t,r = e.time,e.region
            if e.kind == 'q_load':
                pass  # emitted above, with the method's load/gather rule
            elif e.kind == 'kv_load':
                timeline.memory(t,r,ready=e.ready)
            elif e.kind in ('q_copy','kv_copy'):
                timeline.memory(t,physical(e.source))
                timeline.send(t+1,e.source.core,e.storage.core,e.storage.area*2)
                timeline.memory(t+2,physical(e.storage),True)
            elif e.kind == 'q_save':
                timeline.memory(t+1,physical(e.storage),True)
            elif e.kind in ('q_restore','kv_restore'):
                src=physical(e.storage)
                ready=(t+1 if src.core!=e.core else e.ready) if e.kind=='kv_restore' else None
                timeline.memory(t,src,ready=ready)
                timeline.send(t+1,src.core,e.core,src.area*2)
            elif e.kind in ('q_move','kv_move','kv_send'):
                size = r.area*2 if e.kind != 'q_move' else r.area*6+r.rows*p.group*8
                timeline.send(t,e.core,e.peer,size)
            elif e.kind == 'update':
                k = kv[e.other]; b = e.key[1]
                # Charge the issued rectangle, including masked entries on
                # the causal diagonal; the hardware has no triangular MAC.
                end = b*p.context+p.context-p.queries+(r.row-b*p.queries)+r.rows
                tokens = max(0,min(k.rows,end-k.row)) if p.causal else k.rows
                count = r.rows*p.group*tokens
                if not count:
                    continue
                output = r.area
                timeline.work(e.core,t,macs=count*p.head_dim,read=(output+k.area//2)*2,write=count*4)
                timeline.work(e.core,t+1,vectors=9*count,read=9*count*4,write=count*4)
                timeline.work(e.core,t+2,macs=count*p.head_dim,read=count*4+k.area,write=output*4)
                timeline.work(e.core,t+3,vectors=3*output,read=output*8,write=output*4)
                coverage += count; tiles += 1
            elif e.kind == 'merge':
                size = r.area*4+r.rows*p.group*8
                timeline.send(t,e.core,e.peer,size)
                timeline.work(e.peer,t+2,vectors=6*r.area+3*r.rows*p.group,read=2*size,write=size)
            elif e.kind == 'normalize':
                timeline.work(e.core,t,vectors=r.area,read=r.area*4,write=r.area*2)
            elif e.kind == 'output_send':
                timeline.send(t,e.core,e.peer,r.area*2)
            elif e.kind == 'finish':
                out = outviews[(*e.key[:3],e.core)].crop(r.row,r.col,r.rows,r.cols)
                timeline.memory(t+1,out,True)
        graph.add_timeline(name,timeline,category='attention',sram_bytes=plan.sram_bytes(p),
                           score_elements_issued=coverage,compute_tiles=tiles,score_dram_bytes=0,
                           historical_kv_write_bytes=0,query_parts=plan.query_parts,context_parts=plan.context_parts,
                           query_shift=plan.query_shift,kv_shift=plan.kv_shift,output=plan.output,
                           dataflow=plan.dataflow,overlap=True,packet_peak_with_scratch=peak,
                           query_load_noc_bytes=query_noc,
                           kv_remote_load_bytes=sum(e.region.area*2 for e in self.events if e.kind=='kv_move' or
                               e.kind=='kv_restore' and e.storage.core!=e.core),
                           rotation_copies=sum(e.kind in ('q_copy','kv_copy') for e in self.events),
                           rotation_copy_bytes=sum(e.storage.area*2 for e in self.events if e.kind in ('q_copy','kv_copy')))
        graph.release(inputs,keep=query)
        for core,label in scratch:graph.live[core].pop(label,None)
        return outputs

    def replay(self, q, k, v):
        """Replay the same packet movements and updates, including causal masking.

        Arrays have shape (batch, tokens, heads, head_dim). This is a numeric
        oracle for schedule semantics, not part of simulator performance.
        """
        import numpy as np
        p = self.layout.problem; state = {}; kv = {}; seen = set(); finished = {}; backing={}
        assert q.shape == (p.batch,p.queries,p.query_heads,p.head_dim)
        assert k.shape == v.shape == (p.batch,p.context,p.kv_heads,p.head_dim)
        output = np.empty_like(q)
        def address(r):
            return r.core,r.name,r.local_row,r.local_col,r.rows,r.cols
        def read_kv(r,storage):
            if (storage.core,storage.name) in self.scratch:
                label,value=backing[address(storage)]
                assert label==(r.row,r.col,r.rows,r.cols)
                return value
            b=r.row//p.context; h=r.col//(2*p.head_dim); lo=r.row-b*p.context
            assert storage in self.cache or any(storage.intersect(x)==storage for x in self.cache)
            return lo,k[b,lo:lo+r.rows,h],v[b,lo:lo+r.rows,h]
        def merge(a,b):
            m = np.maximum(a[0],b[0]); finite = np.isfinite(m)
            x = np.zeros_like(m); y = np.zeros_like(m)
            dx = np.zeros_like(m); dy = np.zeros_like(m)
            np.subtract(a[0],m,out=dx,where=finite); np.subtract(b[0],m,out=dy,where=finite)
            np.exp(dx,out=x,where=finite); np.exp(dy,out=y,where=finite)
            return m,a[1]*x+b[1]*y,a[2]*x[...,None]+b[2]*y[...,None]
        for _,events in groupby(self.events,key=lambda e:e.time):
            for e in events:
                at = (e.core,e.key); r = e.region
                if e.kind == 'q_load':
                    h,b,s,w,c = e.key
                    data = q[b,r.row-b*p.queries:r.row-b*p.queries+r.rows,h*p.group:(h+1)*p.group]
                    shape = data.shape[:-1]
                    state[at] = (data,(np.full(shape,-np.inf),np.zeros(shape),np.zeros_like(data)))
                elif e.kind == 'kv_load':
                    h,b,*_ = e.key; lo = r.row-b*p.context
                    kv[at] = (lo,k[b,lo:lo+r.rows,h],v[b,lo:lo+r.rows,h])
                elif e.kind == 'kv_restore':
                    kv[at]=read_kv(r,e.storage)
                elif e.kind == 'kv_copy':
                    backing[address(e.storage)]=((r.row,r.col,r.rows,r.cols),read_kv(r,e.source))
                elif e.kind == 'q_save':
                    backing[address(e.storage)]=(e.key,state.pop(at))
                elif e.kind == 'q_restore':
                    label,value=backing[address(e.storage)]
                    assert label==e.key
                    state[at]=value
                elif e.kind == 'q_copy':
                    label,value=backing[address(e.source)]
                    assert label==e.key
                    backing[address(e.storage)]=(label,value)
                elif e.kind in ('q_move','kv_move'):
                    table = state if e.kind == 'q_move' else kv
                    table[e.peer,e.key] = table.pop(at)
                elif e.kind == 'kv_send':
                    kv[e.peer,e.key] = kv[at]
                elif e.kind == 'kv_drop':
                    del kv[at]
                elif e.kind == 'update':
                    data,old = state[at]; lo,keys,values = kv[e.core,e.other]
                    h,b,*_ = e.key
                    logits = np.einsum('qgd,kd->qgk',data,keys)/np.sqrt(p.head_dim)
                    for i in range(r.rows):
                        qi = r.row-b*p.queries+i
                        for j in range(len(keys)):
                            if p.causal and lo+j > p.context-p.queries+qi:
                                logits[i,:,j] = -np.inf
                            else:
                                pair = (h,b,qi,lo+j)
                                assert pair not in seen, ('Duplicate Q/KV pair',pair)
                                seen.add(pair)
                    m = logits.max(axis=-1); finite = np.isfinite(m)
                    z = np.zeros_like(logits)
                    np.subtract(logits,m[...,None],out=z,where=finite[...,None])
                    np.exp(z,out=z,where=finite[...,None]); z[~finite] = 0
                    part = (m,z.sum(axis=-1),np.einsum('qgk,kd->qgd',z,values))
                    state[at] = data,merge(old,part)
                elif e.kind == 'merge':
                    data,part = state.pop(at); dest = (e.peer,e.other)
                    qdst,old = state[dest]; state[dest] = qdst,merge(old,part)
                elif e.kind == 'normalize':
                    _,(_,l,u) = state.pop(at)
                    finished[at] = u/l[...,None]
                elif e.kind == 'output_send':
                    finished[e.peer,e.key] = finished[at]
                elif e.kind == 'finish':
                    h,b,*_ = e.key
                    start = r.row-b*p.queries
                    output[b,start:start+r.rows,h*p.group:(h+1)*p.group] = finished.pop(at)
        expected = p.batch*p.kv_heads*(sum(range(p.context-p.queries+1,p.context+1)) if p.causal else p.queries*p.context)
        assert len(seen) == expected and not state and not kv and not finished
        return output
