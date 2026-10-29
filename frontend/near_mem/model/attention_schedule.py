"""Fused attention packet schedule, shared by value replay and ATLAS lowering.

Q and its online state are one mobile object; paired K/V are another. Each
context segment visits the Cartesian product of its Q and KV packets through
relative cyclic shifts. Context segments own disjoint tokens and merge once.
Only final normalized output is written to DRAM; KV owners never change.
"""
from dataclasses import dataclass, replace
from itertools import groupby
from math import ceil

from .attention import assignments, endpoints


def lane_core(layout, plan, head, request, lane, segment):
    start,_ = layout.head_group(head)
    return start+((request*plan.query_parts+lane)*plan.context_parts+segment)%layout.group_cores


def packets(layout, plan, head, request, segment, cache=None):
    """Per-lane lists of owner-local tiles, clipping both shard and segment edges."""
    p = layout.problem; count = plan.query_parts*plan.context_parts
    step = plan.kv_tile or ceil(p.context/count)
    cache = layout.regions() if cache is None else cache
    result = []
    for lane in range(plan.query_parts):
        part = segment*plan.query_parts+lane
        lo = request*p.context+p.context*part//count
        hi = request*p.context+p.context*(part+1)//count
        pieces = []
        for r in cache:
            if r.col != head*2*p.head_dim:
                continue
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


class Schedule:
    def __init__(self, layout, plan, cache=None):
        plan.validate(layout,plan.sram_bytes(layout.problem))
        self.layout = layout; self.plan = plan; self.events = []
        self.cache = layout.regions() if cache is None else cache
        self._build()
        self.events.sort(key=lambda e:e.time)

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
                        for turn in range(max(map(len,pieces[c]))):
                            kvpos = {}; kvregions = {}
                            for lane,tiles in enumerate(pieces[c]):
                                if turn >= len(tiles):
                                    continue
                                r = tiles[turn]; key = (h,b,c,turn,lane)
                                core = lane_core(self.layout,plan,h,b,lane,c)
                                self.events.append(Event(u,'kv_load',r.core,key,r))
                                if core != r.core:
                                    self.events.append(Event(u+1,'kv_move',r.core,key,r,core))
                                kvpos[key] = lane; kvregions[key] = r
                            u += 3
                            for stage in range(pq):
                                for q,qlane in qpos.items():
                                    if q[-1] != c:
                                        continue
                                    for k,klane in kvpos.items():
                                        if qlane == klane:
                                            core = lane_core(self.layout,plan,h,b,qlane,c)
                                            self.events.append(Event(u,'update',core,q,qregions[q],other=k))
                                u += 5
                                if stage == pq-1:
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
                                            self.events.append(Event(u,kind,src,key,regions[key],dst))
                                        positions[key] = dest
                                u += 2
                            for key,lane in kvpos.items():
                                self.events.append(Event(u,'kv_drop',lane_core(self.layout,plan,h,b,lane,c),key))
                        ends.append(u)
                    t = max(ends)+2
                    for s in range(pq):
                        dst = (h,b,s,wave,pc-1)
                        if dst not in qpos:
                            continue
                        core = lane_core(self.layout,plan,h,b,qpos[dst],pc-1)
                        for c in range(pc-1):
                            src = (h,b,s,wave,c)
                            owner = lane_core(self.layout,plan,h,b,qpos[src],c)
                            self.events.append(Event(t+3*c,'merge',owner,src,qregions[src],core,dst))
                            # Independent query groups merge at the same steps.
                        self.events.append(Event(t+3*(pc-1)+1,'finish',core,dst,qregions[dst]))
                    t += 3*(pc-1)+4

    def lower(self, graph, query, name):
        p,plan = self.layout.problem,self.plan
        inputs = []; qviews = {}; outputs = []; outviews = {}
        for h,b,s,c,r in assignments(self.layout,plan):
            qname = f'{name}_q_h{h}_b{b}_s{s}_c{c}'
            local = next((x for src in query if src.core == r.core and
                          (x := src.intersect(r)) is not None and x.area == r.area),None)
            if local is None:
                graph.tensor(r.core,qname,[r.rows,r.cols]); local = replace(r,name=qname)
            inputs.append(local); qviews[h,b,s,c] = local
        for r in endpoints(self.layout,plan)[1]:
            h = r.col//(p.group*p.head_dim); b = r.row//p.queries
            s = (r.row%p.queries)//ceil(p.queries/plan.query_parts)
            out = replace(r,name=f'{name}_o_h{h}_b{b}_s{s}')
            graph.tensor(out.core,out.name,[out.rows,out.cols])
            outputs.append(out); outviews[h,b,s] = out
        graph.convert(name+'_query_layout',query,inputs)
        timeline = graph.timeline(); coverage = tiles = 0
        kv = {e.key:e.region for e in self.events if e.kind == 'kv_load'}
        for e in self.events:
            t,r = e.time,e.region
            if e.kind == 'q_load':
                h,b,s,w,c = e.key; src = qviews[h,b,s,c]
                timeline.memory(t,src.crop(r.row,r.col,r.rows,r.cols))
            elif e.kind == 'kv_load':
                timeline.memory(t,r)
            elif e.kind in ('q_move','kv_move'):
                size = r.area*2 if e.kind == 'kv_move' else r.area*6+r.rows*p.group*8
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
            elif e.kind == 'finish':
                out = outviews[e.key[:3]].crop(r.row,r.col,r.rows,r.cols)
                timeline.work(e.core,t,vectors=r.area,read=r.area*4,write=r.area*2)
                timeline.memory(t+1,out,True)
        graph.add_timeline(name,timeline,category='attention',sram_bytes=plan.sram_bytes(p),
                           score_elements_issued=coverage,compute_tiles=tiles,score_dram_bytes=0,
                           historical_kv_write_bytes=0,query_parts=plan.query_parts,context_parts=plan.context_parts,
                           query_shift=plan.query_shift,kv_shift=plan.kv_shift)
        graph.release(inputs,keep=query)
        return outputs

    def replay(self, q, k, v):
        """Replay the same packet movements and updates, including causal masking.

        Arrays have shape (batch, tokens, heads, head_dim). This is a numeric
        oracle for schedule semantics, not part of simulator performance.
        """
        import numpy as np
        p = self.layout.problem; state = {}; kv = {}; seen = set()
        assert q.shape == (p.batch,p.queries,p.query_heads,p.head_dim)
        assert k.shape == v.shape == (p.batch,p.context,p.kv_heads,p.head_dim)
        output = np.empty_like(q)
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
                elif e.kind in ('q_move','kv_move'):
                    table = state if e.kind == 'q_move' else kv
                    table[e.peer,e.key] = table.pop(at)
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
                elif e.kind == 'finish':
                    _,(_,l,u) = state.pop(at); h,b,*_ = e.key
                    start = r.row-b*p.queries
                    output[b,start:start+r.rows,h*p.group:(h+1)*p.group] = u/l[...,None]
        expected = p.batch*p.kv_heads*(sum(range(p.context-p.queries+1,p.context+1)) if p.causal else p.queries*p.context)
        assert len(seen) == expected and not state and not kv
        return output
