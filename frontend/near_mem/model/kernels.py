"""Graph adapters for the full GEMM compiler and the original ATLAS pipeline."""
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isqrt

from ..gemm.compiler import compile_kernel
from ..tasks import Region, coalesce, transfers, unique, window
from ..gemm.plan import Problem, execution_problem
from ..sim import strides


def program_regions(program, tensor):
    return program.regions(tensor)


def bind_input(graph, name, source, regions, tensors):
    """Retain exact compatible producer storage; materialize other partitions."""
    target = []; names = {}
    for core, tensor in enumerate(tensors):
        expected = [r for r in regions if r.core == core]
        pairs = transfers(source, expected)
        candidates = {a.name for a,b in pairs if a.core == core}
        alias = next(iter(candidates)) if len(candidates) == 1 else None
        if alias is not None:
            physical = graph.tensors[core][alias]
            if (physical['shape'] != tensor['shape'] or physical['strides'] != tensor['strides']
                    or any(a.core != core or a.name != alias or (a.local_row, a.local_col) != (b.local_row, b.local_col)
                           for a,b in pairs)):
                alias = None
        if alias is None:
            alias = name
            graph.tensor(core, alias, tensor['shape'], tensor['strides'])
        names[core] = alias
        target.extend(replace(r, name=alias) for r in expected)
    graph.convert(name+'_layout', source, target)
    return target, names


@dataclass
class Gemm:
    problem: Problem
    plan: object
    native: bool = False
    direction: int = 0
    gated: bool = False

    def bind(self, cores, sram, capacity):
        """Compile the mapping for a chip; touches no graph state."""
        self.execution_problem = execution_problem(self.problem,self.plan)
        if self.gated and self.problem.n % 2:
            raise ValueError('Gated projection needs equal gate/up widths')
        if self.native:
            self.side = isqrt(cores)
            if self.side*self.side != cores or self.side < 2:
                raise ValueError('The original ATLAS collective requires an even square mesh')
            m,n,k = self.execution_problem.shape
            if n % self.side or k % self.side:
                raise ValueError('Native K/N partitions require mesh-divisible execution dimensions')
        else:
            self.program = compile_kernel(self.problem, self.plan, sram, capacity)
            # Static parameter packing: each even N tile contains corresponding
            # gate/up halves. All partial sums retain the ordinary GEMM layout.
            self.fused_gate = (self.gated and self.program.gate_tile % 2 == 0
                               and self.execution_problem.n == self.problem.n)
        return self

    def prepare(self, graph, name):
        """Allocate persistent weights before any activation scratch."""
        self.bind(graph.cores, graph.sram, graph.capacity); self.name = name
        if self.native:
            m,n,k = self.execution_problem.shape; shape = [k//self.side, n//self.side]
            for core in range(graph.cores):
                graph.tensor(core, name+'_B', shape, column=True)
        else:
            for core, placement in enumerate(self.program.placements):
                b = next(t for t in placement['tensor'] if t['name'] == 'B')
                graph.tensor(core, name+'_B', b['shape'], b['strides'])
        return self

    def input_layout(self, cores):
        """A regions this kernel reads in place, and the per-core A tensors."""
        if not self.native:
            p = self.program
            return program_regions(p, 'A'), [next(t for t in c['tensor'] if t['name'] == 'A') for c in p.placements]
        m,_,k = self.execution_problem.shape; logical_k = self.problem.k; local_k = k//self.side; regions = []
        for core in range(cores):
            row,col = divmod(core, self.side)
            kc = row if self.direction == 0 else col
            start = kc*local_k; width = max(0,min(local_k,logical_k-start))
            if width:
                regions.append(Region(core, 'A', 0, start, self.problem.m, width))
        return regions, [dict(shape=[m,local_k], strides=strides([m,local_k]))]*cores

    def output_layout(self, cores):
        """Final output regions (after any gate), as returned by ``compile``."""
        m,n,_ = self.problem.shape
        if self.native:
            _,execution_n,_ = self.execution_problem.shape
            local_n=execution_n//self.side
            fused=self._native_fused_gate(local_n)
            regions=self._native_output_regions(cores,fused)
            return unique(window(regions,0,0,m,n//2)) if self.gated and not fused else regions
        regions = program_regions(self.program, 'C')
        if self.fused_gate:
            return [replace(r, col=r.col//2, cols=r.cols//2, local_col=r.local_col//2) for r in regions]
        # An unfused gate pairs values at the owners of the gate half.
        return unique(window(regions, 0, 0, m, n//2)) if self.gated else regions

    def _native_fused_gate(self, local_n):
        """Whether ATLAS's packed reduce can keep both logical gate halves local."""
        return self.gated and self.execution_problem.n==self.problem.n and local_n%2==0

    def _native_output_regions(self, cores, gated):
        _,execution_n,_=self.execution_problem.shape
        local_n=execution_n//self.side
        width=local_n//2 if gated else local_n
        columns=self.problem.n//2 if gated else self.problem.n
        regions=[]
        for core in range(cores):
            col=(core%self.side if self.direction==0 else core//self.side)*width
            extent=max(0,min(width,columns-col))
            if extent:regions.append(Region(core,'C',0,col,self.problem.m,extent))
        return regions

    def compile(self, graph, source, rope_width=0):
        if self.native:
            return self._native(graph, source)
        p = self.program; out = 'C'+p.suffix
        if rope_width and (rope_width % 2 or p.gate_tile % 2):
            raise ValueError('Fused RoPE requires complete even-sized Q column tiles')
        inputs, names = bind_input(graph, self.name+'_A', source, *self.input_layout(graph.cores))
        outputs = program_regions(p, 'C')
        active = {r.core for r in outputs}
        scratch = []
        for core, placement in enumerate(p.placements):
            for t in placement['tensor']:
                if t['name'] in ('A', 'B'):
                    continue
                graph.tensor(core, self.name+'_'+t['name'], t['shape'], t['strides'])
                if t['name'] == out and self.fused_gate:
                    graph.tensor(core, self.name+'_G'+p.suffix, [t['shape'][0], t['shape'][1]//2])
                    # A padded FlashGEMM mesh row still executes and writes its
                    # physical gate tile, even when that row has no logical
                    # output. Keep that scratch allocated for the kernel, then
                    # release it with the other non-output tensors below.
                    if core not in active:
                        scratch.append((core, self.name+'_G'+p.suffix))
                if t['name'] != out or self.fused_gate or core not in active:
                    scratch.append((core, self.name+'_'+t['name']))
        def epilogue(action):
            if self.fused_gate:
                count = action['shape'][0]*action['shape'][1]//2
                final = dict(action,tensor='G',**{key:[action[key][0],action[key][1]//2]
                                                for key in ('base','extent','shape')})
                return final,5*count,4*count,2*count
            column = action['key'][1]*action['shape'][1]
            count = action['shape'][0]*max(0,min(action['shape'][1],rope_width-column))
            return action,6*count,2*count,2*count
        raw, manifest = p.lower(graph.flit,epilogue if rope_width or self.fused_gate else None)
        for core, phases in enumerate(raw):
            for phase in phases:
                for task in phase['dram']:
                    task['name'] = names[core] if task['name'] == 'A' else self.name+'_'+task['name']
        graph.add(self.name, raw, dict(manifest, category='gemm', native=False,
                  fused_gate=self.fused_gate,weight_columns='gate/up halves per N tile' if self.fused_gate else 'logical'))
        for core, name in scratch:
            graph.live[core].pop(name, None)
        graph.release(inputs, keep=source)
        if self.gated and not self.fused_gate:
            return gate(graph, [replace(r,name=self.name+'_'+out) for r in outputs], self.problem.m, self.problem.n//2, self.name+'_gate')
        final = self.name+('_G' if self.fused_gate else '_C')+p.suffix
        return [replace(r,name=final) for r in self.output_layout(graph.cores)]

    def _native(self, graph, source, reduce=True):
        from frontend.auto.tiling_explorer_cloud import make_gemm_description
        from frontend.auto.generalize import expand_gemm
        from frontend.auto.collectives import allreduce_tasks
        from frontend.model_parser import Operator, OperatorType
        m,n,k = self.execution_problem.shape; side = self.side
        local_k, local_n = k//side, n//side
        fused_gate=self._native_fused_gate(local_n)
        inputs, names = bind_input(graph, self.name+'_A', source, *self.input_layout(graph.cores))
        raw = []; outputs = []
        for core in range(graph.cores):
            partial = graph.tensor(core, self.name+'_partial', [m,local_n])
            result = graph.tensor(core, self.name+'_C', [m,local_n//(2 if fused_gate else 1)]) if reduce else partial
            op = Operator(name=self.name, op_type=OperatorType.GEMM, B=1, M=m, N=local_n, K=local_k)
            tm,tn,tk = self.plan.tile
            native = make_gemm_description(op, 2, (tm,tk,tn), names[core], partial['name'], self.name+'_B')
            raw.append(expand_gemm(native, list(graph.tensors[core].values()))['execution'])
            if not reduce:
                nc = core % side if self.direction == 0 else core//side
                outputs.append(Region(core, partial['name'], 0, nc*local_n, m, local_n))
        if reduce:
            regions=(self._native_output_regions(graph.cores,fused_gate) if self.gated and not fused_gate
                     else self.output_layout(graph.cores))
            outputs = [replace(r, name=result['name']) for r in regions]
        graph.add(self.name, raw, dict(category='gemm', native=True, tile=self.plan.tile))
        if not reduce:
            graph.release(inputs, keep=source)
            return outputs
        op, raw = allreduce_tasks(partial, result, self.name, m, 0, local_n, 2, 1,
                                 self.direction, side, graph.flit, [0]*graph.cores, gated=fused_gate)
        graph.add(self.name+'_reduce', raw, dict(category='collective', native=True, gated=fused_gate),
                  kind='communication', operator=op)
        graph.release(inputs, keep=source)
        graph.release([Region(core, partial['name'], 0,0,m,local_n) for core in range(graph.cores)])
        if self.gated and not fused_gate:
            return gate(graph, outputs, self.problem.m, self.problem.n//2, self.name+'_gate')
        return outputs


    def qkv(self, graph, source, layout, cache, rope=False):
        p = layout.problem
        if not self.native or p.kv_heads > 1 or self.plan.logical_shape is not None:
            # Head-partitioned KV cannot use the native packed-KV collective,
            # which assumes every core holds all heads in one row partition.
            output = self.compile(graph, source,rope_width=p.hidden if rope and not self.native else 0)
            if rope and self.native:
                query = window(output, 0, 0, p.batch*p.queries, p.hidden)
                pointwise(graph,self.name+'_rope_q',query,6)
            layout.append(graph, output, cache, self.name+'_kv_append')
            query = window(output, 0, 0, p.batch*p.queries, p.hidden)
            return query, output
        # Preserve ATLAS's optimized Q allreduce/allgather and KV tree
        # reduce/scatter/gather/scatter. Only the append destinations change.
        from frontend.auto.tiling_explorer_cloud import generate_q_proj_comm, generate_kv_proj_comm
        if self.direction != 0:
            raise ValueError('Native QKV protocol expects the forward mesh direction')
        if p.kv_heads != 1:
            raise AssertionError('Head-partitioned KV requires the compiled QKV path')
        side = self.side; rows = p.batch*p.queries
        if p.hidden % side or 2*p.kv_heads*p.head_dim % side:
            raise ValueError('Native packed Q/KV columns must divide mesh side')
        partials = self._native(graph, source, reduce=False)
        query = []
        for core in range(graph.cores):
            q = graph.tensor(core, self.name+'_Q', [rows,p.hidden])
            query.append(Region(core,q['name'],0,0,rows,p.hidden))
        partial = graph.tensors[0][partials[0].name]
        qwidth = p.hidden//side; kvwidth = 2*p.kv_heads*p.head_dim//side
        qcomm = generate_q_proj_comm(partial,q,self.name,rows,0,qwidth,2,1,0,
                                    str(graph.chip),graph.cores,side,graph.flit,[0]*graph.cores,'',build_only=True)
        if rope:
            # Fuse Q RoPE at each completed reduce-scatter chunk, before either
            # gather creates replicas. This performs the transform exactly once.
            if qwidth % (2*side):
                raise ValueError('Native fused RoPE requires complete even-sized Q chunks')
            for core,tasks in enumerate(qcomm[1][0][1]):
                rx = tasks['communication']['on_chip'][side-2]['rx']
                count = rows*qwidth//side
                rx['vector'].append(dict(name='rope',vec_count=6*count))
                rx['buffer_load'].append(dict(name='rope',byte_count=2*count,is_write=False))
                rx['buffer_store'].append(dict(name='rope',byte_count=2*count,is_write=True))
        for op, raw in qcomm[1]:
            graph.add(op['name'],raw,dict(category='collective',native=True),kind='communication',operator=op)
        slots = [[[] for _ in range(side)] for _ in range(side)]
        for request in range(p.batch):
            for token in range(p.context-p.queries,p.context):
                row,col = divmod(layout.token_owner(request,0,token),side)
                slots[row][col].append(layout.token_slot(request,0,token))
        placement = graph.tensors[0][cache[0].name]
        kvcomm = generate_kv_proj_comm(partial,placement,self.name,rows,qwidth,kvwidth,
                                       layout.capacity,p.kv_heads,2,1,str(graph.chip),
                                       graph.cores,side,graph.flit,[0]*graph.cores,slots,'',build_only=True)
        for op, raw in kvcomm[1]:
            graph.add(op['name'],raw,dict(category='kv_append',native=True,
                      appended_kv_bytes=rows*2*p.kv_heads*p.head_dim*2),kind='communication',operator=op)
        graph.release(partials)
        return query, query


def gate(graph, source, rows, width, name):
    """Pair completed gate/up values at output owners, then apply SiLU(gate)*up."""
    left = unique(window(source, 0, 0, rows, width))
    right = window(source, 0, width, rows, width)
    target = []; output = []
    for i,r in enumerate(left):
        graph.tensor(r.core, name+f'_up{i}', [r.rows,r.cols])
        graph.tensor(r.core, name+f'_out{i}', [r.rows,r.cols])
        target.append(replace(r, name=name+f'_up{i}', local_row=0, local_col=0))
        output.append(replace(r, name=name+f'_out{i}', local_row=0, local_col=0))
    graph.convert(name+'_pair', right, target)
    timeline = graph.timeline(); t = 0
    for a,b,c in zip(left,target,output):
        max_cols = max(1,graph.sram//8)
        for col in range(0,a.cols,max_cols):
            width = min(max_cols,a.cols-col); max_rows = max(1,graph.sram//(8*width))
            for start in range(0,a.rows,max_rows):
                count = min(max_rows,a.rows-start)
                x = a.crop(a.row+start,a.col+col,count,width)
                y = b.crop(x.row,x.col,x.rows,x.cols); z = c.crop(x.row,x.col,x.rows,x.cols)
                timeline.memory(t,x); timeline.memory(t,y)
                timeline.work(a.core,t+1,vectors=5*x.area,read=4*x.area,write=2*x.area)
                timeline.memory(t+2,z,True); t += 3
    graph.add_timeline(name,timeline,category='activation')
    graph.release(target); graph.release(source)
    return output


def pointwise(graph, name, regions, operations):
    """In-place vector work, tiled to bounded SRAM; no tensor layout change."""
    timeline = graph.timeline(); times = [0]*graph.cores
    for region in coalesce(regions):
        width = min(region.cols, graph.sram//8)
        for col in range(0, region.cols, width):
            nc = min(width, region.cols-col)
            step = max(1, graph.sram//(8*nc))
            for row in range(0, region.rows, step):
                tile = region.crop(region.row+row, region.col+col, min(step,region.rows-row), nc)
                t = times[region.core]
                timeline.memory(t,tile)
                timeline.work(region.core,t+1,vectors=operations*tile.area,read=2*tile.area,write=2*tile.area)
                timeline.memory(t+2,tile,True); times[region.core] += 3
    graph.add_timeline(name,timeline,category='vector')
    return regions


def residual(graph, name, value, skip):
    targets = []
    for i,r in enumerate(value):
        graph.tensor(r.core,name+f'_skip{i}',[r.rows,r.cols])
        targets.append(replace(r,name=name+f'_skip{i}',local_row=0,local_col=0))
    graph.convert(name+'_layout',skip,targets)
    timeline = graph.timeline(); times = [0]*graph.cores
    for a,b in zip(value,targets):
        width = min(a.cols,graph.sram//12)
        for col in range(0,a.cols,width):
            nc = min(width,a.cols-col); height = max(1,graph.sram//(12*nc))
            for row in range(0,a.rows,height):
                x = a.crop(a.row+row,a.col+col,min(height,a.rows-row),nc)
                y = b.crop(x.row,x.col,x.rows,x.cols); t = times[a.core]
                timeline.memory(t,x); timeline.memory(t,y)
                timeline.work(a.core,t+1,vectors=x.area,read=4*x.area,write=2*x.area)
                timeline.memory(t+2,x,True); times[a.core] += 3
    graph.add_timeline(name,timeline,category='residual')
    graph.release(targets)
    return value


def rmsnorm(graph, name, source, rows, width):
    """Distributed row statistics, preserving the producer's activation layout.

    Full activations are never gathered merely for normalization. FP32 row
    statistics reduce across unique column shards, then broadcast to replicas.
    Two passes over DRAM are conservative and common to both compiler paths.
    """
    chosen = unique(source); output = []; weights = []
    for i,r in enumerate(source):
        graph.tensor(r.core,name+f'_out{i}',[r.rows,r.cols])
        graph.tensor(r.core,name+f'_gamma{i}',[1,r.cols],persistent=True)
        output.append(replace(r,name=name+f'_out{i}',local_row=0,local_col=0))
        weights.append(Region(r.core,name+f'_gamma{i}',0,r.col,1,r.cols))
    boundaries = sorted({0,rows,*(r.row for r in source),*(r.row+r.rows for r in source)})
    timeline = graph.timeline(); t = 0
    for start,end in zip(boundaries,boundaries[1:]):
        step = max(1,graph.sram//(8*width))
        if width*8 > graph.sram:
            raise ValueError('RMSNorm row needs column streaming for this unusually small SRAM')
        for first in range(start,end,step):
            count = min(step,end-first); participants = set(); readers = set()
            box = Region(0,'',first,0,count,width)
            parts = [x for r in chosen if (x := r.intersect(box)) is not None]
            assert sum(r.area for r in parts) == count*width
            for r in parts:
                timeline.memory(t,r)
                timeline.work(r.core,t+1,vectors=2*r.area,read=2*r.area,write=count*4)
                participants.add(r.core)
            active = sorted(participants); t += 2
            stride = 1
            while stride < len(active):
                for i in range(0,len(active),2*stride):
                    if i+stride < len(active):
                        dst,src = active[i],active[i+stride]
                        timeline.send(t,src,dst,count*4)
                        timeline.work(dst,t+1,vectors=count,read=count*8,write=count*4)
                t += 2; stride *= 2
            root = active[0]
            timeline.work(root,t,vectors=count*3,read=count*4,write=count*4); t += 1
            for r in source:
                if r.intersect(box) is not None:
                    readers.add(r.core)
            # Tree broadcast includes cores that only hold a replicated shard.
            ready = [root]; remaining = sorted(readers-{root})
            while remaining:
                batch = remaining[:len(ready)]; remaining = remaining[len(ready):]
                for src,dst in zip(ready,batch):
                    timeline.send(t,src,dst,count*4)
                ready += batch; t += 1
            for src,dst,gamma in zip(source,output,weights):
                a = src.intersect(box)
                if a is None:
                    continue
                b = dst.crop(a.row,a.col,a.rows,a.cols)
                timeline.memory(t,a); timeline.memory(t,gamma)
                timeline.work(a.core,t+1,vectors=2*a.area,read=a.area*2+count*4+a.cols*2,write=a.area*2)
                timeline.memory(t+2,b,True)
            t += 3
    graph.add_timeline(name,timeline,category='normalization',statistics_dtype_bytes=4)
    graph.release(weights)
    return output
