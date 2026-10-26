"""Executable ownership algebra for arbitrary legal multi-axis Ps/Pt.

This is a numerical correctness oracle, NOT a latency evaluator or a replacement
for lowering actual loads, stores, buffers and NoC messages into native ATLAS.
It makes unequal periods and C rotation precise before backend migration.
"""
from itertools import product
from math import prod

from .plan import DEPENDENCIES, REPLICA_AXIS


class Ownership:
    def __init__(self, problem, plan):
        plan.validate(problem)
        self.problem,self.plan=problem,plan
        self.layers=plan.layers
        self.coords=list(product(*(range(n) for n in plan.ps)))
        self.ids={c:i for i,c in enumerate(self.coords)}
        self.base=tuple(n//(sp*tp) for n,sp,tp in zip(problem.shape,plan.ps,plan.periods))
        self.strides={v:{} for v in DEPENDENCIES}
        for v in DEPENDENCIES:
            stride=1
            for i,layer in enumerate(self.layers):
                if v in layer.tensors:
                    self.strides[v][i]=stride
                    stride*=layer.radix

    def digit(self, core, layer, tensor):
        return (self.coords[core][REPLICA_AXIS[tensor]]//self.strides[tensor][layer])%self.layers[layer].radix

    def digits(self, core, state):
        return tuple((sum(self.digit(core,i,v) for v in layer.tensors)+state[i])%layer.radix
                     for i,layer in enumerate(self.layers))

    def axis_index(self, digits, axis):
        value=0
        for d,layer in zip(digits,self.layers):
            if layer.axis==axis:
                value=value*layer.radix+d
        return value

    def block(self, core, state):
        ds=self.digits(core,state)
        return tuple(c*p+self.axis_index(ds,d) for d,(c,p) in enumerate(zip(self.coords[core],self.plan.periods)))

    def neighbor(self, core, layer, tensor, delta):
        """Move only this tensor's mixed-radix replica digit; preserve other axes."""
        c=list(self.coords[core]);d=REPLICA_AXIS[tensor]
        old=self.digit(core,layer,tensor)
        new=(old+delta)%self.layers[layer].radix
        c[d]+=(new-old)*self.strides[tensor][layer]
        return self.ids[tuple(c)]

    def tensor_blocks(self, core, tensor):
        initial=self.digits(core,(0,)*len(self.layers))
        choices=[(initial[i],) if tensor in layer.tensors else range(layer.radix)
                 for i,layer in enumerate(self.layers)]
        axes=DEPENDENCIES[tensor]
        blocks=set()
        for digits in product(*choices):
            blocks.add(tuple(self.coords[core][d]*self.plan.periods[d]+self.axis_index(digits,d) for d in axes))
        return sorted(blocks)

    def events(self):
        """No implicit bring-back: retain inner-layer phase after each outer step."""
        order=self.plan.layer_order or tuple(range(len(self.layers)))
        state=[0]*len(self.layers)
        def visit(depth):
            if depth==len(order):
                yield 'compute',tuple(state)
                return
            i=order[depth]
            for j in range(self.layers[i].radix):
                yield from visit(depth+1)
                if j<self.layers[i].radix-1:
                    yield 'shift',i
                    state[i]=(state[i]+1)%self.layers[i].radix
        yield from visit(0)

    def replay(self, a, b):
        import numpy as np
        p=self.problem
        if a.shape!=(p.m,p.k) or b.shape!=(p.k,p.n):
            raise ValueError('Input shape mismatch')
        tensors={v:[] for v in DEPENDENCIES}
        for v,axes in DEPENDENCIES.items():
            for core in range(self.plan.cores):
                blocks={}
                for key in self.tensor_blocks(core,v):
                    shape=tuple(self.base[d] for d in axes)
                    slices=tuple(slice(x*s,(x+1)*s) for x,s in zip(key,shape))
                    blocks[key]=(np.zeros(shape) if v=='C' else (a if v=='A' else b)[slices].copy())
                tensors[v].append(blocks)
        coverage=set()
        for op,arg in self.events():
            if op=='compute':
                for core in range(self.plan.cores):
                    m,n,k=self.block(core,arg)
                    if (m,n,k) in coverage:
                        raise AssertionError('Repeated MAC region')
                    coverage.add((m,n,k))
                    tensors['C'][core][m,n]+=tensors['A'][core][m,k]@tensors['B'][core][k,n]
            else:
                for v in self.layers[arg].tensors:
                    old=tensors[v]
                    tensors[v]=[old[self.neighbor(dst,arg,v,+1)] for dst in range(self.plan.cores)]
        expected=prod(n//s for n,s in zip(p.shape,self.base))
        assert len(coverage)==expected
        # Semantic reduction only: native execution must emit this communication
        # and restore/output ownership explicitly; this routine grants no timing.
        out=np.zeros((p.m,p.n));copies=np.zeros_like(out,dtype=int)
        for blocks in tensors['C']:
            for (m,n),value in blocks.items():
                sl=(slice(m*self.base[0],(m+1)*self.base[0]),slice(n*self.base[1],(n+1)*self.base[1]))
                out[sl]+=value;copies[sl]+=1
        assert np.all(copies==self.plan.ps[2]//prod(self.plan.c))
        return out
