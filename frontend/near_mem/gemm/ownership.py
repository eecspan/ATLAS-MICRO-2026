"""Executable ownership algebra for arbitrary legal multi-axis Ps/Pt.

This is a numerical correctness oracle, NOT a latency evaluator or a replacement
for lowering actual loads, stores, buffers and NoC messages into native ATLAS.
It makes unequal periods and C rotation precise before backend migration.

Transfer order. Per axis, the periods of its two tensors form gcd layers: a
shared layer (both rotate) and, for unequal periods, a solo layer (only the
longer-period tensor). In general each layer moves its own mixed-radix digit,
so the solo layer jumps by the shared radix. When an axis's shared layer is
the outermost and its solo layer the next (the order the ELK rule gives a
plan whose only rotations are on that axis), the axis instead follows T10's
uniform rings: the longer tensor moves one core per step on its whole ring
and the shorter one every ``q`` steps (rotating paces); every core starts at
``x_long + q*x_short``. Both orders visit the same blocks with the same
transfers; the uniform one keeps every hop between ring neighbours.
"""
from itertools import product
from math import isqrt, prod

from .plan import DEPENDENCIES, REPLICA_AXIS


class Ownership:
    def __init__(self, problem, plan):
        plan.validate(problem)
        self.problem,self.plan=problem,plan
        self.layers=plan.layers
        coords=list(product(*(range(n) for n in plan.ps)))
        if plan.transpose:
            # Physical core r*s+c takes the coordinates of logical core c*s+r.
            s=isqrt(len(coords));coords=[coords[(j%s)*s+j//s] for j in range(len(coords))]
        self.coords=coords
        self.ids={c:i for i,c in enumerate(self.coords)}
        self.uniform=self._uniform()
        self.base=tuple(n//(sp*tp) for n,sp,tp in zip(problem.shape,plan.ps,plan.periods))
        self.strides={v:{} for v in DEPENDENCIES}
        for v in DEPENDENCIES:
            stride=1
            for i,layer in enumerate(self.layers):
                if v in layer.tensors:
                    self.strides[v][i]=stride
                    stride*=layer.radix

    def _uniform(self):
        """``(shared, solo, long, short, p, q)`` of the axis on uniform rings, or None."""
        order=self.plan.layer_order or tuple(range(len(self.layers)))
        if len(order)<2:return None
        shared,solo=self.layers[order[0]],self.layers[order[1]]
        if shared.axis!=solo.axis or len(shared.tensors)!=2 or len(solo.tensors)!=1:return None
        long=solo.tensors[0];short=next(v for v in shared.tensors if v!=long)
        return order[0],order[1],long,short,shared.radix,solo.radix

    def ring(self, core, tensor):
        """Position of the long tensor on its whole uniform ring of length p*q."""
        i,_,_,_,p,q=self.uniform
        return (self.coords[core][REPLICA_AXIS[tensor]]//self.strides[tensor][i])%(p*q)

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
        index=[self.axis_index(ds,d) for d in range(3)]
        if self.uniform:
            # Steps taken on the uniform axis during its single traversal.
            i,j,long,short,p,q=self.uniform
            step=state[i]*q+(state[j]+state[i])%q
            index[self.layers[i].axis]=(self.ring(core,long)+q*self.digit(core,i,short)+step)%(p*q)
        return tuple(c*p+x for c,p,x in zip(self.coords[core],self.plan.periods,index))

    def neighbor(self, core, layer, tensor, delta):
        """Move only this tensor's replica position (its layer digit, or its whole
        uniform ring); preserve other axes."""
        c=list(self.coords[core]);d=REPLICA_AXIS[tensor]
        if self.uniform and layer in self.uniform[:2] and tensor==self.uniform[2]:
            i=self.uniform[0];size=self.uniform[4]*self.uniform[5]
            old=self.ring(core,tensor);c[d]+=((old+delta)%size-old)*self.strides[tensor][i]
            return self.ids[tuple(c)]
        old=self.digit(core,layer,tensor)
        new=(old+delta)%self.layers[layer].radix
        c[d]+=(new-old)*self.strides[tensor][layer]
        return self.ids[tuple(c)]

    def homing(self, state):
        """Shifts per layer that return every tensor to its initial placement."""
        counts=[(-s)%layer.radix for s,layer in zip(state,self.layers)]
        if self.uniform:
            i,j,_,_,p,q=self.uniform
            step=state[i]*q+(state[j]+state[i])%q
            counts[i]=(-state[i])%p;counts[j]=(-(step+counts[i]))%(p*q)
        return counts

    def tensor_blocks(self, core, tensor):
        """Blocks a core holds initially: layers moving ``tensor`` at state 0, all states of the others."""
        choices=[(0,) if tensor in layer.tensors else range(layer.radix) for layer in self.layers]
        axes=DEPENDENCIES[tensor]
        return sorted({tuple(self.block(core,state)[d] for d in axes) for state in product(*choices)})

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
