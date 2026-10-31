"""Value-carrying tiled program shared by scheduling and backend execution."""
from dataclasses import dataclass
from math import prod

from .plan import DEPENDENCIES
from ..tasks import Region, coalesce


@dataclass
class Program:
    problem: object
    plan: object
    phases: list
    placements: list
    sram_bytes: int
    overlap: bool = True

    @property
    def suffix(self):
        """Final outputs go to ``C+suffix``: a global layout writes them to their homes."""
        return 'g' if self.plan.layout=='global' else ''

    @property
    def gate_tile(self):
        """N extent of each final output tile."""
        return self.plan.tile[1]

    def regions(self, tensor):
        """Global windows of ``tensor`` in DRAM: inputs as read, C as finally written."""
        from .layout import GlobalLayout
        from .ownership import Ownership
        p=self.plan
        if p.layout=='global':
            return GlobalLayout(self.problem,p.cores).regions(tensor)
        regions=[]
        if tensor=='C':
            for phase in self.phases:
                for core,actions in enumerate(phase):
                    for a in actions:
                        if a['op']=='store' and a.get('final'):
                            regions.append(Region(core,'C',a['key'][0]*p.tile[0],a['key'][1]*p.tile[1],
                                                  *a['extent'],*a['base']))
                        elif a['op']=='forward' and a.get('final'):  # a landed C copied home
                            regions.append(Region(a['dst'],'C',a['key'][0]*p.tile[0],a['key'][1]*p.tile[1],
                                                  *a['extent'],*a['base']))
        else:
            own=Ownership(self.problem,p);axes=DEPENDENCIES[tensor];shape=[own.base[d] for d in axes]
            for core in range(p.cores):
                blocks=own.tensor_blocks(core,tensor)
                indices=[sorted({b[i] for b in blocks}) for i in range(2)]
                for block in blocks:
                    regions.append(Region(core,tensor,block[0]*shape[0],block[1]*shape[1],*shape,
                                          indices[0].index(block[0])*shape[0],indices[1].index(block[1])*shape[1]))
        return coalesce(regions)

    def lower(self, flit, final_ops=None):
        from .lower import lower_tasks
        return lower_tasks(self,flit,final_ops)

    def replay(self, a, b):
        import numpy as np
        p=self.problem;plan=self.plan
        if a.shape!=(p.m,p.k) or b.shape!=(p.k,p.n):
            raise ValueError('Input shape mismatch')
        sram=[{v:{} for v in ('A','B','C','partial')} for _ in range(plan.cores)]
        dram=[{} for _ in range(plan.cores)]
        output=np.zeros((p.m,p.n));written=np.zeros((p.m,p.n),dtype=int)
        macs=0
        for phase in self.phases:
            incoming=[]
            for core,actions in enumerate(phase):
                for op in actions:
                    kind=op['op'];v=op.get('tensor')
                    if kind=='clear':
                        sram[core][v]={key:sram[core][v][key] for key in op.get('keep',[])}
                    elif kind=='drop':
                        del sram[core][v][op['key']]
                    elif kind in ('load','zero'):
                        key=tuple(op['key']);shape=op['shape']
                        if kind=='zero':value=np.zeros(shape)
                        elif v=='C':value=dram[core][key].copy()
                        else:
                            pos=[k*t for k,t in zip(key,shape)]
                            value=(a if v=='A' else b)[pos[0]:pos[0]+shape[0],pos[1]:pos[1]+shape[1]].copy()
                        sram[core][v][key]=value
                    elif kind=='gemm':
                        m,n,k=op['index']
                        sram[core]['C'][m,n]+=sram[core]['A'][m,k]@sram[core]['B'][k,n]
                        macs+=prod(plan.tile)
                    elif kind=='send':
                        keys=[tuple(op['key'])] if 'key' in op else list(sram[core][v])
                        incoming.append((op['dst'],op.get('target',v),{k:sram[core][v][k].copy() for k in keys}))
                    elif kind=='copy':
                        key=tuple(op['key'])
                        incoming.append((op['dst'],'replica',{key:sram[core][v][key].copy()}))
                    elif kind=='finalize':
                        pass  # graph epilogues are injected only during lowering
                    elif kind=='add':
                        key=tuple(op['key'])
                        if key in sram[core]['partial']:sram[core]['C'][key]+=sram[core]['partial'].pop(key)
                    elif kind=='forward' and v=='C':
                        key=tuple(op['key']);dram[op['dst']][key]=value=dram[core][key].copy()
                        if op.get('final'):
                            m,n=key;tm,tn=op['extent']
                            output[m*tm:(m+1)*tm,n*tn:(n+1)*tn]=value
                            written[m*tm:(m+1)*tm,n*tn:(n+1)*tn]+=1
                    elif kind=='store':
                        key=tuple(op['key']);value=sram[core]['C'][key]
                        dram[core][key]=value.copy()
                        if op.get('final'):
                            m,n=key;tm,tn=op['shape']
                            previous=written[m*tm:(m+1)*tm,n*tn:(n+1)*tn]
                            if np.any(previous):
                                np.testing.assert_allclose(output[m*tm:(m+1)*tm,n*tn:(n+1)*tn],value)
                            output[m*tm:(m+1)*tm,n*tn:(n+1)*tn]=value
                            written[m*tm:(m+1)*tm,n*tn:(n+1)*tn]+=1
            for core,v,value in incoming:
                if v=='replica':
                    sram[core]['C'].update(value)
                elif v=='partial':  # received partial sums accumulate per tile
                    for k,x in value.items():sram[core][v][k]=sram[core][v].get(k,0)+x
                else:sram[core][v]=value
        assert macs==p.m*p.n*p.k
        copies=plan.ps[2]//prod(plan.c) if plan.output=='replicated' else 1
        assert np.all(written==copies)
        return output
