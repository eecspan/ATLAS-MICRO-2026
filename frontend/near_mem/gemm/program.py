"""Value-carrying tiled program shared by scheduling and backend execution."""
from dataclasses import dataclass
from math import prod

from .plan import DEPENDENCIES


@dataclass
class Program:
    problem: object
    plan: object
    phases: list
    placements: list
    sram_bytes: int

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
                        incoming.append((op['dst'],op.get('target',v),{k:x.copy() for k,x in sram[core][v].items()}))
                    elif kind=='add':
                        for key,value in sram[core]['partial'].items():sram[core]['C'][key]+=value
                        sram[core]['partial']={}
                    elif kind=='store':
                        key=tuple(op['key']);value=sram[core]['C'][key]
                        dram[core][key]=value.copy()
                        if op.get('final'):
                            m,n=key;tm,tn=op['shape']
                            output[m*tm:(m+1)*tm,n*tn:(n+1)*tn]=value
                            written[m*tm:(m+1)*tm,n*tn:(n+1)*tn]+=1
            for core,v,value in incoming:sram[core][v]=value
        assert macs==p.m*p.n*p.k
        assert np.all(written==1)
        return output
