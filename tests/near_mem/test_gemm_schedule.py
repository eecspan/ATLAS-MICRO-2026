from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from frontend.auto.gemm_schedule import schedule_gemm
from frontend.auto.generalize import expand_gemm
from frontend.auto.tiling_explorer_cloud import make_gemm_description


@pytest.mark.parametrize('shape,tile', [((8,12,16),(4,4,8)), ((8,12,16),(8,12,16)),
                                       ((8,12,16),(8,4,16)), ((8,12,16),(4,12,16)), ((8,12,16),(4,4,16))])
def test_mnk_is_exact_native_task_expansion(shape,tile):
    m,n,k=shape;tm,tn,tk=tile
    native=make_gemm_description(SimpleNamespace(name='gemm',M=m,N=n,K=k),2,(tm,tk,tn),'A','C','B')
    tensors=[dict(name='A',shape=[m,k]),dict(name='B',shape=[k,n]),dict(name='C',shape=[m,n])]
    actual,_=schedule_gemm(shape,tile)
    assert actual==expand_gemm(native,tensors)


@pytest.mark.parametrize('loop',[''.join(x) for x in permutations('mnk')])
@pytest.mark.parametrize('shape,tile',[((8,12,16),(4,4,8)),((4,8,12),(4,4,4)),((8,4,12),(4,4,4))])
def test_actual_memory_timeline_and_all_six_orders(loop,shape,tile):
    m,n,k=shape;tm,tn,tk=tile
    rng=np.random.default_rng(42);a=rng.normal(size=(m,k));b=rng.normal(size=(k,n))
    core,computes=schedule_gemm(shape,tile,loop)
    events={};comp=dict(computes)
    for raw,phase in enumerate(core['execution']):
        for task in phase['dram']:
            events.setdefault(raw+(2 if task['is_write'] else 0),[]).append(task)
    dram={};produced={};loaded={};ready={};visited=set();last_write={};active=None;value=None
    macs=0
    for time in range(core['iteration']+2):
        loaded.update(ready.pop(time,{}))
        for task in events.get(time,[]):
            v=task['name'];base=task['access_base'];extent=task['access_extent']
            key=tuple(x//t for x,t in zip(base,extent));tag=(v,key)
            if task['is_write']:
                assert v=='C' and produced[key][0]<time
                dram[key]=produced[key][1].copy();last_write[key]=time
            else:
                if v=='C':
                    assert key in dram and last_write[key]<time
                    data=dram[key].copy()
                else:
                    x,y=base;dx,dy=extent
                    data=(a if v=='A' else b)[x:x+dx,y:y+dy].copy()
                ready.setdefault(time+1,{})[tag]=data
        if time in comp:
            mi,ni,ki=comp[time];key=(mi,ni)
            if active!=key:
                value=loaded.pop(('C',key)).copy() if key in visited else np.zeros((tm,tn))
                active=key
            value+=loaded['A',(mi,ki)]@loaded['B',(ki,ni)]
            produced[key]=(time,value.copy());visited.add(key);macs+=tm*tn*tk
    output=np.empty((m,n))
    for (mi,ni),v in dram.items():output[mi*tm:(mi+1)*tm,ni*tn:(ni+1)*tn]=v
    assert macs==m*n*k
    np.testing.assert_allclose(output,a@b,atol=1e-12)


def test_two_output_k_outer_order_inserts_write_read_dependency_gap():
    core,computes=schedule_gemm((4,8,12),(4,4,4),'kmn')
    assert core['iteration']>len(computes)
