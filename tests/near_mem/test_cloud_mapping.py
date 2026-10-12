from itertools import permutations
import json
from math import prod

import numpy as np
import pytest

from frontend.auto.tiling_explorer_cloud import make_gemm_description
from frontend.near_mem.cloud_mapping import ring_program, ring_replacement
from frontend.near_mem.cloud_compare import execute
from frontend.near_mem.plan import Plan
from frontend.near_mem.run import make_hardware
from types import SimpleNamespace


def boundary():
    m,n,k=8,12,16
    tensors=[dict(name='input_qkv_proj',shape=[m,k],strides=[k,1],element_size=2,base_addr=0),
             dict(name='weight_qkv_proj_b0',shape=[k,n],strides=[1,k],element_size=2,base_addr=131072),
             dict(name='output_qkv_proj',shape=[m,n],strides=[n,1],element_size=2,base_addr=262144)]
    op=make_gemm_description(SimpleNamespace(name='qkv_proj',M=m,N=n,K=k),2,(2,2,3),
                             tensors[0]['name'],tensors[2]['name'],tensors[1]['name'])
    return op,dict(tensor=tensors)


@pytest.mark.parametrize('a',[(1,2),(1,4),(2,1),(2,2),(4,1)])
@pytest.mark.parametrize('loop',[''.join(x) for x in permutations('mnk')])
def test_ring_keeps_native_input_weight_and_partial_output_homes(tmp_path,a,loop):
    chip=make_hardware(tmp_path/'hardware',16)
    op,placement=boundary();plan=Plan((1,4,1),(2,3,2),a=a,loop=loop)
    program,_,_,side=ring_program(op,placement,plan,chip)
    rng=np.random.default_rng(9);A=rng.normal(size=(8,64));B=rng.normal(size=(64,48))
    result=np.zeros((8,48))
    for krow in range(side):
        local_a=A[:,krow*16:(krow+1)*16];local_b=B[krow*16:(krow+1)*16,:]
        result+=program.replay(local_a,local_b)
        for phase in program.phases:
            for col,actions in enumerate(phase):
                for action in actions:
                    if action['op']!='load' or action['tensor']=='C':continue
                    x,y=action['base'];dx,dy=action['extent'];i,j=action['key']
                    matrix=local_a if action['tensor']=='A' else local_b[:,col*12:(col+1)*12]
                    reference=local_a if action['tensor']=='A' else local_b
                    np.testing.assert_array_equal(matrix[x:x+dx,y:y+dy],reference[i*dx:(i+1)*dx,j*dy:(j+1)*dy])
    np.testing.assert_allclose(result,A@B,atol=1e-11)


def test_ring_native_packets_are_confined_to_n_groups(tmp_path):
    chip=make_hardware(tmp_path/'hardware',16)
    op,placement=boundary();plan=Plan((1,4,1),(2,3,2),a=(1,4))
    repl,manifest=ring_replacement(op,placement,plan,chip,tmp_path/'tasks')
    sends=[];receives=[]
    for core in range(16):
        tasks=json.loads((tmp_path/f'tasks/core_{core}.yaml').read_text())
        for phase in tasks['execution']:
            for key,dest in (('noc_tx',sends),('noc_rx',receives)):
                for t in phase[key]:
                    assert t['src']//4==t['dst']//4==core//4
                    dest.append((t['src'],t['dst'],t['init_flit_id'],t['flit_num']))
    assert sorted(sends)==sorted(receives)
    ids=sorted((t[2],t[2]+t[3]) for t in sends)
    assert all(a[1]<=b[0] for a,b in zip(ids,ids[1:]))
    result=execute(chip,tmp_path/'run',dict(operator=[repl]),placement)
    assert result['flop_count']==(2*manifest['macs']+manifest['vector_ops'])*manifest['replicas']
    assert result['noc_cycles']>0
