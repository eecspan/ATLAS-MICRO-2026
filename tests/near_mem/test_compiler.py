from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from frontend.near_mem.plan import Plan, Problem
from frontend.near_mem.compiler import compile_plan


PLANS=[Plan((1,1,1),(2,2,2)),Plan((1,2,3),(2,2,2)),
       Plan((2,2,1),(2,2,2),a=(1,2),b=(2,1)),
       Plan((2,4,1),(2,2,2),a=(1,4),b=(2,1)),
       Plan((1,4,4),(2,2,2),a=(2,2),c=(2,2)),
       Plan((2,2,2),(2,2,2),a=(2,1),b=(1,2),c=(2,1))]


@pytest.mark.parametrize('plan',PLANS)
@pytest.mark.parametrize('loop',[''.join(p) for p in permutations('mnk')])
@pytest.mark.parametrize('overlap',[False,True])
def test_general_program_values(plan,loop,overlap):
    problem=Problem(16,16,48)
    plan=replace(plan,loop=loop)
    program=compile_plan(problem,plan,1024**2,1024**3,overlap=overlap)
    rng=np.random.default_rng(31)
    a,b=rng.normal(size=(16,48)),rng.normal(size=(48,16))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)


@pytest.mark.parametrize('loop',[''.join(p) for p in permutations('mnk')])
def test_grouped_epochs_and_layer_orders(loop):
    problem=Problem(16,16,16)
    p=Plan((2,2,2),(2,2,2),a=(2,1),b=(1,2),c=(2,1),loop=loop)
    group=[1,1,1];group['mnk'.index(loop[-1])]=2
    p=replace(p,group=tuple(group))
    a=np.arange(256).reshape(16,16)/41;b=np.sin(a)
    for order in p.orders():
        program=compile_plan(problem,replace(p,layer_order=order),65536,1024**2)
        np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)


def test_buffer_constraints_and_spill_endpoint_fail_closed():
    problem=Problem(16,16,16);p=Plan((2,2,1),(2,2,2),a=(1,2),b=(2,1),group=(1,1,2))
    program=compile_plan(problem,p,65536,1024**2)
    with pytest.raises(ValueError,match='SRAM'):compile_plan(problem,p,program.sram_bytes-1,1024**2)
    with pytest.raises(ValueError,match='Partial-panel'):compile_plan(problem,replace(p,panel=(1,1,1)),65536,1024**2)


@pytest.mark.parametrize('plan',[Plan((1,4,4),(4,4,4),a=(2,2),c=(2,2)),
                                Plan((2,4,2),(4,4,4),a=(1,4),b=(2,1),c=(2,1),loop='knm')])
def test_general_native_backend(tmp_path,plan):
    from frontend.near_mem.backend import simulate
    from frontend.near_mem.run import ROOT,make_hardware
    if not list((ROOT/'simulator/build/lib').glob('atlasim.*.so')):pytest.skip('Build ATLAS first')
    problem=Problem(16,32,32)
    chip=make_hardware(tmp_path/'hardware',16)
    result=simulate(problem,plan,chip,tmp_path/'execute')
    assert result['manifest']['macs']==16*32*32
    assert result['result']['noc_cycles']>0
    assert result['result']['e2e_cycles']>0
