from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from frontend.near_mem.gemm.plan import Plan, Problem
from frontend.near_mem.gemm.compiler import compile_plan


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
    from frontend.near_mem.gemm.lower import simulate
    from frontend.near_mem.sim import ROOT,make_hardware
    if not list((ROOT/'simulator/build/lib').glob('atlasim.*.so')):pytest.skip('Build ATLAS first')
    problem=Problem(16,32,32)
    chip=make_hardware(tmp_path/'hardware',16)
    result=simulate(problem,plan,chip,tmp_path/'execute')
    assert result['manifest']['macs']==16*32*32
    assert result['result']['noc_cycles']>0
    assert result['result']['e2e_cycles']>0


@pytest.mark.parametrize('plan',[Plan((2,2,1),(2,2,2),a=(1,2),b=(2,1),receive='dram_land'),
                                Plan((1,4,4),(2,2,2),a=(2,2),receive='dram_land'),
                                Plan((2,2,1),(2,2,2),a=(1,2),b=(2,1),layout='global'),
                                Plan((1,4,4),(2,2,2),a=(2,2),c=(2,2),layout='global')])
def test_baseline_variants_values_and_lowering(plan):
    from frontend.near_mem.gemm.lower import lower_tasks
    problem=Problem(16,16,16)
    program=compile_plan(problem,plan,1024**2,1024**3)
    rng=np.random.default_rng(5)
    a,b=rng.normal(size=(16,16)),rng.normal(size=(16,16))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)
    _,manifest=lower_tasks(program,64)
    assert manifest['macs']==16**3


@pytest.mark.parametrize('dataflow,mt,kt,resident,nt',[('os',4,8,False,None),('ws',8,4,False,None),
                                                       ('ws',4,4,True,None),('os',4,8,False,4),('ws',8,4,False,2)])
def test_flashgemm_program(dataflow,mt,kt,resident,nt):
    from frontend.near_mem.gemm.flash import FlashProgram
    from frontend.near_mem.gemm.plan import FlashPlan
    program=FlashProgram(Problem(32,32,32),FlashPlan(dataflow,mt,kt,resident,16,n_tile=nt),1024**2,1024**3)
    _,manifest=program.lower(64)
    assert manifest['macs']==32**3
    assert manifest['logical_dram_write_bytes']==32*32*2


def test_flashgemm_panels_only_when_the_paper_plan_does_not_fit():
    from frontend.near_mem.gemm.flash import candidates
    assert {p.n_tile for p in candidates(Problem(64,64,64),16,1024**2,4)}=={None}
    small=list(candidates(Problem(64,256,64),16,4096,4))
    assert small and all(p.n_tile for p in small)


@pytest.mark.parametrize('direction',[0,1])
def test_tp_graph_plan_matches_native_layout(direction):
    """The TP grid reads native A chunks; its reduce-scatter output lies inside the native all-reduce output."""
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.model.kernels import Gemm
    problem=Problem(16,64,2048)
    plans=list(METHODS['tp'].space(problem,cores=4,sram=1<<22,dram=1<<30,direction=direction))
    assert plans and all(p.ps==(1,2,2) and p.loop=='mnk' and p.transpose==(direction==0) for p in plans)
    ours,native=(Gemm(problem,plans[0],native=n,direction=direction).bind(4,1<<22,1<<30) for n in (False,True))
    key=lambda rs:sorted((r.core,r.row,r.col,r.rows,r.cols) for r in rs)
    assert key(ours.input_layout(4)[0])==key(native.input_layout(4)[0])
    held={r.core:range(r.col,r.col+r.cols) for r in native.output_layout(4)}
    assert all(r.col in held[r.core] and r.col+r.cols-1 in held[r.core] for r in ours.output_layout(4))


@pytest.mark.parametrize('plan',[Plan((1,1,4),(4,4,2)),Plan((1,2,2),(2,4,4)),Plan((2,1,2),(2,2,4),loop='kmn'),
                                Plan((1,2,2),(2,2,2),a=(1,2),c=(2,1)),
                                Plan((1,2,2),(2,2,2),c=(2,1),receive='dram_land'),
                                Plan((1,1,4),(4,4,2),c=(2,2),receive='dram_land'),
                                Plan((1,2,2),(2,2,2),a=(1,2),c=(2,1),receive='dram_land')])
def test_split_k_reduce_scatter_and_landed_c(plan):
    from frontend.near_mem.gemm.lower import lower_tasks
    problem=Problem(16,16,16)
    program=compile_plan(problem,plan,1024**2,1024**3)
    rng=np.random.default_rng(7)
    a,b=rng.normal(size=(16,16)),rng.normal(size=(16,16))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)
    _,manifest=lower_tasks(program,64)
    assert manifest['macs']==16**3
