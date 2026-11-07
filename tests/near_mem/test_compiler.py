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


def test_flashgemm_tiles_only_k_when_the_whole_local_m_fits():
    from frontend.near_mem.gemm.flash import Geometry, candidates
    # OS fits the whole local M: only K tiles vary.
    problem=Problem(512,4096,14336);plans=list(candidates(problem,16,1<<20,16,'all'))
    assert {p.dataflow for p in plans}=={'os','ws'}
    os_plans=[p for p in plans if p.dataflow=='os']
    assert all(Geometry(problem,p).panels==1 and p.n_tile is None for p in os_plans)
    assert len({p.k_tile for p in os_plans})>1
    # WS cannot hold the whole local M with any K tile: M is split minimally,
    # one M tile shared by every proposed K tile.
    assert len({p.m_tile for p in plans if p.dataflow=='ws'})==1


def test_flashgemm_panels_only_when_the_paper_plan_does_not_fit():
    from frontend.near_mem.gemm.flash import candidates
    assert {p.n_tile for p in candidates(Problem(64,64,64),16,1024**2,4)}=={None}
    small=list(candidates(Problem(64,256,64),16,4096,4))
    assert small and all(p.n_tile for p in small)


def test_t10_single_copy_uses_1d_k_and_n_layout_for_m1():
    from frontend.near_mem.gemm.layout import GlobalLayout
    problem=Problem(1,32,32);layout=GlobalLayout(problem,16)
    assert layout.block('A')==(1,2)       # one K stripe per DRAM bank
    assert layout.block('B')==(32,2)      # full K, N striped over all banks
    assert layout.block('C')==(1,2)       # output N striped over all banks
    areas={'A':problem.m*problem.k,'B':problem.k*problem.n,'C':problem.m*problem.n}
    for tensor in 'ABC':
        regions=layout.regions(tensor)
        assert len(regions)==16
        assert sum(r.rows*r.cols for r in regions)==areas[tensor]

    plan=Plan((1,1,16),(1,1,1),layout='global')
    program=compile_plan(problem,plan,1024**2,1024**3)
    rng=np.random.default_rng(71)
    a=rng.normal(size=(1,32));b=rng.normal(size=(32,32))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)
    _,manifest=program.lower(64)
    assert manifest['macs']==problem.m*problem.n*problem.k


def test_t10_search_pads_only_k_n_tails_for_m1_and_keeps_n_split_plans():
    from frontend.near_mem.methods import METHODS
    logical=Problem(1,30,31)
    plans=list(METHODS['t10'].space(logical,16,1<<20,1<<30,tile_min=1))
    assert plans
    assert all(p.logical_shape==logical.shape and p.execution_shape==(1,32,32) for p in plans)
    assert any(p.ps==(1,16,1) for p in plans)
    program=compile_plan(logical,plans[0],1<<20,1<<30)
    rng=np.random.default_rng(73)
    a=rng.normal(size=(1,31));b=rng.normal(size=(31,30))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)
    assert sum(r.area for r in program.regions('C'))==logical.m*logical.n


def test_native_atlas_pads_only_when_its_mn_partition_space_is_empty():
    from frontend.near_mem.gemm.native import legal_partitions
    logical=Problem(1,30,31)
    physical,parts=legal_partitions(logical,16)
    assert physical.shape==(4,32,32) and parts
    exact,exact_parts=legal_partitions(Problem(1,32,32),16)
    assert exact.shape==(1,32,32) and exact_parts


def test_mesh_baseline_padding_is_explicit_and_replayed_as_zero_extension():
    from frontend.near_mem.gemm.plan import execution_problem
    logical=Problem(1,7,7);plan=Plan((1,2,2),(1,1,1),logical_shape=logical.shape,execution_shape=(1,8,8))
    physical=execution_problem(logical,plan)
    program=compile_plan(logical,plan,1024**2,1024**3)
    rng=np.random.default_rng(72)
    a=rng.normal(size=(1,7));b=rng.normal(size=(7,7))
    assert physical.shape==(1,8,8)
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-11)


def test_flashgemm_pads_nondivisible_dimensions_to_mesh_side():
    from frontend.near_mem.gemm.flash import candidates
    from frontend.near_mem.gemm.plan import execution_problem
    problem=Problem(1,30,31);plans=list(candidates(problem,16,1<<20,4))
    assert plans
    assert all(p.logical_shape==problem.shape and p.execution_shape==(4,32,32) for p in plans)
    assert all(execution_problem(problem,p).shape==(4,32,32) for p in plans)
    from frontend.near_mem.gemm.flash import FlashProgram
    program=FlashProgram(problem,plans[0],1<<20,1<<30)
    _,manifest=program.lower(64)
    assert manifest['macs']==4*32*32
    areas={'A':problem.m*problem.k,'B':problem.k*problem.n,'C':problem.m*problem.n}
    assert {tensor:sum(r.area for r in program.regions(tensor)) for tensor in 'ABC'}==areas


def test_flash_padding_keeps_graph_edges_at_logical_extent(tmp_path):
    from frontend.near_mem.gemm.flash import candidates
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.sim import make_hardware
    from frontend.near_mem.tasks import Graph,Region,unique
    problem=Problem(1,30,31);plan=next(candidates(problem,16,1<<20,4))
    graph=Graph(make_hardware(tmp_path/'hw',16,64));graph.tensor(0,'input',[1,31])
    kernel=Gemm(problem,plan).prepare(graph,'flash')
    output=kernel.compile(graph,[Region(0,'input',0,0,1,31)])
    assert sum(r.area for r in kernel.input_layout(graph.cores)[0])==problem.m*problem.k
    assert sum(r.area for r in output)==problem.m*problem.n


def test_padded_flash_gated_gemm_allocates_inactive_mesh_scratch(tmp_path):
    from frontend.near_mem.gemm.flash import candidates
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.sim import ROOT, make_hardware
    from frontend.near_mem.tasks import Graph, Region, unique
    if not list((ROOT/'simulator/build/lib').glob('atlasim.*.so')):
        pytest.skip('Build ATLAS first')
    problem=Problem(1,16,8)
    plan=next(candidates(problem,4,128*1024,4))
    assert plan.logical_shape==problem.shape and plan.execution_shape==(2,16,8)
    graph=Graph(make_hardware(tmp_path/'hw',4,128,flit_size=64))
    graph.tensor(0,'input',[1,8])
    kernel=Gemm(problem,plan,gated=True).prepare(graph,'expand')
    output=kernel.compile(graph,[Region(0,'input',0,0,1,8)])
    assert sum(r.area for r in unique(output))==problem.m*(problem.n//2)
    result=graph.simulate(tmp_path/'execute')
    assert result['result']['e2e_cycles']>0


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


@pytest.mark.parametrize('direction',[0,1])
def test_fixed_tp_graph_pads_mesh_axes_and_keeps_logical_boundaries(tmp_path,direction):
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.sim import make_hardware
    from frontend.near_mem.tasks import Graph,Region,unique
    logical=Problem(1,5,7)
    plans=list(METHODS['tp'].space(logical,cores=4,sram=1<<20,dram=1<<30,direction=direction))
    assert plans and all(p.logical_shape==logical.shape and p.execution_shape==(1,6,8) for p in plans)
    plan=plans[0]
    graph=Graph(make_hardware(tmp_path/'hw',4,64));graph.tensor(0,'input',[1,7])
    kernel=Gemm(logical,plan,direction=direction).prepare(graph,'tp')
    output=kernel.compile(graph,[Region(0,'input',0,0,1,7)])
    assert sum(r.area for r in unique(kernel.input_layout(4)[0]))==logical.m*logical.k
    assert sum(r.area for r in unique(output))==logical.m*logical.n
    assert graph.operations[-1]['metadata']['macs']==1*6*8
    next_problem=Problem(1,3,5)
    next_plan=next(METHODS['tp'].space(next_problem,cores=4,sram=1<<20,dram=1<<30,direction=direction))
    next_kernel=Gemm(next_problem,next_plan,direction=direction).prepare(graph,'tp_next')
    next_output=next_kernel.compile(graph,output)
    assert sum(r.area for r in unique(next_output))==next_problem.m*next_problem.n


def test_standalone_tp_pads_only_when_no_mn_partition_is_legal():
    from frontend.near_mem.methods import METHODS
    logical=Problem(1,3,5)
    plans=list(METHODS['tp'].space(logical,cores=4,sram=1<<20,dram=1<<30))
    assert plans
    assert all(p.logical_shape==logical.shape and p.execution_shape==(1,4,5) for p in plans)


def test_native_tp_graph_can_execute_a_padded_k_n_grid(tmp_path):
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.sim import make_hardware
    from frontend.near_mem.tasks import Graph,Region,unique
    logical=Problem(1,5,7)
    plan=next(METHODS['tp'].space(logical,cores=4,sram=1<<20,dram=1<<30,direction=0))
    graph=Graph(make_hardware(tmp_path/'hw',4,64));graph.tensor(0,'input',[1,7])
    kernel=Gemm(logical,plan,native=True,direction=0).prepare(graph,'tp_native')
    output=kernel.compile(graph,[Region(0,'input',0,0,1,7)])
    assert sum(r.area for r in unique(kernel.input_layout(4)[0]))==logical.m*logical.k
    assert sum(r.area for r in unique(output))==logical.m*logical.n
    assert any(op['metadata'].get('category')=='gemm' for op in graph.operations)


def test_native_transformer_tuner_pads_k_n_to_its_fixed_grid(tmp_path,monkeypatch):
    from frontend.near_mem.model import transformer
    from frontend.near_mem.sim import make_hardware
    calls=[]
    monkeypatch.setattr(transformer,'best_tile',lambda shape,*args,**kwargs:
                        (calls.append(shape) or {'tiling_factors':(1,1,1)}))
    workload=transformer.Workload(1,1,4,1,1,2,3,True)
    chip=make_hardware(tmp_path/'hw',16,64)
    kernels=transformer.tune_native(workload,chip,tmp_path/'tuning',workers=1)
    assert kernels['expand'].plan.logical_shape==workload.problems['expand'].shape
    assert kernels['expand'].plan.execution_shape==(1,8,4)
    assert kernels['contract'].plan.execution_shape==(1,4,4)
    assert kernels['qkv'].plan.execution_shape==(1,8,4)
    assert all(shape[1]>0 and shape[2]>0 for shape in calls)
    graph=transformer.build(workload,chip,transformer.configuration(kernels,transformer.AttentionPlan(1,1,1)))
    assert graph.operations


@pytest.mark.parametrize('native',[False,True])
def test_padded_gated_gemm_keeps_logical_gate_halves(tmp_path,native):
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.sim import make_hardware
    from frontend.near_mem.tasks import Graph,Region,unique
    logical=Problem(1,6,2)
    plan=Plan((1,2,2),(1,2,1),logical_shape=logical.shape,execution_shape=(1,8,2))
    graph=Graph(make_hardware(tmp_path/'hw',4,64));graph.tensor(0,'input',[1,2])
    kernel=Gemm(logical,plan,native=native,direction=0,gated=True).prepare(graph,'expand')
    output=kernel.compile(graph,[Region(0,'input',0,0,1,2)])
    assert sum(r.area for r in unique(output))==logical.m*(logical.n//2)
    if native:
        reduce=next(op for op in graph.operations if op['name']=='expand_reduce')
        assert not reduce['metadata']['gated']


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
