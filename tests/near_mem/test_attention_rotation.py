"""Joint rotations must visit each legal pair once and carry online state."""
from dataclasses import replace

import numpy as np
import pytest

from frontend.near_mem.model.attention import Attention, AttentionPlan, KVLayout, plans, endpoints, compile_attention
from frontend.near_mem.model.attention_schedule import Schedule
from frontend.near_mem.tasks import Graph, Region
from frontend.near_mem.sim import make_hardware


@pytest.mark.parametrize('queries,context,causal',[(9,19,True),(8,8,True),(7,13,False),(1,17,True)])
def test_rotating_packets_match_dense_attention(queries,context,causal):
    p = Attention(2,queries,context,4,1,4,causal)
    layout = KVLayout(p,8); rng = np.random.default_rng(37)
    q = rng.normal(size=(p.batch,queries,4,4)); k = rng.normal(size=(p.batch,context,1,4)); v = rng.normal(size=k.shape)
    scores = np.einsum('bqhd,bkd->bqhk',q,k[:,:,0])/2
    if causal:
        mask = np.arange(context)[None,:] > context-queries+np.arange(queries)[:,None]
        scores = np.where(mask[None,:,None,:],-np.inf,scores)
    weights = np.exp(scores-scores.max(axis=-1,keepdims=True)); weights /= weights.sum(axis=-1,keepdims=True)
    expected = np.einsum('bqhk,bkd->bqhd',weights,v[:,:,0])
    for pq in (1,2,4):
        if pq > queries:
            continue
        for pc in (1,2):
            shifts = [(0,1)] if pq == 1 else [(0,1),(1,0)]+([(1,2),(2,1),(3,2)] if pq == 4 else [])
            for a,b in shifts:
                plan = AttentionPlan(pq,pc,2,3,a,b)
                with np.errstate(invalid='ignore'):
                    np.testing.assert_allclose(Schedule(layout,plan).replay(q,k,v),expected,atol=1e-12)


def test_space_nests_and_rejects_incomplete_pair_coverage():
    layout = KVLayout(Attention(1,16,32,4,1,8),4)
    native = set(plans(layout,16384,False)); old = set(plans(layout,16384,joint=False)); new = set(plans(layout,16384))
    assert native < old < new
    assert any(p.query_shift and p.kv_shift for p in new)
    with pytest.raises(ValueError,match='cover'):
        AttentionPlan(4,1,1,4,1,3).validate(layout,16384)
    for p in new:
        p.validate(layout,16384)
        with pytest.raises(ValueError,match='SRAM'):
            p.validate(layout,p.sram_bytes(layout.problem)-1)


@pytest.mark.parametrize('shift',[(0,1),(1,0),(1,2),(2,1)])
def test_joint_rotation_lowering_and_output_port(tmp_path,shift):
    chip = make_hardware(tmp_path/'chip',4,32)
    graph = Graph(chip); p = Attention(1,9,19,4,1,8)
    layout = KVLayout(p,4); cache = layout.allocate(graph)
    graph.tensor(0,'q',[9,32]); q = [Region(0,'q',0,0,9,32)]
    plan = AttentionPlan(4,1,2,3,*shift)
    result = compile_attention(graph,layout,plan,q,cache)
    sig = lambda rs: sorted((r.core,r.row,r.col,r.rows,r.cols) for r in rs)
    assert sig(result) == sig(endpoints(layout,plan)[1])
    meta = graph.operations[-1]['metadata']
    assert meta['macs'] >= 2*4*8*sum(range(11,20))
    assert meta['macs'] == 2*8*meta['score_elements_issued']
    assert meta['dram_write_bytes'] == 9*32*2
    for core,phases in enumerate(graph.operations[-1]['raw']):
        for phase in phases:
            for task in phase['dram']:
                tensor = graph.tensors[core][task['name']]
                assert all(0 <= a and a+n <= d for a,n,d in zip(task['access_base'],task['access_extent'],tensor['shape']))
    measured = graph.simulate(tmp_path/'run')['result']
    assert measured['operators']['attention']['flop_count'] == 2*meta['macs']+meta['vector_ops']


def test_joint_attention_runs_inside_transformer(tmp_path):
    from frontend.near_mem.gemm.plan import Plan
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.model.transformer import Workload, build, configuration
    chip = make_hardware(tmp_path/'chip',4,64)
    w = Workload(1,16,16,4,1,8,32)
    kernels = {name:Gemm(problem,Plan((1,2,2),(4,8,8)),gated=name=='expand')
               for name,problem in w.problems.items()}
    cfg = configuration(kernels,AttentionPlan(4,1,2,2,1,2))
    graph = build(w,chip,cfg)
    result = graph.simulate(tmp_path/'layer')['result']
    assert result['e2e_cycles'] > 0
    assert sum(op['span'] for op in result['operators'].values()) == result['e2e_cycles']
