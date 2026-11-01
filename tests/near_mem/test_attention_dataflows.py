"""Fused baseline semantics, real transfers, capacity and graph boundaries."""
from dataclasses import replace

import numpy as np
import pytest

from frontend.near_mem.methods import METHODS
from frontend.near_mem.model.attention import Attention, AttentionPlan, KVLayout, build_attention, endpoints
from frontend.near_mem.model.attention_schedule import Schedule
from frontend.near_mem.model.attention_proxy import AttentionProxy
from frontend.near_mem.output import POLICIES
from frontend.near_mem.sim import make_hardware


FLOWS=('ring','t10','dram','flash_os','flash_ws')


def reference(p,q,k,v):
    k=np.repeat(k,p.group,axis=2); v=np.repeat(v,p.group,axis=2)
    score=np.einsum('bqhd,bkhd->bhqk',q,k)/np.sqrt(p.head_dim)
    if p.causal:
        mask=np.arange(p.context)[None,:]>p.context-p.queries+np.arange(p.queries)[:,None]
        score=np.where(mask,-np.inf,score)
    prob=np.exp(score-score.max(axis=-1,keepdims=True)); prob/=prob.sum(axis=-1,keepdims=True)
    return np.einsum('bhqk,bkhd->bqhd',prob,v)


@pytest.mark.parametrize('flow',FLOWS)
@pytest.mark.parametrize('queries,context',[(1,17),(5,19),(17,17)])
@pytest.mark.parametrize('causal',[False,True])
def test_fused_dataflows_match_attention_with_ragged_tiles(flow,queries,context,causal):
    p=Attention(2,queries,context,4,2,4,causal); layout=KVLayout(p,16)
    pq=1 if queries==1 and flow!='dram' else 2; pc=layout.group_cores//pq
    shifts=[(0,0)] if flow.startswith('flash') else [(0,1),(1,0)] if pq>1 and queries>1 else [(0,1)]
    rng=np.random.default_rng(2)
    q=rng.normal(size=(p.batch,p.queries,p.query_heads,p.head_dim))
    k=rng.normal(size=(p.batch,p.context,p.kv_heads,p.head_dim)); v=rng.normal(size=k.shape)
    expected=reference(p,q,k,v)
    for a,b in shifts:
        plan=AttentionPlan(pq,pc,2,2,a,b,dataflow=flow)
        schedule=Schedule(layout,plan)
        np.testing.assert_allclose(schedule.replay(q,k,v),expected,atol=1e-12)
        assert schedule.buffer_peak()<=plan.sram_bytes(p)


@pytest.mark.parametrize('flow',FLOWS)
@pytest.mark.parametrize('output',POLICIES)
def test_fused_outputs_are_shared_across_methods(tmp_path,flow,output):
    p=Attention(1,9,23,4,1,4); layout=KVLayout(p,4)
    a,b=(0,0) if flow.startswith('flash') else (1,0)
    plan=AttentionPlan(2,2,2,3,a,b,output,flow)
    rng=np.random.default_rng(8)
    q=rng.normal(size=(1,9,4,4)); k=rng.normal(size=(1,23,1,4)); v=rng.normal(size=k.shape)
    schedule=Schedule(layout,plan)
    np.testing.assert_allclose(schedule.replay(q,k,v),reference(p,q,k,v),atol=1e-12)
    chip=make_hardware(tmp_path/'hw',4,16)
    graph=build_attention(p,chip,plan)
    meta=graph.operations[-1]['metadata']
    assert meta['score_dram_bytes']==meta['historical_kv_write_bytes']==0
    assert meta['packet_peak_with_scratch']<=meta['sram_bytes']<=graph.sram
    assert sum(r.area for r in endpoints(layout,plan)[1])==p.queries*p.hidden*(2 if output=='replicated' else 1)
    for op in graph.operations:
        for core,phases in enumerate(op['raw']):
            for phase in phases:
                for task in phase['dram']:
                    tensor=graph.tensors[core][task['name']]
                    assert all(0<=a and a+n<=limit for a,n,limit in
                               zip(task['access_base'],task['access_extent'],tensor['shape']))
                    assert not (task['name']=='kv' and task['is_write'])
    assert not any('work_' in label for names in graph.live for label in names)


def test_dram_rotation_volume_is_independent_of_compute_tiles():
    p=Attention(1,16,64,4,1,8); layout=KVLayout(p,4); volumes=[]; chunks=[]
    for tq,tk in [(1,1),(2,4),(4,8)]:
        plan=AttentionPlan(4,1,tq,tk,1,2,dataflow='dram')
        s=Schedule(layout,plan)
        copies=[e for e in s.events if e.kind in ('kv_copy','q_copy')]
        volumes.append(tuple(sum(e.storage.area*2 for e in copies if e.kind==kind) for kind in ('kv_copy','q_copy')))
        chunks.append(len(copies))
        assert s.buffer_peak()<=plan.sram_bytes(p)
        # A copied KV block is read again by the receiving compute core.
        copied={e.storage for e in copies if e.kind=='kv_copy'}
        assert copied and all(any(e.kind=='kv_restore' and e.storage==r for e in s.events) for r in copied)
    assert len(set(volumes))==1 and len(set(chunks))==3


@pytest.mark.parametrize('flow',['ring','t10','dram'])
def test_joint_rotation_preserves_state_across_shards_and_query_waves(flow):
    p=Attention(2,17,29,8,2,4); layout=KVLayout(p,16)
    rng=np.random.default_rng(11)
    q=rng.normal(size=(p.batch,p.queries,p.query_heads,p.head_dim))
    k=rng.normal(size=(p.batch,p.context,p.kv_heads,p.head_dim)); v=rng.normal(size=k.shape)
    for a,b in [(1,2),(2,1),(3,2)]:
        plan=AttentionPlan(4,2,2,2,a,b,dataflow=flow)
        schedule=Schedule(layout,plan)
        np.testing.assert_allclose(schedule.replay(q,k,v),reference(p,q,k,v),atol=1e-12)
        assert schedule.buffer_peak()<=plan.sram_bytes(p)


@pytest.mark.parametrize('capacity',[1024,16384])
def test_method_spaces_keep_their_dataflows_and_dram_never_falls_back(capacity):
    p=Attention(1,16,64,4,1,8); layout=KVLayout(p,4)
    expected={'tp':{'ring'},'local':{'ring'},'ours':{'ring'},'dram':{'dram'},
              't10':{'t10'},'flash':{'flash_os','flash_ws'}}
    for name,flows in expected.items():
        space=METHODS[name].attention(layout,capacity)
        assert space and {p.dataflow for p in space}==flows
        for plan in space:
            plan.validate(layout,capacity)
            assert Schedule(layout,plan).buffer_peak()<=plan.sram_bytes(p)
    assert not METHODS['dram'].includes
    assert all(p.query_parts>1 for p in METHODS['dram'].attention(layout,16384))
    assert any(p.query_shift for p in METHODS['t10'].attention(layout,16384))
    assert any(p.query_shift and p.kv_shift for p in METHODS['t10'].attention(layout,16384))
    assert all(not p.query_shift and not p.kv_shift for p in METHODS['flash'].attention(layout,16384))
    decode=KVLayout(Attention(1,1,64,4,1,8),4)
    assert all(p.query_shift==0 for name in expected for p in METHODS[name].attention(decode,16384))


def test_t10_queries_have_fixed_homes_and_real_remote_reads(tmp_path):
    p=Attention(1,16,64,4,1,8); layout=KVLayout(p,4)
    plan=AttentionPlan(1,4,4,4,dataflow='t10')
    assert endpoints(layout,plan)[0]==endpoints(layout,replace(plan,query_parts=2,context_parts=2))[0]
    assert sum(r.area for r in endpoints(layout,plan)[0])==p.queries*p.hidden
    chip=make_hardware(tmp_path/'hw',4,16)
    g=build_attention(p,chip,plan,placed=True)
    assert g.operations[-1]['metadata']['query_load_noc_bytes']==p.queries*p.hidden*2*3
    assert not any(op['metadata'].get('noc_data_bytes') for op in g.operations[:-1])


def test_flash_gathers_each_panel_before_use_and_preserves_next_panel_overlap():
    p=Attention(1,16,64,4,1,8); layout=KVLayout(p,4)
    plan=AttentionPlan(2,2,2,2,0,0,dataflow='flash_os'); s=Schedule(layout,plan)
    assert not any(e.kind in ('q_move','q_copy','kv_copy') for e in s.events)
    # Each wave may gather the same logical panel again. Check its local
    # load-to-use interval, not the last send from a later query wave.
    sends=[e for e in s.events if e.kind=='kv_send']
    assert sends
    assert any(a.kind=='kv_load' and any(b.kind=='update' and b.time<=a.time<b.time+4 and
               b.other[3]==a.key[3]-1 for b in s.events) for a in s.events)
    ready={}
    for e in s.events:
        if e.kind=='kv_load':ready[e.core,e.key]=e.time+1
        elif e.kind in ('kv_move','kv_send'):
            assert ready[e.core,e.key]<=e.time
            ready[e.peer,e.key]=e.time+1
            if e.kind=='kv_move':del ready[e.core,e.key]
        elif e.kind=='kv_drop':del ready[e.core,e.key]
        elif e.kind=='update':assert ready[e.core,e.other]<=e.time


@pytest.mark.parametrize('flow',FLOWS)
def test_fused_dataflows_run_on_the_native_simulator(tmp_path,flow):
    chip=make_hardware(tmp_path/'hw',4,16)
    p=Attention(1,8,16,4,1,8)
    a,b=(0,0) if flow.startswith('flash') else (1,0) if flow=='t10' else (0,1)
    plan=AttentionPlan(2,2,2,2,a,b,dataflow=flow)
    graph=build_attention(p,chip,plan)
    meta=graph.operations[-1]['metadata']
    result=graph.simulate(tmp_path/'run')['result']
    assert result['e2e_cycles']>0
    assert result['flop_count']==sum(2*op['metadata'].get('macs',0)+op['metadata'].get('vector_ops',0)
                                      for op in graph.operations)
    assert (meta['rotation_copy_bytes']>0)==(flow=='dram')
    proxy=AttentionProxy(chip)
    assert proxy.score(p,plan)>0


@pytest.mark.parametrize('flow',FLOWS)
def test_transformer_layout_edges_match_new_attention_ports(tmp_path,flow):
    from frontend.near_mem.gemm.plan import Plan
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.model.transformer import Workload,build,configuration,edge_regions,port
    from frontend.near_mem.tasks import traffic
    chip=make_hardware(tmp_path/'hw',4,16)
    w=Workload(1,8,16,4,1,8,64)
    kernels={name:Gemm(p,Plan((1,2,2),(4,8,8)),direction=int(name=='expand'),gated=name=='expand')
             for name,p in w.problems.items()}
    a,b=(0,0) if flow.startswith('flash') else (1,0)
    attention=AttentionPlan(2,2,2,2,a,b,output='replicated',dataflow=flow)
    graph=build(w,chip,configuration(kernels,attention))
    for src,dst,name in [('qkv.C','attention.Q','attention_query_layout'),
                         ('attention.O','output.A','output_A_layout')]:
        def choice(endpoint):
            op=endpoint.split('.')[0]
            return attention if op=='attention' else kernels[op].bind(4,graph.sram,graph.capacity)
        regions=edge_regions(w,4,dst,port(w,4,src,choice(src)),port(w,4,dst,choice(dst)))
        estimated=sum(x[1]*2 for load in traffic(*regions,True) for x in load.values())
        measured=next(e['dram_write_bytes'] for e in graph.edges if e['name']==name)
        assert estimated==measured
    graph.simulate(tmp_path/'graph')
