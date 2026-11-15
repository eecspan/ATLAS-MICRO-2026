"""Output ownership is searched jointly with consumers, using real traffic."""
from dataclasses import asdict, replace
import json
from math import prod

import numpy as np
import pytest

from frontend.near_mem.output import POLICIES, variants
from frontend.near_mem.gemm.plan import Plan, Problem
from frontend.near_mem.gemm.compiler import compile_plan
from frontend.near_mem.model.attention import Attention, AttentionPlan, KVLayout, endpoints
from frontend.near_mem.model.attention_schedule import Schedule
from frontend.near_mem.sim import make_hardware


@pytest.mark.parametrize('output',POLICIES)
@pytest.mark.parametrize('plan',[Plan((1,1,4),(2,4,2)),Plan((1,2,4),(2,2,2),c=(1,2)),
                                 Plan((1,1,4),(2,4,2),receive='dram_land')])
def test_gemm_output_contract_values(plan,output):
    plan=replace(plan,output=output); problem=Problem(16,16,16)
    program=compile_plan(problem,plan,1<<20,1<<30)
    rng=np.random.default_rng(15); a=rng.normal(size=(16,16)); b=rng.normal(size=(16,16))
    np.testing.assert_allclose(program.replay(a,b),a@b,atol=1e-12)
    regions=program.regions('C'); coverage=np.zeros((16,16),dtype=int)
    for r in regions:coverage[r.row:r.row+r.rows,r.col:r.col+r.cols]+=1
    copies=plan.ps[2]//prod(plan.c) if output=='replicated' else 1
    assert np.all(coverage==copies)
    raw,info=program.lower(64)
    assert info['macs']==16**3
    # The broadcast uses SRAM copies, and does not read final results back.
    if output=='replicated':
        base=compile_plan(problem,replace(plan,output='sharded'),1<<20,1<<30).lower(64)[1]
        assert info['logical_dram_read_bytes']==base['logical_dram_read_bytes']
        assert info['logical_dram_write_bytes']==base['logical_dram_write_bytes']+(copies-1)*16*16*2


@pytest.mark.parametrize('output',POLICIES)
def test_attention_output_contract_values_and_ports(output):
    p=Attention(2,9,23,4,1,4); layout=KVLayout(p,8)
    plan=AttentionPlan(4,2,2,2,1,2,output)
    rng=np.random.default_rng(4)
    q=rng.normal(size=(2,9,4,4)); k=rng.normal(size=(2,23,1,4)); v=rng.normal(size=k.shape)
    expected=Schedule(layout,replace(plan,output='sharded')).replay(q,k,v)
    schedule=Schedule(layout,plan)
    np.testing.assert_allclose(schedule.replay(q,k,v),expected,atol=1e-12)
    assert schedule.buffer_peak()<=plan.sram_bytes(p)
    out=endpoints(layout,plan)[1]
    assert sum(r.area for r in out)==2*9*16*(2 if output=='replicated' else 1)


def test_attention_pipeline_retains_kv_during_send_and_prefetch():
    p=Attention(1,16,64,4,1,8); plan=AttentionPlan(4,1,2,2)
    schedule=Schedule(KVLayout(p,4),plan)
    events=schedule.events
    assert any(a.kind=='update' and any(b.kind=='kv_send' and b.time==a.time and b.core==a.core
                                       and b.key==a.other for b in events) for a in events)
    assert any(a.kind=='kv_load' and a.key[3]>0 and any(b.kind=='update' and b.time==a.time
                                                     and b.other[3]==a.key[3]-1 for b in events) for a in events)
    assert schedule.buffer_peak()<=plan.sram_bytes(p)
    for e in events:
        if e.kind=='q_move':
            assert not any(u.kind=='update' and u.key==e.key and u.time<=e.time<u.time+4 for u in events)


def test_output_contract_shared_by_gemm_and_attention_and_gated_epilogue(tmp_path):
    from frontend.near_mem.model.kernels import Gemm
    from frontend.near_mem.tasks import Graph, Region
    chip=make_hardware(tmp_path/'hw',4,64)
    for mode in POLICIES:
        assert mode in [p.output for p in variants(Plan((1,2,2),(4,8,8)))]
        assert mode in [p.output for p in variants(AttentionPlan(1,4))]
    g=Graph(chip); problem=Problem(8,32,16)
    kernel=Gemm(problem,Plan((1,2,2),(4,8,8),output='replicated'),gated=True).prepare(g,'expand')
    g.tensor(0,'x',[8,16]); out=kernel.compile(g,[Region(0,'x',0,0,8,16)])
    raw=g.operations[-1]['raw']
    ops=sum(t['vec_count'] for phases in raw for ph in phases for t in ph['vector'] if t['name']=='output_epilogue')
    assert ops==5*8*16  # finalize once, then broadcast half-width gated values
    assert sum(r.area for r in out)==2*8*16
    g.simulate(tmp_path/'gated')


def test_transformer_search_includes_output_contracts(tmp_path):
    from frontend.near_mem.experiments.transformer import Experiment, Search, propose_gemm, _entry_score
    from frontend.near_mem.methods import target
    from frontend.near_mem.model.transformer import Workload
    chip=make_hardware(tmp_path/'hw',4,64); w=Workload(1,8,8,4,1,8,32); hw=dict(target(chip),single_b=True)
    search=Search(Experiment(w,chip,tmp_path/'search'),hw,'ours',1,True)
    pool=[]; proposals,_=propose_gemm(w,chip,'ours',hw,1,1,'output',pool=pool)
    split=Plan((1,2,2),(4,8,8))  # a residual split-K reduction has output contracts
    pool.append((split,*_entry_score(w,chip,hw,'output',dict(plan=asdict(split),native=False,direction=0,gated=False))))
    options=search.gemm('output',proposals,pool)
    assert {c['plan']['output'] for c in options}>={'sharded','replicated'}
    layouts=[tuple(map(tuple,search.ports['output',json.dumps(c,sort_keys=True)])) for c in options]
    assert len(layouts)==len(set(layouts))  # one option per boundary layout
    chosen=search.attention([AttentionPlan(1,4,2,2)])
    kept=chosen+[a for c in chosen for a in search.alternates.get(('attention',json.dumps(asdict(c),sort_keys=True)),[])]
    assert {c.output for c in kept}>={'sharded','first','replicated'}  # equal layouts share one option


def test_baseline_boundary_keeps_its_contract(tmp_path):
    from frontend.near_mem.experiments.transformer import Experiment, Search, propose_gemm
    from frontend.near_mem.methods import METHODS, target
    from frontend.near_mem.model.transformer import Workload
    assert {m for m, x in METHODS.items() if x.boundary} == {'ours', 'local', 'dram'}
    chip=make_hardware(tmp_path/'hw',4,64); w=Workload(1,8,8,4,1,8,32); hw=dict(target(chip),single_b=True)
    search=Search(Experiment(w,chip,tmp_path/'search'),hw,'t10',1,False)
    proposals,_=propose_gemm(w,chip,'t10',hw,1,1,'output')
    assert {json.dumps(c,sort_keys=True) for c in search.gemm('output',proposals,None)}<={json.dumps(c,sort_keys=True) for c in proposals}
    assert {c['plan']['output'] for c in proposals}=={'sharded'}
    assert {c.output for c in search.attention([AttentionPlan(1,4,2,2)])}=={'sharded'}
