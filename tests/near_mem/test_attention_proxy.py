"""A proposal proxy respects causal phases, prefetch and graph boundaries."""
from dataclasses import replace

from frontend.near_mem.model.attention import Attention,AttentionPlan,build_attention,plans,KVLayout
from frontend.near_mem.model.attention_proxy import AttentionProxy
from frontend.near_mem.output import variants
from frontend.near_mem.sim import make_hardware
from frontend.near_mem.tasks import Timeline


def test_attention_proxy_waits_for_serial_work_but_overlaps_prefetch(tmp_path):
    chip=make_hardware(tmp_path/'hw',1); proxy=AttentionProxy(chip)
    def score(ready):
        t=Timeline(1,64); t.access(0,1,'kv',[0,0],[1,16384],ready=ready)
        for i in range(1,5):t.work(0,i,macs=100*1024)
        t.work(0,5,macs=2000*1024)
        return proxy.tasks(t.finish())
    assert score(5)<score(None)
    assert score(5)>=2400


def test_attention_proxy_shortlist_keeps_output_and_spatial_options(tmp_path):
    chip=make_hardware(tmp_path/'hw',4); proxy=AttentionProxy(chip)
    p=Attention(1,16,32,4,1,8)
    space=[v for plan in plans(KVLayout(p,4),128*1024) for v in variants(plan)]
    selected=proxy.select(p,space,4,placed=True)
    family=lambda x:(x.query_parts,x.context_parts,x.output)
    assert set(map(family,selected))==set(map(family,space))
    assert len(selected)<len(space)
    assert set(proxy.select(p,space,0))==set(space)
    best=min(space,key=lambda x:proxy.score(p,x,True))
    assert best in selected


def test_attention_proxy_charges_distribution_only_at_standalone_boundary(tmp_path):
    chip=make_hardware(tmp_path/'hw',4); proxy=AttentionProxy(chip)
    p=Attention(1,16,32,4,1,8); plan=AttentionPlan(1,4,4)
    assert proxy.score(p,plan)>proxy.score(p,plan,True)
    # Replication at the output is real work in both analytical and measured paths.
    replicated=replace(plan,output='replicated')
    assert proxy.score(p,replicated,True)>proxy.score(p,plan,True)
    graph=build_attention(p,chip,plan,True)
    assert not any(op['metadata'].get('noc_data_bytes') for op in graph.operations[:-1])
