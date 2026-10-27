import json

from frontend.near_mem.experiments.gemm import run, as_plan
from frontend.near_mem.gemm.plan import Problem
from frontend.near_mem.sim import make_hardware
from frontend.near_mem.gemm.space import candidates


def test_local_space_is_an_exact_no_rotation_subset():
    args=(Problem(8,8,8),4,8192,4,'all','all')
    joint=set(candidates(*args));local=set(candidates(*args,rotation=False))
    assert local=={p for p in joint if not p.layers}
    assert any(p.ps[2]>1 for p in local)
    assert {p.loop for p in local}=={'mnk','mkn','nmk','nkm','kmn','knm'}


def test_complete_standalone_comparison_contains_native_and_nests_spaces(tmp_path):
    chip=make_hardware(tmp_path/'hardware',4,8)
    out=tmp_path/'compare'
    summary=run(Problem(8,8,16),chip,out,budget=2,workers=2,tile_min=4,layer_policy='all')
    assert summary['method']=='ours' and summary['speedup']>=1
    plans=[as_plan(r['plan']) for r in json.loads((out/'results.json').read_text())]
    assert as_plan(summary['atlas']['plan']) in plans
    assert any(p.b!=(1,1) for p in plans)
    assert any(p.c!=(1,1) for p in plans)
    assert (out/'selected/operators.yaml').is_file()


def test_attention_comparison_nests_native_splits(tmp_path):
    from frontend.near_mem.experiments.attention import run as attention
    from frontend.near_mem.model.attention import Attention
    chip=make_hardware(tmp_path/'hardware',4,64)
    local=attention(Attention(2,32,32,4,2,16),chip,tmp_path/'local','local',2)
    ours=attention(Attention(2,32,32,4,2,16),chip,tmp_path/'ours','ours',2)
    assert local['plans']['atlas']==local['plans']['method']<ours['plans']['method']
    assert local['speedup']==1 and ours['speedup']>=1


def test_single_weight_copy_requires_b_rotation_for_m_splits():
    args=(Problem(8,8,8),4,8192,4,'all','all')
    plans=list(candidates(*args,single_b=True))
    assert plans and all(p.ps[0]==p.b[0]*p.b[1] for p in plans)
    assert any(p.ps[0]>1 for p in plans)
