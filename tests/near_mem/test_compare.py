import json

from frontend.near_mem.experiments.gemm import run
from frontend.near_mem.gemm.plan import load as as_plan
from frontend.near_mem.gemm.plan import Problem
from frontend.near_mem.sim import make_hardware
from frontend.near_mem.gemm.space import candidates


def test_local_space_is_an_exact_no_rotation_subset():
    args=(Problem(8,8,8),4,8192,4,'all','all')
    joint=set(candidates(*args,distinct=False));local=set(candidates(*args,rotation=False,distinct=False))
    assert local=={p for p in joint if not p.layers}
    assert any(p.ps[2]>1 for p in local)
    assert {p.loop for p in local}=={'mnk','mkn','nmk','nkm','kmn','knm'}


def test_methods_run_separately_and_compare_nests_spaces(tmp_path):
    from frontend.near_mem.experiments.compare import compare
    chip=make_hardware(tmp_path/'hardware',4,8)
    args=dict(budget=2,workers=2,tile_min=4,layer_policy='all')
    tp=run(Problem(8,8,16),chip,tmp_path/'tp','tp',**args)
    ours=run(Problem(8,8,16),chip,tmp_path/'ours','ours',**args)
    plans=[as_plan(r['plan']) for r in json.loads((tmp_path/'ours'/'results.json').read_text())]
    assert any(p.b!=(1,1) for p in plans) and any(p.c!=(1,1) for p in plans)
    assert (tmp_path/'ours'/'selected/operators.yaml').is_file()
    [row]=compare([tmp_path])
    assert row['cycles']=={'ours':ours['cycles'],'tp':tp['cycles']}
    assert row['speedup']['tp']==1 and 'roofline' in row['speedup']
    assert ours['cycles']<=tp['cycles']


def test_attention_comparison_nests_native_splits(tmp_path):
    from frontend.near_mem.experiments.attention import run as attention
    from frontend.near_mem.model.attention import Attention
    chip=make_hardware(tmp_path/'hardware',4,64)
    problem=Attention(2,32,32,4,2,16)
    atlas,local,ours=(attention(problem,chip,tmp_path/m,m,2) for m in ('atlas','local','ours'))
    assert atlas['plans']==local['plans']<ours['plans']
    assert atlas['cycles']==local['cycles']>=ours['cycles']


def test_single_weight_copy_requires_b_rotation_for_m_splits():
    args=(Problem(8,8,8),4,8192,4,'all','all')
    plans=list(candidates(*args,single_b=True))
    assert plans and all(p.ps[0]==p.b[0]*p.b[1] for p in plans)
    assert any(p.ps[0]>1 for p in plans)
