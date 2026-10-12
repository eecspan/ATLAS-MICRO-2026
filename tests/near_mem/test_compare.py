import json

from frontend.near_mem.compare import run, as_plan
from frontend.near_mem.plan import Problem
from frontend.near_mem.run import make_hardware
from frontend.near_mem.search import candidates


def test_local_space_is_an_exact_no_rotation_subset():
    args=(Problem(8,8,8),4,8192,4,'all','all')
    joint=set(candidates(*args));local=set(candidates(*args,rotation=False))
    assert local=={p for p in joint if not p.layers}
    assert any(p.ps[2]>1 for p in local)
    assert {p.loop for p in local}=={'mnk','mkn','nmk','nkm','kmn','knm'}


def test_complete_standalone_comparison_uses_native_and_joint_space(tmp_path):
    chip=make_hardware(tmp_path/'hardware',4,8)
    out=tmp_path/'compare'
    summary=run(Problem(8,8,16),chip,out,budget=2,workers=2,tile_min=4,layer_policy='all')
    assert summary['native_incumbent_exact']
    assert summary['selected_speedup']>=1
    assert summary['joint_over_local']>=1
    plans=[as_plan(r['plan']) for r in json.loads((out/'results.json').read_text())]
    assert any(p.b!=(1,1) for p in plans)
    assert any(p.c!=(1,1) for p in plans)
    assert (out/'selected/operators.yaml').is_file()
    assert json.loads((out/'numerical.json').read_text())['max_abs_error']<1e-9

    # Finalization may add matched controls to an already measured run. It must
    # refresh the selected artifact without requiring baseline retuning.
    import time
    from frontend.near_mem.backend import fingerprint
    from frontend.near_mem.compare import finish
    spaces=json.loads((out/'search.json').read_text())['selected']
    spaces={k:[as_plan(p) for p in values] for k,values in spaces.items()}
    again=finish(Problem(8,8,16),chip,out,summary['native'],spaces,
                 json.loads((out/'results.json').read_text()),time.monotonic(),fingerprint(chip),2)
    assert again['selected']['result']==summary['selected']['result']
