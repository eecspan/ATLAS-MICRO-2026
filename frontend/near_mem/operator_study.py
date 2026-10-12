"""Retune a native QKV prefix for an operator-level study, with exact tuning acceleration.

The saved full compiler supplies placement and Q/KV communication. Only QKV
GEMM is retuned by the upstream exhaustive tile explorer. Private identical
DRAMs and a communication-free GEMM permit one-core tuning; its selected kernel
must match a complete-chip replay before this baseline is accepted.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import shutil

import yaml

from frontend.auto.tiling_explorer_cloud import explore_gemm_comp
from frontend.hardware_parser import load_chip_config_from_yaml
from frontend.model_parser import Operator, OperatorType
from .backend import fingerprint
from .cloud_compare import contract, dump, execute, freeze_graph
from .run import ROOT


def prepare(source, out, workers):
    source=source.resolve();out=out.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True)
    config=yaml.safe_load((source/'system.yaml').read_text())
    chip=Path(config['system']['chip']['config_path']).resolve()
    # Reusing communication on another hardware would invalidate this control.
    original=json.loads((source/'manifest.json').read_text())
    for path,digest in original['sha256'].items():
        if '/configs/architecture/' in path or path==str(chip):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:
                raise ValueError('Saved communication hardware changed: '+path)
    graph,placement,task_hashes=freeze_graph(source/'native',out/'tasks')
    if graph['operator'][0]['name']!='qkv_proj':raise ValueError('Expected QKV first')
    stop=next(i for i,op in enumerate(graph['operator']) if op['type']=='decode-attention')
    graph['operator']=graph['operator'][:stop]
    problem,names=contract(graph['operator'][0],placement)
    tensors={t['name']:t for t in placement['tensor']}
    cfg=yaml.safe_load(chip.read_text());single=deepcopy(cfg)
    single['architecture']['core_num']=1;single['architecture'].pop('noc',None)
    tuning=out/'tuning_chip.yaml';tuning.write_text(yaml.safe_dump(single))
    op=Operator(name='qkv_proj',op_type=OperatorType.GEMM,B=1,M=problem.m,N=problem.n,K=problem.k)
    best=explore_gemm_comp(tensors[names['A']],tensors[names['C']],[tensors[names['B']]],op,2,
          load_chip_config_from_yaml(cfg),str(tuning),8,512,8,workers,str(out/'native_tuning'))
    graph['operator'][0]=best['comp_description_list'][0]
    measured=execute(chip,out/'native_kernel',dict(operator=[graph['operator'][0]]),placement)
    if measured['e2e_cycles']!=best['latency']:
        raise AssertionError('One-core tuning did not match full-chip kernel replay')
    native=out/'native';native.mkdir()
    dump(native/'operator_description.yaml',graph);dump(native/'data_placement.yaml',placement)
    shutil.copy2(source/'system.yaml',out/'system.yaml')
    hashes=fingerprint(chip)
    manifest=dict(evaluation_scope='QKV projection plus native Q/KV communication prefix; not an LLM layer',
        source=str(source),batch=original['batch'],context=original['context'],
        native_tiling=best['tiling_factors'],single_core_tuning_cycles=best['latency'],
        full_chip_kernel_cycles=measured['e2e_cycles'],exact_tuning_replay=True,
        search='upstream exhaustive legal tiles, original minimum M=8 K=512 N=8',
        communication='copied unchanged from saved complete auto compiler graph',task_sha256=task_hashes,
        sha256={p:h for p,h in hashes.items() if Path(p).is_file()},backend=hashes)
    dump(out/'manifest.json',manifest)
    result=execute(chip,out/'native_prefix',graph,placement)
    dump(out/'result.json',dict(layer_cycles=result['e2e_cycles'],
          scope='field name retained for runner compatibility; this is a QKV prefix, not a layer',**result))
    print(json.dumps(dict(native_kernel=measured['e2e_cycles'],prefix=result['e2e_cycles'],
                         tiling=best['tiling_factors']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();prepare(a.source,a.output,a.workers)
