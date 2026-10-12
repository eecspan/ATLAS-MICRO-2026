"""Original ATLAS GEMM pipeline and exhaustive tiling as a standalone baseline.

The simulator's GEMM template has no global partition selector. We expose all
legal M/N partitions, retain its original tile minima and SRAM rule, and tune
with the upstream explorer. No LLM or KV context is needed.
"""
from copy import deepcopy
from dataclasses import asdict
import json
from math import prod
from pathlib import Path

import yaml

from .backend import hardware, fingerprint
from .run import execute, dump
from .plan import Plan
from .search import divisors




def matrix_placement(shape, dtype, capacity):
    m,n,k=shape;address=0;tensors=[]
    for name,extent,strides in [('A',[m,k],[k,1]),('B',[k,n],[1,k]),('C',[m,n],[n,1])]:
        address=(address+127)//128*128
        tensors.append(dict(name=name,shape=extent,strides=strides,element_size=dtype,base_addr=address))
        address+=prod(extent)*dtype
    if address>capacity:raise ValueError('Native placement exceeds per-core DRAM capacity')
    return dict(tensor=tensors)


def partitions(problem,cores):
    for pm in divisors(cores):
        pn=cores//pm
        if problem.m%pm==0 and problem.n%pn==0:yield pm,pn,1


def tune(problem,chip,out,workers=4):
    from frontend.auto.tiling_explorer_cloud import explore_gemm_comp
    from frontend.hardware_parser import load_chip_config_from_yaml
    from frontend.model_parser import Operator, OperatorType
    chip=Path(chip).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    arch,capacity,_=hardware(chip);cfg=yaml.safe_load(chip.read_text())
    single=deepcopy(cfg);single['architecture']['core_num']=1;single['architecture'].pop('noc',None)
    tuning=out/'tuning_chip.yaml';tuning.write_text(yaml.safe_dump(single))
    rows=[]
    for ps in partitions(problem,arch['core_num']):
        shape=tuple(n//s for n,s in zip(problem.shape,ps))
        try:placement=matrix_placement(shape,problem.dtype_bytes,capacity)
        except ValueError:continue
        folder=out/('ps_'+'_'.join(map(str,ps)));folder.mkdir()
        m,n,k=shape;a,b,c=placement['tensor']
        op=Operator(name='gemm',op_type=OperatorType.GEMM,B=1,M=m,N=n,K=k)
        best=explore_gemm_comp(a,c,[b],op,problem.dtype_bytes,load_chip_config_from_yaml(cfg),
                              str(tuning),8,512,8,workers,str(folder/'tuning'))
        if best is None:continue
        measured=execute(chip,folder,dict(operator=best['comp_description_list']),placement)
        if measured['e2e_cycles']!=best['latency']:
            raise AssertionError('Single-core native tuning differs from full-chip replay')
        tm,tk,tn=best['tiling_factors'];plan=Plan(ps,(tm,tn,tk))
        assert measured['flop_count']==2*prod(problem.shape)+prod(problem.shape)//tk
        row=dict(plan=asdict(plan),result=measured,path=str(folder),exact_tuning_replay=True)
        rows.append(row);dump(out/'results.json',rows)
    if not rows:raise ValueError('No legal native M/N partition under the configured capacities')
    best=min(rows,key=lambda r:r['result']['e2e_cycles'])
    dump(out/'manifest.json',dict(problem=asdict(problem),sha256=fingerprint(chip),
         baseline='Original ATLAS MNK generator and exhaustive upstream tile explorer',
         spatial='All legal M/N partitions, no split-K; an explicit standalone harness policy',
         tile_minima_MKN=[8,512,8],packing=False,one_core_tuning_full_chip_verified=True))
    dump(out/'summary.json',dict(best=best,partitions=len(rows)))
    return best
