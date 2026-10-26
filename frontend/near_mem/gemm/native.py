"""Original ATLAS GEMM pipeline and exhaustive tiling as a standalone baseline.

The simulator's GEMM template has no global partition selector. We expose all
legal M/N partitions, retain its original tile minima and SRAM rule, and tune
with the upstream explorer. No LLM or KV context is needed.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import prod
from pathlib import Path

import yaml

from ..sim import ROOT, dump, execute, fingerprint, hardware, span, strides
from .plan import Plan
from .space import divisors
from .distribution import charge, distribute, received_bytes


def matrix_placement(shape, dtype, capacity):
    m,n,k=shape;address=0;tensors=[]
    for name,extent in [('A',[m,k]),('B',[k,n]),('C',[m,n])]:
        address=(address+127)//128*128;pitch=strides(extent,dtype,name=='B')
        tensors.append(dict(name=name,shape=extent,strides=pitch,element_size=dtype,base_addr=address))
        address+=span(extent,pitch,dtype)
    if address>capacity:raise ValueError('Native placement exceeds per-core DRAM capacity')
    return dict(tensor=tensors)


def partitions(problem,cores):
    for pm in divisors(cores):
        pn=cores//pm
        if problem.m%pm==0 and problem.n%pn==0:yield pm,pn,1


def tuning_cache(chip):
    """Native tile optima depend only on hardware and the simulator, not on our code."""
    _,_,paths=hardware(chip);fp=fingerprint(chip)
    keys=[fp[str(p)] for p in paths]+[v for k,v in sorted(fp.items()) if k in ('atlas','dram/ramulator2','noc/booksim2') or k.startswith('binary:')]
    keys.append('pitch: 4 KiB leading dimensions padded by one burst')  # sim.strides
    key=hashlib.sha256(json.dumps(keys).encode()).hexdigest()[:16]
    return ROOT/'results/near_mem/native_tiles'/key


def tune(problem,chip,out,workers=4):
    """Exhaustive native tiles per M/N partition; total cycles include input replication.

    A partition's replication time is at least its largest inbound volume
    divided by the flit size, so partitions already beaten by that bound are
    not distributed in simulation.
    """
    from frontend.auto.tiling_explorer_cloud import explore_gemm_comp
    from frontend.hardware_parser import load_chip_config_from_yaml
    from frontend.model_parser import Operator, OperatorType
    chip=Path(chip).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    arch,capacity,_=hardware(chip);cfg=yaml.safe_load(chip.read_text())
    single=deepcopy(cfg);single['architecture']['core_num']=1;single['architecture'].pop('noc',None)
    tuning=out/'tuning_chip.yaml';tuning.write_text(yaml.safe_dump(single))
    kernels=[]
    for ps in partitions(problem,arch['core_num']):
        shape=tuple(n//s for n,s in zip(problem.shape,ps))
        try:placement=matrix_placement(shape,problem.dtype_bytes,capacity)
        except ValueError:continue
        folder=out/('ps_'+'_'.join(map(str,ps)));folder.mkdir(exist_ok=True)
        m,n,k=shape;a,b,c=placement['tensor']
        op=Operator(name=f'gemm_{m}_{n}_{k}',op_type=OperatorType.GEMM,B=1,M=m,N=n,K=k)
        best=explore_gemm_comp(a,c,[b],op,problem.dtype_bytes,load_chip_config_from_yaml(cfg),
                              str(tuning),8,512,8,workers,str(folder/'tuning'),str(tuning_cache(chip)))
        if best is None:continue
        measured=execute(chip,folder,dict(operator=best['comp_description_list']),placement)
        if measured['e2e_cycles']!=best['latency']:
            raise AssertionError('Single-core native tuning differs from full-chip replay')
        tm,tk,tn=best['tiling_factors'];plan=Plan(ps,(tm,tn,tk))
        assert measured['flop_count']==2*prod(problem.shape)+prod(problem.shape)//tk
        bound=measured['e2e_cycles']+received_bytes(problem,plan)/arch['noc']['flit_size']
        kernels.append((bound,plan,measured,folder))
    rows=[];incumbent=float('inf')
    for bound,plan,measured,folder in sorted(kernels,key=lambda x:x[0]):
        if bound>=incumbent:continue
        measured=charge(measured,distribute(problem,plan,chip,folder/'distribution'))
        incumbent=min(incumbent,measured['e2e_cycles'])
        rows.append(dict(plan=asdict(plan),result=measured,path=str(folder),exact_tuning_replay=True))
        dump(out/'results.json',rows)
    if not rows:raise ValueError('No legal native M/N partition under the configured capacities')
    best=min(rows,key=lambda r:r['result']['e2e_cycles'])
    dump(out/'manifest.json',dict(problem=asdict(problem),sha256=fingerprint(chip),
         baseline='Original ATLAS MNK generator and exhaustive upstream tile explorer',
         spatial='All legal M/N partitions, no split-K; an explicit standalone harness policy',
         tile_minima_MKN=[8,512,8],inputs='single DRAM copy; replicas charged by linear mesh all-gather; bound-pruned partitions',one_core_tuning_full_chip_verified=True))
    dump(out/'summary.json',dict(best=best,partitions=len(rows)))
    return best
