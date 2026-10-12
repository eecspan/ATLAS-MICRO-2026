"""Standalone GEMM: original native tuning, no-rotation control, joint space.

All variants use one chip, matrix DRAM layout, native component accounting and
prefetch. Shards are preplaced; final C includes any necessary split-K reduction.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from itertools import permutations
from pathlib import Path
import shutil
import time

from .backend import fingerprint, hardware
from .compiler import compile_plan
from .native import tune
from .plan import Plan, Problem
from .search import candidates, Proxy, shortlist, evaluate, resident_bytes


def as_plan(data):
    return Plan(**{k:tuple(v) if isinstance(v,list) else v for k,v in data.items()})


def run(problem,chip,out,budget=16,workers=4,tile_min=16,tile_policy='all',layer_policy='elk'):
    chip=Path(chip).resolve();out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty output directory')
    out.mkdir(parents=True);start=time.monotonic()
    arch,dram,_=hardware(chip);capacity=arch['core']['buffer']['buffer_size']*1024
    provenance=fingerprint(chip)
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2))
    save('manifest.json',dict(problem=asdict(problem),chip=str(chip),sha256=provenance,
         boundary='Single GEMM with preplaced per-plan input shards; complete C, including residual reduction',
         common='Same chip, matrix layout, native work accounting, operand/output prefetch, capacity checks',
         excluded='Input distribution, persistent weight setup and inter-operator layout conversion',
         budget=budget,tile_min=tile_min,tile_policy=tile_policy,layer_policy=layer_policy))
    native=tune(problem,chip,out/'native',workers)
    print('native',native['result']['e2e_cycles'],flush=True)
    score=Proxy(chip);selection={};counts={}
    for name,rotation in [('local',False),('joint',True)]:
        domain=candidates(problem,arch['core_num'],capacity,tile_min,tile_policy,layer_policy,dram,rotation)
        selected,count=shortlist(problem,domain,score,budget)
        selection[name]=[p for p,_ in selected];counts[name]=count
    # The extension explicitly contains both original and stronger no-rotation
    # candidates; a heuristic shortlist may not discard those incumbents.
    native_plan=as_plan(native['plan'])
    selection['local']=list(dict.fromkeys([native_plan]+selection['local']))
    selection['joint']=list(dict.fromkeys(selection['local']+selection['joint']))
    plans=selection['joint']
    save('search.json',dict(generated=counts,simulations=len(plans),
         selected={name:[asdict(p) for p in values] for name,values in selection.items()},
         pruning='Divisibility/capacity exact; proxy shortlist and layer-order sampling heuristic'))
    rows=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs=[(problem,p,chip,out/'cache',provenance) for p in plans]
        for row in pool.map(evaluate,jobs):
            row['path']=str(out/'cache'/row['key']);rows.append(row);save('results.json',rows)
            print('candidate',len(rows),len(plans),row['result']['e2e_cycles'],flush=True)
    matched=next(r for r in rows if as_plan(r['plan'])==native_plan)
    for field,value in native['result'].items():
        if field!='operators' and matched['result'][field]!=value:
            raise AssertionError('Native incumbent differs through shared general compiler: '+field)
    return finish(problem,chip,out,native,selection,rows,start,provenance,workers)


def finish(problem,chip,out,native,selection,rows,start,provenance,workers):
    """Add a matched-tile control before final selection and value validation."""
    arch,dram,_=hardware(chip);capacity=arch['core']['buffer']['buffer_size']*1024
    def save(name,data):(out/name).write_text(json.dumps(data,indent=2))
    rotating=[r for r in rows if as_plan(r['plan']).layers]
    if rotating:
        winner=min(rotating,key=lambda r:r['result']['e2e_cycles'])
        p=as_plan(winner['plan']);measured={as_plan(r['plan']) for r in rows}
        controls=[replace(p,a=(1,1),b=(1,1),c=(1,1),group=(1,1,1),
                          layer_order=(),loop=''.join(loop)) for loop in permutations('mnk')]
        controls=[p for p in controls if resident_bytes(p,problem.dtype_bytes)<=capacity]
        missing=[p for p in controls if p not in measured]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs=[(problem,p,chip,out/'cache',provenance) for p in missing]
            for row in pool.map(evaluate,jobs):
                row['path']=str(out/'cache'/row['key']);rows.append(row)
        for name in selection:selection[name]=list(dict.fromkeys(selection[name]+controls))
        save('results.json',rows)
        save('matched_control.json',dict(rotation_plan=winner['plan'],
             controls=[asdict(p) for p in controls],extra_simulations=len(missing),
             scope='Winning rotation spatial/tile choices, six local orders, no temporal partition'))
    best={}
    for name,space in selection.items():
        best[name]=min((r for r in rows if as_plan(r['plan']) in space),key=lambda r:r['result']['e2e_cycles'])
    rotations=[r for r in rows if as_plan(r['plan']).layers]
    if rotations:best['rotation']=min(rotations,key=lambda r:r['result']['e2e_cycles'])
    winner_name,winner=min([('native',native),*best.items()],key=lambda item:item[1]['result']['e2e_cycles'])
    # Check the actually selected complete GEMM, including C rotation/reduction.
    import numpy as np
    rng=np.random.default_rng(42);a=rng.normal(size=(problem.m,problem.k));b=rng.normal(size=(problem.k,problem.n))
    program=compile_plan(problem,as_plan(winner['plan']),capacity,dram)
    expected=a@b;actual=program.replay(a,b)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-9)
    save('numerical.json',dict(max_abs_error=float(np.max(np.abs(actual-expected))),
         scope='Full-shape algebraic replay; not FP16 bit-exact',winner=winner_name))
    dest=out/'selected';dest.mkdir(exist_ok=True)
    for name in ('operators.yaml','placement.yaml'):shutil.copy2(Path(winner['path'])/name,dest/name)
    summary=dict(native=native,**best,winner=winner_name,selected=winner,
        raw_rotation_speedup=native['result']['e2e_cycles']/best['rotation']['result']['e2e_cycles'] if rotations else None,
        selected_speedup=native['result']['e2e_cycles']/winner['result']['e2e_cycles'],
        joint_over_local=best['local']['result']['e2e_cycles']/best['joint']['result']['e2e_cycles'],
        native_incumbent_exact=True,wall_seconds=time.monotonic()-start)
    save('summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,dict)},indent=2),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key,value in [('m',128),('n',6144),('k',4096),('budget',16),('workers',4),('tile-min',16)]:
        ap.add_argument('--'+key,type=int,default=value)
    ap.add_argument('--chip',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--tile-policy',choices=['powers','all'],default='all')
    ap.add_argument('--layer-policy',choices=['elk','all'],default='elk')
    a=ap.parse_args()
    if min(a.m,a.n,a.k,a.budget,a.workers,a.tile_min)<=0:ap.error('Counts must be positive')
    run(Problem(a.m,a.n,a.k),a.chip,a.output,a.budget,a.workers,a.tile_min,a.tile_policy,a.layer_policy)


if __name__=='__main__':main()
