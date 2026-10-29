"""Compare nested attention spaces on the same backend, tiles and KV placement.

Measure the union once, then select each subset independently. This isolates
space enlargement from backend fixes; historical results are not comparable.
"""
import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import time

from .attention import measure
from ..model.attention import Attention, KVLayout, plans
from ..search import measure_all
from ..sim import dump, hardware, fingerprint


def run(problem, chip, output, workers=8):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    arch,_,_ = hardware(chip); layout = KVLayout(problem,arch['core_num'])
    sram = arch['core']['buffer']['buffer_size']*1024
    spaces = dict(context=set(plans(layout,sram,False)),kv=set(plans(layout,sram,joint=False)))
    space = list(plans(layout,sram)); spaces['full'] = set(space)
    spaces['q_only'] = {p for p in space if p.query_shift and not p.kv_shift}
    spaces['joint_only'] = {p for p in space if p.query_shift and p.kv_shift}
    started = time.monotonic()
    rows = measure_all(space,partial(measure,problem,Path(chip),output/'runs'),workers,processes=True)
    dump(output/'results.json',rows)
    best = {name:min((r for p,r in zip(space,rows) if p in subset),key=lambda r:r['result']['e2e_cycles'])
            for name,subset in spaces.items() if subset}
    cycles = {name:r['result']['e2e_cycles'] for name,r in best.items()}
    summary = dict(problem=asdict(problem),chip=str(chip),cycles=cycles,best=best,
                   candidates={name:len(s) for name,s in spaces.items()},seconds=time.monotonic()-started,
                   gain_over_kv=cycles['kv']/cycles['full'],gain_over_context=cycles['context']/cycles['full'],
                   sha256=fingerprint(chip))
    dump(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='best'},indent=2),flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--chip',required=True,type=Path); ap.add_argument('--output',required=True,type=Path)
    for key,value in [('batch',1),('queries',128),('context',128),('query-heads',8),('kv-heads',2),('head-dim',64),('workers',8)]:
        ap.add_argument('--'+key,type=int,default=value)
    a = ap.parse_args()
    run(Attention(a.batch,a.queries,a.context,a.query_heads,a.kv_heads,a.head_dim),a.chip,a.output,a.workers)


if __name__ == '__main__':
    main()
