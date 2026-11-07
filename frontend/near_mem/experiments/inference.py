"""Fixed-batch inference latency from representative steady-state layers.

A quick experiment protocol, shared by every method. Prefill splits each
request into ``chunk``-token chunks (the last one is padded); the GEMMs see
M=chunk regardless of batch, and attention is taken at the mean chunk prefix
length. Decode uses the given batch with attention at the middle generated
token. Each phase searches and simulates one layer at its representative
point (``experiments.transformer``); the totals scale it by the step count and
the layer count.
"""
import argparse
from dataclasses import asdict
import json
from math import ceil
from pathlib import Path

from ..methods import METHODS
from ..model.transformer import Workload
from ..sim import WORKERS, dump
from . import transformer


def points(batch, prompt, generate, chunk):
    """``{phase: (representative layer, steps)}`` of one fixed-shape request batch."""
    n = ceil(prompt/chunk)
    return dict(prefill=(dict(batch=1, queries=chunk, context=chunk*(n+1)//2), batch*n),
                decode=(dict(batch=batch, queries=1, context=prompt+generate//2), generate))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--chip', required=True, type=Path); ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--method', choices=list(METHODS), default='ours')
    ap.add_argument('--phase', choices=('prefill', 'decode', 'both'), default='both')
    for name, value in [('batch', 1), ('input', 2048), ('generate', 256), ('chunk', 512), ('layers', 32),
                        ('query-heads', 32), ('kv-heads', 8), ('head-dim', 128), ('ffn', 14336),
                        ('budget', 4), ('workers', WORKERS), ('tile-min', 8), ('passes', 2), ('width', 64),
                        ('keep', 8), ('tiles', 2), ('refine', 3), ('attention-budget', 32)]:
        ap.add_argument('--'+name, type=int, default=value)
    ap.add_argument('--deadline', type=float, help='common wall-clock cap per phase in seconds')
    a = ap.parse_args()
    if min(a.batch, a.input, a.generate, a.chunk, a.layers) < 1:
        ap.error('Counts must be positive')
    phases = ('prefill', 'decode') if a.phase == 'both' else (a.phase,)
    shape = dict(query_heads=a.query_heads, kv_heads=a.kv_heads, head_dim=a.head_dim, ffn=a.ffn)
    result = {}
    for phase in phases:
        point, steps = points(a.batch, a.input, a.generate, a.chunk)[phase]
        workload = Workload(**point, **shape)
        s = transformer.run(workload, a.chip, a.output/phase, a.method, a.budget, a.workers, a.tile_min, a.passes,
                            a.width, a.keep, a.tiles, a.refine, a.deadline, a.attention_budget)
        result[phase] = dict(workload=asdict(workload), layer_cycles=s['cycles'], steps=steps,
                             cycles=s['cycles']*steps*a.layers, hit_deadline=s['hit_deadline'])
    summary = dict(experiment='inference', method=a.method, chip=str(a.chip.resolve()),
                   workload=dict(batch=a.batch, input=a.input, generate=a.generate, chunk=a.chunk,
                                 layers=a.layers, phases=list(phases), **shape),
                   phases=result, cycles=sum(r['cycles'] for r in result.values()))
    if 'decode' in result:
        summary['tpot_cycles'] = result['decode']['cycles']//a.generate
    dump(a.output/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'cycles')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
