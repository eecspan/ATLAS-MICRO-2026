"""Single GEMM: the original ATLAS compiler against our extended space.

Inputs live in DRAM once; any replica a plan needs is created by the shared
linear all-gather and charged (``gemm.distribution``). The native ATLAS
incumbent is always measured and always eligible, so the extended spaces can
only match or beat it.
"""
import argparse
from dataclasses import asdict, replace
from functools import partial
from itertools import permutations
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ..gemm.compiler import compile_plan
from ..gemm.lower import measure
from ..gemm.native import tune
from ..gemm.plan import Plan, Problem
from ..gemm.space import Proxy, families, family, resident_bytes
from ..methods import propose, target, union
from ..search import best, measure_all, shortlist
from ..sim import dump, fingerprint


def as_plan(data):
    return Plan(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def matched_controls(plan, problem, capacity):
    """The winning rotation's spatial/tile choice without rotation, all six loops."""
    controls = [replace(plan, a=(1, 1), b=(1, 1), c=(1, 1), group=(1, 1, 1), layer_order=(), loop=''.join(o))
                for o in permutations('mnk')]
    return [p for p in controls if resident_bytes(p, problem.dtype_bytes) <= capacity]


def run(problem, chip, out, budget=16, workers=4, tile_min=16, tile_policy='powers', layer_policy='elk'):
    chip = Path(chip).resolve(); out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True); start = time.monotonic()
    hw = target(chip, tile_min, tile_policy, layer_policy); sram = hw['sram']
    provenance = fingerprint(chip)
    native = tune(problem, chip, out/'native', workers)
    native_plan = as_plan(native['plan'])
    proxy = Proxy(chip)
    # Exact branch and bound: a prefix whose proxy lower bound cannot enter
    # the shortlist is never expanded.
    bound = lambda q: proxy.lower(problem, q)+proxy.distribution(problem, q)
    pick = lambda space: shortlist(space, lambda p: proxy(problem, p), budget, family, bound, families)
    ranked, generated = propose(problem, hw, pick)
    spaces = {name: union(ranked, name, seed=[native_plan]) for name in ranked}
    evaluate = partial(measure, problem, chip=chip, cache=out/'cache', provenance=provenance)
    rows = measure_all(spaces['ours'], evaluate, workers, processes=True)
    rotating = [r for r in rows if as_plan(r['plan']).layers]
    if rotating:
        controls = matched_controls(as_plan(best(rotating)['plan']), problem, sram)
        rows += measure_all([p for p in controls if p not in spaces['ours']], evaluate, workers, processes=True)
        for name in spaces:
            spaces[name] = list(dict.fromkeys(spaces[name]+controls))
    dump(out/'results.json', rows)
    dump(out/'search.json', dict(generated=generated, measured=len(rows),
                                 spaces={k: [asdict(p) for p in v] for k, v in spaces.items()}))
    winners = {name: best([r for r in rows if as_plan(r['plan']) in set(space)]) for name, space in spaces.items()}
    selected = winners['ours']
    native_measured = next(r for r in rows if as_plan(r['plan']) == native_plan)
    if native_measured['result']['kernel_cycles'] != native['result']['kernel_cycles']:
        raise AssertionError('Native plan differs when lowered by the shared compiler')
    program = compile_plan(problem, as_plan(selected['plan']), sram, hw['dram'])
    rng = np.random.default_rng(42)
    a = rng.normal(size=(problem.m, problem.k)); b = rng.normal(size=(problem.k, problem.n))
    np.testing.assert_allclose(program.replay(a, b), a@b, rtol=1e-10, atol=1e-9)
    (out/'selected').mkdir()
    for name in ('operators.yaml', 'placement.yaml'):
        shutil.copy2(Path(selected['path'])/name, out/'selected'/name)
    cycles = lambda r: r['result']['e2e_cycles']
    summary = dict(problem=asdict(problem), chip=str(chip), atlas=native, **winners,
                   speedup={name: cycles(native)/cycles(r) for name, r in winners.items()},
                   rotation_gain=cycles(winners['local'])/cycles(winners['ours']),
                   seconds=time.monotonic()-start, sha256=provenance)
    dump(out/'summary.json', summary)
    print(json.dumps(dict(speedup=summary['speedup'], rotation_gain=summary['rotation_gain']), indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, value in [('m', 128), ('n', 6144), ('k', 4096), ('budget', 16), ('workers', 4), ('tile-min', 16)]:
        ap.add_argument('--'+key, type=int, default=value)
    ap.add_argument('--chip', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tile-policy', choices=['powers', 'all'], default='powers')
    ap.add_argument('--layer-policy', choices=['elk', 'all'], default='elk')
    a = ap.parse_args()
    if min(a.m, a.n, a.k, a.budget, a.workers, a.tile_min) <= 0:
        ap.error('Counts must be positive')
    run(Problem(a.m, a.n, a.k), a.chip, a.output, a.budget, a.workers, a.tile_min, a.tile_policy, a.layer_policy)


if __name__ == '__main__':
    main()
