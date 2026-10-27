"""Single GEMM: the original ATLAS compiler against one method's space.

Inputs live in DRAM once; any replica a plan needs is created by the shared
linear all-gather and charged (``gemm.distribution``). The native ATLAS
incumbent is always measured and always eligible, so a method can only match
or beat it. Each run searches one method; compare methods by separate runs.
"""
import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import shutil
import time

import numpy as np

from ..gemm.compiler import compile_plan
from ..gemm.lower import measure
from ..gemm.native import tune
from ..gemm.plan import Plan, Problem
from ..gemm.space import Proxy, families, family
from ..methods import METHODS, propose, target, union
from ..search import best, measure_all, shortlist
from ..sim import WORKERS, dump, fingerprint


def as_plan(data):
    return Plan(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def run(problem, chip, out, method='ours', budget=16, workers=WORKERS, tile_min=16, tile_policy='powers', layer_policy='elk'):
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
    ranked, generated = propose(problem, hw, pick, method)
    space = union(ranked, method, seed=[native_plan])
    evaluate = partial(measure, problem, chip=chip, folder=out/'runs', provenance=provenance)
    rows = measure_all(space, evaluate, workers, processes=True)
    native_measured = next(r for r in rows if as_plan(r['plan']) == native_plan)
    if native_measured['result']['kernel_cycles'] != native['result']['kernel_cycles']:
        raise AssertionError('Native plan differs when lowered by the shared compiler')
    selected = best(rows)
    dump(out/'results.json', rows)
    dump(out/'search.json', dict(generated=generated, measured=len(rows), space=[asdict(p) for p in space]))
    program = compile_plan(problem, as_plan(selected['plan']), sram, hw['dram'])
    rng = np.random.default_rng(42)
    a = rng.normal(size=(problem.m, problem.k)); b = rng.normal(size=(problem.k, problem.n))
    np.testing.assert_allclose(program.replay(a, b), a@b, rtol=1e-10, atol=1e-9)
    (out/'selected').mkdir()
    for name in ('operators.yaml', 'placement.yaml'):
        shutil.copy2(Path(selected['path'])/name, out/'selected'/name)
    cycles = lambda r: r['result']['e2e_cycles']
    summary = dict(problem=asdict(problem), chip=str(chip), method=method, atlas=native, best=selected,
                   speedup=cycles(native)/cycles(selected),
                   seconds=time.monotonic()-start, sha256=provenance)
    dump(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'speedup', 'seconds')}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, value in [('m', 128), ('n', 6144), ('k', 4096), ('budget', 16), ('workers', WORKERS), ('tile-min', 16)]:
        ap.add_argument('--'+key, type=int, default=value)
    ap.add_argument('--chip', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=list(METHODS), default='ours')
    ap.add_argument('--tile-policy', choices=['powers', 'all'], default='powers')
    ap.add_argument('--layer-policy', choices=['elk', 'all'], default='elk')
    a = ap.parse_args()
    if min(a.m, a.n, a.k, a.budget, a.workers, a.tile_min) <= 0:
        ap.error('Counts must be positive')
    run(Problem(a.m, a.n, a.k), a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.tile_policy, a.layer_policy)


if __name__ == '__main__':
    main()
