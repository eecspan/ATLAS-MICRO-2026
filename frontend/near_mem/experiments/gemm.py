"""Single GEMM: one method's search, measured by ATLAS.

Inputs live in DRAM once; any replica a plan needs is created by the shared
linear all-gather and charged (``gemm.distribution``). Every method lets the
shared proxy shortlist its space and simulates the shortlist in proxy order; ``deadline``
caps the wall-clock seconds of a run (no simulation starts after it, and one
still running is aborted and dropped) so all methods can be held to one limit; ``atlas``
instead runs the unmodified upstream tiling explorer (a check of ``tp``). Each run writes only its own
cycles; ``experiments.compare`` turns separate runs into speedups. The summary
also reports the Roofline ideal.
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
from ..gemm.plan import Plan, Problem, load
from ..gemm.space import Proxy, families, family
from ..methods import METHODS, target
from ..search import best, measure_all, shortlist
from ..resources import report
from ..sim import WORKERS, dump, fingerprint


def search(problem, chip, out, method, budget, workers, hw, proxy, provenance, deadline=None):
    """Proxy shortlist of ``method`` (exact branch and bound), then simulation."""
    bound = lambda q: proxy.lower(problem, q)+proxy.distribution(problem, q)
    pick = lambda space, m: shortlist(space, lambda p: proxy(problem, p), budget, family, bound,
                                      lambda q: families(q, m.loops))
    ranked, generated = pick(lambda skip=None: METHODS[method].space(problem, skip=skip, **hw), METHODS[method])
    space = [p for p, _ in ranked]  # ascending proxy score
    evaluate = partial(measure, problem, chip=chip, folder=out/'runs', provenance=provenance, deadline=deadline)
    rows = measure_all(space, evaluate, workers, processes=True, deadline=deadline)
    dump(out/'results.json', rows)
    dump(out/'search.json', dict(generated=generated, measured=len(rows), space=[asdict(p) for p in space]))
    selected = best(rows); chosen = load(selected['plan'])
    if isinstance(chosen, Plan):
        program = compile_plan(problem, chosen, hw['sram'], hw['dram'])
        rng = np.random.default_rng(42)
        a = rng.normal(size=(problem.m, problem.k)); b = rng.normal(size=(problem.k, problem.n))
        np.testing.assert_allclose(program.replay(a, b), a@b, rtol=1e-10, atol=1e-9)
    (out/'selected').mkdir()
    for name in ('operators.yaml', 'placement.yaml'):
        shutil.copy2(Path(selected['path'])/name, out/'selected'/name)
    return selected


def run(problem, chip, out, method='ours', budget=16, workers=WORKERS, tile_min=8, tile_policy='all', layer_policy='elk',
        deadline=None):
    chip = Path(chip).resolve(); out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True); start = time.monotonic(); stop = deadline and start+deadline
    hw = target(chip, tile_min, tile_policy, layer_policy); proxy = Proxy(chip); provenance = fingerprint(chip)
    if METHODS[method].native:
        selected = tune(problem, chip, out/'native', workers)
    else:
        selected = search(problem, chip, out, method, budget, workers, hw, proxy, provenance, stop)
    summary = dict(experiment='gemm', problem=asdict(problem), chip=str(chip), method=method, best=selected,
                   cycles=selected['result']['e2e_cycles'], roofline=proxy.roofline(problem),
                   seconds=time.monotonic()-start, deadline=deadline,
                   hit_deadline=bool(stop and time.monotonic() > stop), sha256=provenance)
    if 'resources' in selected['result']:  # the upstream ``atlas`` check carries no task counts
        summary.update(report(selected['result']['resources'], summary['cycles'], chip))
    dump(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'cycles', 'roofline', 'seconds')}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, value in [('m', 128), ('n', 6144), ('k', 4096), ('budget', 16), ('workers', WORKERS), ('tile-min', 8)]:
        ap.add_argument('--'+key, type=int, default=value)
    ap.add_argument('--chip', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=list(METHODS), default='ours')
    ap.add_argument('--tile-policy', choices=['powers', 'all'], default='all')
    ap.add_argument('--layer-policy', choices=['elk', 'all'], default='elk')
    ap.add_argument('--deadline', type=float, help='common wall-clock cap in seconds (no simulation starts later)')
    a = ap.parse_args()
    if min(a.m, a.n, a.k, a.budget, a.workers, a.tile_min) <= 0:
        ap.error('Counts must be positive')
    run(Problem(a.m, a.n, a.k), a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.tile_policy, a.layer_policy,
        a.deadline)


if __name__ == '__main__':
    main()
