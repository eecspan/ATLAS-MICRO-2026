"""Original ATLAS GEMM pipeline and its tiling as a standalone baseline.

The simulator's GEMM template has no global partition selector. We expose all
legal M/N partitions and keep its original tile minima and SRAM rule. Tiles
follow the same two-stage search as every method: the shared proxy ranks the
native tile domain and the upstream explorer simulates the best ``BUDGET``.
No LLM or KV context is needed.
"""
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import multiprocessing
from math import prod
from pathlib import Path
from types import SimpleNamespace

import yaml

from ..search import shortlist
from ..sim import dump, execute, fingerprint, hardware, span, strides
from .plan import Plan, Problem
from .space import Proxy, divisors
from .distribution import charge, distribute, received_bytes

# Proxy top-k simulated per local shape. Over 11 local shapes on both chips,
# the best of the top 16 is within 0.54% of the exhaustive optimum.
BUDGET = 16
MINIMA = (8, 512, 8)  # upstream tM, tK, tN


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


def tiles(shape, sram, dtype=2):
    """Upstream native tile domain ``(tm, tk, tn)``: divisors above the minima
    (halved until some exist) whose double-buffered tiles fit half the SRAM."""
    def factors(n, low):
        while low > 1 and not any(d >= low for d in divisors(n)):
            low //= 2
        return [d for d in divisors(n) if d >= low]
    m, n, k = shape
    return [(tm, tk, tn) for tm, tk, tn in product(factors(m, MINIMA[0]), factors(k, MINIMA[1]), factors(n, MINIMA[2]))
            if dtype*(tm*tk+tk*tn+2*tm*tn) <= sram/2]


def _simulate_tiles(args):
    from frontend.auto.tiling_explorer_cloud import explore_gemm_comp_worker
    return explore_gemm_comp_worker(*args)


def best_tile(shape, chip, out, workers=4, budget=BUDGET):
    """Best native tile of one core's local GEMM: proxy top-``budget``, then
    the upstream single-core explorer. Every core runs the same kernel without
    communication, so one core is exact (``tune`` checks this on the chip)."""
    chip = Path(chip).resolve(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    arch, capacity, _ = hardware(chip)
    single = deepcopy(yaml.safe_load(chip.read_text()))
    single['architecture']['core_num'] = 1; single['architecture'].pop('noc', None)
    (out/'chip.yaml').write_text(yaml.safe_dump(single))
    problem = Problem(*shape); proxy = Proxy(chip)
    domain = tiles(shape, arch['core']['buffer']['buffer_size']*1024)
    if not domain:
        raise ValueError('Native tile domain has no SRAM-feasible point')
    plans = [Plan((1, 1, 1), (tm, tn, tk)) for tm, tk, tn in domain]
    ranked, _ = shortlist(lambda skip: iter(plans), lambda p: proxy.kernel(problem, p), budget)
    chosen = [(p.tile[0], p.tile[2], p.tile[1]) for p, _ in ranked]
    a, b, c = matrix_placement(shape, problem.dtype_bytes, capacity)['tensor']
    op = SimpleNamespace(name='gemm_{}_{}_{}'.format(*shape), M=shape[0], N=shape[1], K=shape[2], B=1)
    jobs = [(a, c, [b], op, problem.dtype_bytes, str(out/'chip.yaml'), chosen[i::workers], i, str(out))
            for i in range(min(workers, len(chosen)))]
    with multiprocessing.get_context('spawn').Pool(len(jobs)) as pool:
        results = pool.map(_simulate_tiles, jobs)
    return dict(min(results, key=lambda r: r['latency']), domain=len(domain), simulated=len(chosen))


def tune(problem,chip,out,workers=4):
    """Native tiles per M/N partition; total cycles include input replication.

    A partition's replication time is at least its largest inbound volume
    divided by the flit size, so partitions already beaten by that bound are
    not distributed in simulation.
    """
    chip=Path(chip).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    arch,capacity,_=hardware(chip)
    kernels=[]
    for ps in partitions(problem,arch['core_num']):
        shape=tuple(n//s for n,s in zip(problem.shape,ps))
        try:placement=matrix_placement(shape,problem.dtype_bytes,capacity)
        except ValueError:continue
        folder=out/('ps_'+'_'.join(map(str,ps)));folder.mkdir(exist_ok=True)
        try:best=best_tile(shape,chip,folder/'tuning',workers)
        except ValueError:continue
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
         baseline=f'Original ATLAS MNK generator; upstream explorer on the proxy top-{BUDGET} native tiles',
         spatial='All legal M/N partitions, no split-K; an explicit standalone harness policy',
         tile_minima_MKN=list(MINIMA),inputs='single DRAM copy; replicas charged by linear mesh all-gather; bound-pruned partitions',one_core_tuning_full_chip_verified=True))
    dump(out/'summary.json',dict(best=best,partitions=len(rows)))
    return best
