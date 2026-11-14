"""One steady-state Transformer layer: one method's graph, measured by ATLAS.

Each operator has a few options:
for a GEMM the method's proxy proposals (``tp``: the fixed TP grid along the
GEMM's mesh direction, tiles only), for attention the shared context splits
plus the method's additions. ``atlas`` uses the upstream natively tuned TP
GEMMs and collectives instead (a check of ``tp``). Every option is simulated
alone with its inputs in place, and every layout conversion between adjacent
choices is scored from its busiest port or mesh link. Methods whose plans
move DRAM residency (``Method.boundary``: ours, local, dram) also expand the
output contract of every reduction and search the graph jointly: beam search
over this chain proposes the ``keep`` best complete graphs, which are
simulated whole. Baselines keep the boundary their dataflow prescribes, so
each operator takes its fastest standalone option and the graph is simulated
once (relayouts at mismatched boundaries are still charged); ``--no-joint``
gives ours the same treatment as an ablation.
The chain score misses overlap between operators, so a coordinate descent
then simulates, in parallel, every graph that changes one decision to one of
its few best standalone options, keeps the best, and repeats up to
``passes`` times while it improves. ``deadline`` caps the wall-clock seconds,
so all methods can be held to one limit: standalone options are measured in
proxy order until half of it (each operator's best-ranked option always
completes; unmeasured options are dropped), and after it no further whole
graph starts and running ones are aborted (the first joint graphs always
complete). Weights are stored once for every method:
an M split must rotate B. Each run writes only its own cycles;
``experiments.compare`` turns separate runs into speedups.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from subprocess import TimeoutExpired
from pathlib import Path
import time

from ..gemm.plan import load
from ..gemm.space import Proxy, structure
from ..methods import METHODS, target
from ..output import variants as output_variants
from ..model.attention import AttentionPlan, KVLayout, build_attention
from ..model.attention_proxy import AttentionProxy
from ..model.schedule_proxy import ScheduleProxy
from ..model.kernels import Gemm
from ..model.transformer import (Workload, build, configuration, edges, norm_score, norms, port, relayout,
                                 roles, tune_native)
from ..search import beam, diverse, diverse_part, measure_all, merge_diverse
from ..resources import breakdown, report as resource_report
from ..sim import WORKERS, dump, fingerprint
from ..tasks import Graph
from . import attention as standalone_attention

cycles = lambda row: row['result']['e2e_cycles']


def simulate_graph(workload, chip, config, folder, deadline=None):
    """Build and simulate one whole graph; a process-pool job (lowering is Python-bound)."""
    run = build(workload, chip, config).simulate(folder, deadline=deadline)
    return run['result'], run['graph']


def simulate_operator(workload, chip, hw, out, name, choice, deadline=None):
    """Standalone cycles of one operator option with its inputs already in place."""
    if name == 'attention':
        return standalone_attention.measure(workload.attention, chip, out/'attention', choice, placed=True,
                                            deadline=deadline)['result']['e2e_cycles']
    key = hashlib.sha256(json.dumps([name, choice], sort_keys=True).encode()).hexdigest()[:20]
    graph = operator_graph(workload, chip, hw, name, choice)
    return graph.simulate(out/'operators'/key, deadline=deadline)['result']['e2e_cycles']


def operator_graph(workload, chip, hw, name, choice):
    """The same placed-input GEMM, including fused work, for scoring and measurement."""
    return compiled_operator(workload, chip, hw, name, choice)[0]


def compiled_operator(workload, chip, hw, name, choice):
    """``(graph, kernel)`` of ``operator_graph``; the kernel knows its boundary layouts."""
    graph = Graph(chip); kernel = bind_kernel(workload, hw, name, choice).prepare(graph, name)
    regions, tensors = kernel.input_layout(graph.cores)
    for core, t in enumerate(tensors):
        graph.tensor(core, 'x', t['shape'], t['strides'])
    kernel.compile(graph, [replace(r, name='x') for r in regions],
                   rope_width=workload.attention.hidden if workload.rope and name == 'qkv' else 0)
    return graph, kernel


def bind_kernel(workload, hw, name, entry):
    k = Gemm(workload.problems[name], load(entry['plan']), entry['native'], entry['direction'], entry['gated'])
    return k.bind(hw['cores'], hw['sram'], hw['dram'])


class Experiment:
    """Measures each distinct whole-graph configuration once per run."""
    def __init__(self, workload, chip, out):
        self.workload = workload; self.chip = Path(chip).resolve(); self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.provenance = fingerprint(chip); self.rows = []; self.seen = {}

    @staticmethod
    def key(config):
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]

    def row(self, config, label):
        key = self.key(config); result, manifest, folder = self.seen[key]
        row = dict(label=label, key=key, configuration=config, result=result,
                   peak_dram_bytes=max(manifest['peak_dram_bytes']), path=str(folder),
                   layout_noc_bytes=sum(e['noc_data_bytes'] for e in manifest['edges']),
                   layout_write_bytes=sum(e['dram_write_bytes'] for e in manifest['edges']),
                   resources=manifest['resources'])
        return row

    def record(self, row):
        self.rows.append(row); dump(self.out/'measurements.json', self.rows)
        print(f'{row["label"]}: {cycles(row)} cycles, peak {row["peak_dram_bytes"]} B/core', flush=True)

    def measure_many(self, configs, label, workers, deadline=None, abort=True):
        """Measure independent graphs in parallel, in order, until ``deadline``;
        with ``abort`` graphs still running at the deadline are dropped.
        Each distinct graph is simulated once per run; repeats reuse its row."""
        unique = {self.key(cfg): cfg for cfg in configs}
        todo = [k for k in unique if k not in self.seen]
        if todo:
            job = partial(_graph_job, self.workload, self.chip, self.out/'runs', unique, deadline if abort else None)
            try:
                done = measure_all(todo, job, workers, processes=True, deadline=deadline)
            except TimeoutExpired:
                done = []
            for key, result, manifest in done:
                self.seen[key] = (result, manifest, self.out/'runs'/key)
        rows = [self.row(cfg, label) for key, cfg in unique.items() if key in self.seen]
        if not rows:
            raise TimeoutExpired('graph', 0)
        for row in rows:
            self.record(row)
        return rows


def _boundary(workload, hw, name, entry):
    try:
        k = bind_kernel(workload, hw, name, entry)
    except ValueError:
        return None
    return k.input_layout(hw['cores'])[0], k.output_layout(hw['cores'])


def _operator_job(workload, chip, hw, out, jobs, deadline, i):
    return i, simulate_operator(workload, chip, hw, out, *jobs[i], deadline=deadline)


def _graph_job(workload, chip, runs, unique, deadline, key):
    return (key, *simulate_graph(workload, chip, unique[key], runs/key, deadline))


def variants(config, edit, choices):
    """Copies of ``config`` with one decision replaced by each choice."""
    result = []
    for choice in choices:
        cfg = deepcopy(config); edit(cfg, choice); result.append(cfg)
    return result


def set_kernel(name):
    return lambda cfg, choice: cfg['kernels'][name].update(choice)


def set_attention(cfg, plan):
    cfg['attention'] = asdict(plan)


def descend(exp, start, neighbours, label, passes, workers, deadline=None):
    """Coordinate descent over whole graphs; ``neighbours(best)`` gives each decision's alternatives."""
    best = start
    for sweep in range(passes):
        if deadline and time.monotonic() >= deadline:
            break
        configs = [cfg for name, choices in neighbours(best).items()
                   for cfg in variants(best['configuration'], set_attention if name == 'attention' else set_kernel(name),
                                       choices)]
        try:
            rows = exp.measure_many(configs, f'{label}_descent{sweep}', workers, deadline)
        except TimeoutExpired:
            break
        candidate = min([best, *rows], key=cycles)
        if candidate['key'] == best['key']:
            break
        best = candidate
    return best


def token(choice):
    """Hashable identity of a kernel entry or an attention plan."""
    return json.dumps(asdict(choice) if isinstance(choice, AttentionPlan) else choice, sort_keys=True)


def _relayout_job(workload, chip, request):
    if request[0] == 'norm':
        return norm_score(workload, chip, request[3])
    return relayout(workload, chip, *request)


class Edges:
    """Compiled-schedule scores of layout edges, one per distinct ``(target
    port, alias, source layout, target layout)``, and of RMSNorms
    ``('norm', operator, None, layout)``; ``prepare`` scores many in
    parallel. Port layouts are interned once per list object (ports are cached
    lists), so lookups do not rehash thousands of regions."""
    def __init__(self, workload, chip, workers):
        self.workload = workload; self.chip = chip; self.workers = workers; self.scores = {}
        self.ids = {}; self.layouts = {}

    def layout(self, regions):
        found = self.ids.get(id(regions))
        if found is None or found[0] is not regions:
            found = (regions, self.layouts.setdefault(tuple(regions), len(self.layouts)))
            self.ids[id(regions)] = found  # holds ``regions``, so its id stays unique
        return found[1]

    def key(self, dst, alias, source, target):
        return dst, alias, None if source is None else self.layout(source), self.layout(target)

    def prepare(self, requests):
        todo = list({self.key(*r): r for r in requests if self.key(*r) not in self.scores}.items())
        if not todo:
            return
        job = partial(_relayout_job, self.workload, self.chip)
        if self.workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(min(self.workers, len(todo))) as pool:
                # Costs vary by orders of magnitude between edge kinds: small chunks balance them.
                values = list(pool.map(job, [r for _, r in todo], chunksize=max(1, min(8, len(todo)//(16*self.workers)))))
        else:
            values = [job(r) for _, r in todo]
        self.scores.update(zip((k for k, _ in todo), values))

    def __call__(self, dst, alias, source, target):
        key = self.key(dst, alias, source, target)
        if key not in self.scores:
            self.prepare([(dst, alias, source, target)])
        return self.scores[key]


def joint_search(workload, options, node, port_of, edge_score, width, keep):
    """Beam search over the layer's operator chain: ``node(name, choice)`` plus
    the edges between chosen ports. The ``keep`` best ``(score, {name: choice})``.
    Each step first scores, in parallel, the edges it will charge. An
    RMSNorm (``norms``) is charged with the operator whose output it reads."""
    graph_edges = [(s, d, alias, s.split('.')[0], d.split('.')[0]) for s, d, alias in edges(workload)]
    def charged(name, chosen):
        pick = lambda o: None if o == 'kv' else chosen.get(o, False)
        # Charge each edge once, when its later endpoint is chosen.
        found = [(dst, alias, port_of(src, pick(a)), port_of(dst, pick(b))) for src, dst, alias, a, b in graph_edges
                 if name in (a, b) and pick(a) is not False and pick(b) is not False]
        return found+[('norm', op, None, port_of(src, chosen[name])) for src, op in norms(workload)
                      if src.split('.')[0] == name]
    def cost(name, index, partial):
        chosen = {**{o: options[o][i] for o, i in partial.items()}, name: options[name][index]}
        return node(name, chosen[name])+sum(edge_score(*e) for e in charged(name, chosen))
    def before(name, indices, partials):
        edge_score.prepare([e for partial in partials for i in indices
                            for e in charged(name, {**{o: options[o][j] for o, j in partial.items()},
                                                    name: options[name][i]})])
    ranked = beam([(name, range(len(choices))) for name, choices in options.items()], cost, width, keep,
                  before if hasattr(edge_score, 'prepare') else None)
    return [(score, {name: options[name][i] for name, i in chosen.items()}) for score, chosen in ranked]


class Chain:
    """Joint proxy of a graph: standalone operator cycles plus layout-edge scores.

    Each option is simulated once with its inputs already in its own layout;
    each layout edge is scored on its compiled conversion tasks (``Edges``).
    """
    def __init__(self, exp, hw, workers, attention_budget=32, deadline=None, boundary=True, edges=None):
        self.exp = exp; self.hw = hw; self.workers = workers; self.deadline = deadline
        self.boundary = boundary
        self.cycles = {}; self.ports = {}
        self.edges = edges or Edges(exp.workload, exp.chip, workers)
        self.kernels = {}
        self.attention_proxy = AttentionProxy(exp.chip); self.attention_budget = attention_budget
        self.attention_report = {}; self.attention_selected = None

    def kernel(self, name, entry):
        return bind_kernel(self.exp.workload, self.hw, name, entry)

    def bound(self, name, entry):
        """A GEMM option's ``(A, C)`` boundary layouts, compiled once per run; None if illegal."""
        key = (name, token(entry))
        if key not in self.kernels:
            self.kernels[key] = _boundary(self.exp.workload, self.hw, name, entry)
        return self.kernels[key]

    def compile(self, options):
        """Compile every GEMM option once, in parallel (compiling dominates large meshes)."""
        jobs = list({(n, token(c)): (n, c) for n, choices in options.items() if n != 'attention' for c in choices
                     if (n, token(c)) not in self.kernels}.values())
        if jobs:
            with ProcessPoolExecutor(max(1, min(self.workers, len(jobs)))) as pool:
                found = pool.map(_boundary, *zip(*((self.exp.workload, self.hw, n, c) for n, c in jobs)))
                for (n, c), value in zip(jobs, found):
                    self.kernels[n, token(c)] = value

    def measure(self, jobs, deadline):
        """Standalone cycles of ``(name, choice)`` jobs, in order, until ``deadline``."""
        job = partial(_operator_job, self.exp.workload, self.exp.chip, self.hw, self.exp.out, jobs, deadline)
        try:
            rows = measure_all(range(len(jobs)), job, self.workers, processes=True, deadline=deadline)
        except TimeoutExpired:
            rows = []
        for i, value in rows:
            self.cycles[jobs[i][0], token(jobs[i][1])] = value

    def boundaries(self, name, choice):
        """The boundary variants of an option a joint search may choose from."""
        if not self.boundary:
            return [choice]  # the dataflow's own boundary
        if name == 'attention':
            return output_variants(choice)
        if choice['native']:
            return [choice]  # upstream checker retains its original collective
        return [dict(choice,plan=asdict(p)) for p in output_variants(load(choice['plan']))]

    def select_attention(self, plans):
        """The method's attention plans with their boundary variants, cut by the
        attention proxy to its budget; computed once per run."""
        if self.attention_selected is None:
            candidates = list({token(c): c for p in plans for c in self.boundaries('attention', p)}.values())
            self.attention_selected = self.attention_proxy.select(self.exp.workload.attention, candidates,
                                                                  self.attention_budget, placed=True,
                                                                  workers=self.workers)
            self.attention_report = dict(plans=len(candidates), measured=len(self.attention_selected),
                                         proxy_budget=self.attention_budget)
        return self.attention_selected

    def prepare(self, options):
        """Measure every distinct option once; drop illegal and behaviourally equal options."""
        options = {name: (self.select_attention(choices) if name == 'attention' else
                          [c for choice in choices for c in self.boundaries(name, choice)])
                   for name, choices in options.items()}
        self.compile(options)
        legal = lambda name, c: name == 'attention' or self.bound(name, c) is not None
        options = {name: list({token(c): c for c in choices if legal(name, c)}.values())
                   for name, choices in options.items()}
        # Options are in proxy order. Each operator's first option always
        # completes; the others are measured until the deadline, then dropped.
        # Interleave operators by rank so a deadline cuts every operator's tail.
        ranked = sorted(((rank, name, c) for name, choices in options.items() for rank, c in enumerate(choices)),
                        key=lambda x: x[0])
        jobs = list({(name, token(c)): (rank, name, c) for rank, name, c in ranked
                     if (name, token(c)) not in self.cycles}.values())
        self.measure([(n, c) for r, n, c in jobs if r == 0], None)
        self.measure([(n, c) for r, n, c in jobs if r > 0], self.deadline)
        options = {name: [c for c in choices if (name, token(c)) in self.cycles] for name, choices in options.items()}
        self.measured = {name: len(choices) for name, choices in options.items()}
        # Options with equal measured cycles and equal boundary layouts behave
        # the same in every graph (e.g. loop orders of a one-tile kernel).
        def behaviour(name, c):
            ends = ('Q', 'O') if name == 'attention' else ('A', 'C')
            return (self.cycles[name, token(c)], *(repr(self.port(f'{name}.{e}', c)) for e in ends))
        return {name: list({behaviour(name, c): c for c in reversed(choices)}.values())[::-1]
                for name, choices in options.items()}

    def port(self, name, choice):
        op = name.split('.')[0]; key = (name, None if choice is None else token(choice))
        if key not in self.ports:
            if op in ('attention', 'kv') or choice is None:
                self.ports[key] = port(self.exp.workload, self.hw['cores'], name, choice)
            else:
                self.ports[key] = self.bound(op, choice)[name.endswith('.C')]
        return self.ports[key]

    def search(self, options, width, keep):
        """The ``keep`` best distinct complete assignments ``{operator: choice}`` by joint proxy."""
        return joint_search(self.exp.workload, options, lambda name, c: self.cycles[name, token(c)],
                            self.port, self.edges, width, keep)

    def neighbours(self, options, count):
        """``row -> {decision: alternatives}``: per decision, the ``count`` options of
        lowest standalone cycles other than the one ``row`` uses."""
        def alternatives(row):
            cfg = row['configuration']
            current = {**cfg['kernels'], 'attention': AttentionPlan(**cfg['attention'])}
            return {name: sorted((c for c in choices if token(c) != token(current[name])),
                                 key=lambda c: self.cycles[name, token(c)])[:count]
                    for name, choices in options.items()}
        return alternatives

    def propose(self, options, width, keep):
        """Measure the options alone; the ``keep`` best complete graphs of the chain as configurations.
        Without joint boundary search, the one graph of every operator's fastest option."""
        options = self.prepare(options)
        if self.boundary:
            proposals = self.search(options, width, keep)
        else:
            best = {n: min(c, key=lambda x: self.cycles[n, token(x)]) for n, c in options.items()}
            proposals = [(sum(self.cycles[n, token(c)] for n, c in best.items()), best)]
        configs = [dict(kernels={k: c[k] for k in self.exp.workload.problems}, attention=asdict(c['attention']))
                   for _, c in proposals]
        return options, proposals, configs


ORDER = ('qkv', 'attention', 'output', 'expand', 'contract')

# Expand both coarse structural-family and within-family budgets, not simulator
# budgets. Two scoring channels can nominate up to 2 * factor**2 * budget * tiles.
GEMM_COARSE_FACTOR = 4


def _gemm_scoring(workload, chip, method, hw, name):
    problem = workload.problems[name]; proxy = Proxy(chip); m = METHODS[method]
    score = lambda p: (k := proxy.kernel(problem, p), k+proxy.distribution(problem, p))
    bound = lambda q: (k := proxy.lower(problem, q), k+proxy.distribution(problem, q))
    space = lambda skip=None: m.space(problem, skip=skip, direction=roles(workload, name)['direction'], orient=False, **hw)
    return problem, proxy, m, score, bound, space


def scan_gemm(workload, chip, method, hw, budget, tiles, name, part=0, parts=1):
    """One piece of the expanded cheap scan; bounds apply only to this coarse score."""
    _, _, _, score, bound, space = _gemm_scoring(workload, chip, method, hw, name)
    return diverse_part(space, score, budget*GEMM_COARSE_FACTOR, structure,
                        tiles*GEMM_COARSE_FACTOR, bound, part, parts)


def _scan_job(workload, chip, method, hw, budget, tiles, parts, job):
    return scan_gemm(workload, chip, method, hw, budget, tiles, *job, parts)


def _schedule_score(workload, chip, hw, name, plan, ports=False):
    """Compiled schedule score of one GEMM plan; with ``ports`` also its
    ``(A, C)`` boundary layouts; None if the plan does not compile."""
    started = time.monotonic()
    entry = dict(plan=asdict(plan), native=False, **roles(workload, name))
    try:
        graph, kernel = compiled_operator(workload, chip, hw, name, entry)
    except ValueError:
        if ports:
            return None
        raise
    compiled = time.monotonic()
    proxy = ScheduleProxy(chip)
    value = sum(proxy.tasks(op['raw'], graph.tensors) for op in graph.operations)
    out = dict(kernel=value, compile_seconds=compiled-started, proxy_seconds=time.monotonic()-compiled)
    if ports:
        out['ports'] = (kernel.input_layout(graph.cores)[0], kernel.output_layout(graph.cores))
    return out


def _attention_score(workload, chip, plan):
    graph = build_attention(workload.attention, chip, plan, True); proxy = ScheduleProxy(chip)
    return sum(proxy.tasks(op['raw'], graph.tensors) for op in graph.operations)


def _pool_map(job, items, workers):
    if workers > 1 and len(items) > 1:
        with ProcessPoolExecutor(min(workers, len(items))) as executor:
            return list(executor.map(job, items, chunksize=max(1, min(4, len(items)//(16*workers)))))
    return list(map(job, items))


def propose_gemm(workload, chip, method, hw, budget, tiles, name, pieces=None, *, workers=1, report=None, pool=None):
    """Proxy proposals of one graph GEMM, in proxy order; ``(entries, scanned)``.

    Graph inputs arrive through layout edges whose cost depends on the
    neighbours. Rank by the kernel alone and, in the same scan, by kernel
    plus single-copy distribution, to retain alternatives with cheap edges.
    A cheap scan retains an expanded pool, then compiled schedule scores select
    the original number of families/tiles. No coarse bound prunes fine scores.
    One orientation is scanned and each proposal is followed by its transposed twin, so
    edges and the simulator decide between them.
    ``pieces`` are the ``scan_gemm`` results of a scan split over processes
    (default: one piece scanned here); merging them is exact. A ``pool`` list
    receives every compiled ``(plan, score, (A, C) layouts)`` of the pool and
    of the transposed twin of each family's best plan, for ``nominate_joint``.
    """
    started = time.monotonic()
    problem, proxy, m, coarse_score, _, _ = _gemm_scoring(workload, chip, method, hw, name)
    pieces = pieces or [scan_gemm(workload, chip, method, hw, budget, tiles, name)]
    merged = merge_diverse([row for rows, _ in pieces for row in rows], coarse_score,
                           budget*GEMM_COARSE_FACTOR, structure, tiles*GEMM_COARSE_FACTOR)
    plans = [p for p, _ in merged]
    job = partial(_schedule_score, workload, chip, hw, name, ports=pool is not None)
    values = found = _pool_map(job, plans, workers)
    plans, values = [p for p, v in zip(plans, values) if v is not None], [v for v in values if v is not None]
    if pool is not None:
        pool.extend((p, v['kernel'], v['ports']) for p, v in zip(plans, values))
        if m.oriented(hw['cores']):
            # A structural family fixes the boundary layouts up to tiles: the
            # transposed twin of each family's best plan stands for its layouts.
            best = {}
            for p, v in zip(plans, values):
                if structure(p) not in best or v['kernel'] < best[structure(p)][1]:
                    best[structure(p)] = (p, v['kernel'])
            twins = [replace(p, transpose=True) for p, _ in best.values()]
            twin_values = _pool_map(job, twins, workers); found = values+twin_values
            pool.extend((p, v['kernel'], v['ports']) for p, v in zip(twins, twin_values) if v is not None)
    scores = {p: (v['kernel'], v['kernel']+proxy.distribution(problem, p)) for p, v in zip(plans, values)}
    chosen = diverse(lambda skip=None: iter(plans), scores.__getitem__, budget, structure, tiles)[0]
    scanned = sum(n for _, n in pieces)
    ranked = sorted((p for p, _ in chosen), key=lambda p: scores[p][0])
    if report is not None:
        selected = set(ranked)
        report.update(coarse_factor=GEMM_COARSE_FACTOR, coarse_pool=len(plans), fine_scored=len(found),
                      selected_before_twins=len(ranked), seconds=time.monotonic()-started,
                      candidates=[dict(plan=asdict(p), coarse=list(coarse_score(p)), fine=list(scores[p]),
                                       selected=p in selected, **{k: x for k, x in v.items() if k != 'ports'})
                                  for p, v in zip(plans, values)])
    if m.oriented(hw['cores']):
        ranked = [q for p in ranked for q in (p, replace(p, transpose=True))]
    return [dict(plan=asdict(p), native=False, **roles(workload, name)) for p in ranked], scanned


def nominate_joint(chain, pools, attention, width, keep):
    """GEMM options the joint proxy wants, beyond each GEMM's kernel-ranked proposals.

    A GEMM affects its neighbours only through its boundary layouts, so among
    pool plans with equal ``(A, C)`` layouts the one of least compiled score
    is best in every graph; every distinct layout keeps its best plan (output
    contracts of reductions included). A beam over the layer chain then
    scores each graph by compiled operator scores plus compiled layout edges
    (no simulation), and the GEMM options of its ``keep`` best graphs are
    returned, in graph order, as ``{name: [entry]}`` with a report.
    """
    started = time.monotonic(); exp = chain.exp; workload = exp.workload; hw = chain.hw; workers = chain.workers
    entry = lambda name, p: dict(plan=asdict(p), native=False, **roles(workload, name))
    best = {}
    def keep_best(name, plan, value, ports):
        key = (name, tuple(ports[0]), tuple(ports[1]))
        if key not in best or value < best[key][1]:
            best[key] = (plan, value, ports)
    for name, rows in pools.items():
        for plan, value, ports in rows:
            keep_best(name, plan, value, ports)
    # Output contracts of each layout's best plan (reductions only).
    variants = [(name, v) for (name, _, _), (p, _, _) in list(best.items())
                for v in output_variants(p) if v != p]
    for (name, v), found in zip(variants, _pool_map(partial(_variant_job, workload, exp.chip, hw), variants, workers)):
        if found is not None:
            keep_best(name, v, found['kernel'], found['ports'])
    options = {name: [] for name in pools}; node = {}; ports = {}
    for (name, _, _), (p, value, layout) in best.items():
        e = entry(name, p); options[name].append(e); node[name, token(e)] = value; ports[name, token(e)] = layout
    plans = chain.select_attention(attention)
    for plan, value in zip(plans, _pool_map(partial(_attention_score, workload, exp.chip), plans, workers)):
        node['attention', token(plan)] = value
    options = {name: options[name] if name != 'attention' else plans for name in ORDER}
    def port_of(name, choice):
        op = name.split('.')[0]
        if op in ('attention', 'kv'):
            return chain.port(name, choice)
        return ports[op, token(choice)][name.endswith('.C')]
    graphs = joint_search(workload, options, lambda name, c: node[name, token(c)], port_of, chain.edges, width, keep)
    picked = {name: [] for name in pools}
    for _, chosen in graphs:
        for name in pools:
            base = dict(chosen[name], plan=asdict(replace(load(chosen[name]['plan']), output='sharded')))
            if all(token(base) != token(e) for e in picked[name]):
                picked[name].append(base)
    report = dict(seconds=time.monotonic()-started, layouts={name: len(options[name]) for name in pools},
                  attention=len(plans), edges=len(chain.edges.scores),
                  graphs=[dict(score=s, configuration=dict(kernels={k: c[k] for k in pools}, attention=asdict(c['attention'])))
                          for s, c in graphs])
    return picked, report


def _variant_job(workload, chip, hw, item):
    name, plan = item
    return _schedule_score(workload, chip, hw, name, plan, ports=True)


def run(workload, chip, out, method='ours', budget=4, workers=WORKERS, tile_min=8, passes=2, width=64, keep=8, tiles=2,
        refine=3, deadline=None, attention_budget=32, joint=True):
    exp = Experiment(workload, chip, out); started = time.monotonic(); stop = deadline and started+deadline
    joint = joint and METHODS[method].boundary
    label = method if joint == METHODS[method].boundary else method+'_nojoint'
    # Weights are preplaced once, like the native TP shards; replicating them
    # would trade unbilled DRAM capacity for speed.
    hw = dict(target(chip, tile_min), single_b=True); layout = KVLayout(workload.attention, hw['cores'])
    # Each method supplies its own fused dataflow on the shared execution backend.
    options = {'attention': METHODS[method].attention(layout, hw['sram'])}
    if not options['attention']:
        raise ValueError('No SRAM-feasible attention plan')
    report = {}
    chain = Chain(exp, hw, workers, attention_budget, stop and started+deadline/2, joint)
    if METHODS[method].native:
        kernels = tune_native(workload, chip, exp.out/'native_tuning', workers)
        options.update({k: [v] for k, v in configuration(kernels, options['attention'][0])['kernels'].items()})
    else:
        # The GEMM scans are independent, and each splits exactly into pieces
        # of whole structural families (``search.diverse_part``): run them all
        # in one pool of ``workers`` processes and merge per GEMM.
        names = list(workload.problems)
        parts = max(1, workers//len(names))
        jobs = [(name, part) for name in names for part in range(parts)]
        with ProcessPoolExecutor(max(1, min(workers, len(jobs)))) as pool:
            found = list(pool.map(partial(_scan_job, workload, chip, method, hw, budget, tiles, parts), jobs))
        pools = {}
        for name in names:
            pieces = [piece for (n, _), piece in zip(jobs, found) if n == name]
            nomination = {}; pools[name] = [] if joint else None
            entries, scanned = propose_gemm(workload, chip, method, hw, budget, tiles, name, pieces,
                                            workers=workers, report=nomination, pool=pools[name])
            options[name] = entries  # in proxy order, so a deadline cuts the tail
            report[name] = dict(scanned=scanned, chosen=[e['plan'] for e in entries], nomination=nomination)
            print(f'{name}: {nomination["fine_scored"]} compiled proxy scores -> '
                  f'{len(entries)} GEMM proposals', flush=True)
        if joint:
            # Layouts that only pay off at the boundaries: the GEMM options of
            # the best graphs under the compiled proxy join the proposals.
            picked, report['joint'] = nominate_joint(chain, pools, options['attention'], width, keep)
            for name, extra in picked.items():
                known = {token(e) for e in options[name]}
                added = [e for e in extra if token(e) not in known]
                options[name] = options[name]+added; report[name]['joint_added'] = [e['plan'] for e in added]
            print('joint proxy nomination: ' + ', '.join(f'{n} +{len(report[n]["joint_added"])}' for n in names)
                  + f' ({report["joint"]["seconds"]:.0f} s, {report["joint"]["edges"]} edges)', flush=True)
        dump(exp.out/'nomination.json', report)
    options, proposals, configs = chain.propose({k: options[k] for k in ORDER}, width, keep)
    best = min(exp.measure_many(configs, label+('_joint' if joint else '_graph'), workers, stop, abort=False), key=cycles)
    if joint:
        best = descend(exp, best, chain.neighbours(options, refine), label, passes, workers, stop)
    dump(exp.out/'search.json', dict(gemm=report, attention=chain.attention_report, measured=chain.measured, joint=[dict(score=score, configuration=cfg)
                                                          for (score, _), cfg in zip(proposals, configs)],
                                      budget=budget, tiles=tiles, passes=passes, width=width, keep=keep, refine=refine,
                                      joint_boundary=joint))
    summary = dict(experiment='transformer', workload=asdict(workload), chip=str(Path(chip).resolve()), method=label,
                   best=best, cycles=cycles(best), seconds=time.monotonic()-started, deadline=deadline,
                   hit_deadline=bool(stop and time.monotonic() > stop), sha256=exp.provenance,
                   **resource_report(best['resources']['total'], cycles(best), chip),
                   breakdown=dict(total=breakdown(best['result']),
                                  operators={k: breakdown(v) for k, v in best['result']['operators'].items()}))
    dump(exp.out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'cycles', 'seconds')}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chip', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('prefill', 'decode'), default='prefill')
    parser.add_argument('--method', choices=list(METHODS), default='ours')
    for name, value in [('batch', 1), ('context', 128), ('query-heads', 8), ('kv-heads', 2), ('head-dim', 64),
                        ('ffn', 1536), ('budget', 4), ('workers', WORKERS), ('tile-min', 8), ('passes', 2),
                        ('width', 64), ('keep', 8), ('tiles', 2), ('refine', 3), ('attention-budget', 32)]:
        parser.add_argument('--'+name, type=int, default=value)
    parser.add_argument('--deadline', type=float, help='common wall-clock cap in seconds (no graph starts later)')
    parser.add_argument('--no-joint', action='store_true', help='ablation: no joint boundary search')
    a = parser.parse_args()
    if min(a.batch, a.context, a.query_heads, a.kv_heads, a.head_dim, a.ffn, a.budget, a.workers,
           a.tile_min, a.width, a.keep, a.tiles) < 1 or min(a.passes, a.refine) < 0:
        parser.error('Counts must be positive')
    workload = Workload(a.batch, a.context if a.mode == 'prefill' else 1, a.context, a.query_heads, a.kv_heads,
                        a.head_dim, a.ffn)
    run(workload, a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.passes, a.width, a.keep, a.tiles, a.refine,
        a.deadline,a.attention_budget,not a.no_joint)


if __name__ == '__main__':
    main()
