"""One steady-state Transformer layer: one method's graph, measured by ATLAS.

Every method follows one protocol (``Search``), like offline autotuning
with a cost model and a few timed runs: its GEMM spaces are scanned by the
analytic proxy (exact bound pruning), an expanded shortlist is compiled and
scored on its schedule (``model.schedule_proxy``), the attention space by
its proxy; layout edges and RMSNorms are scored on their compiled tasks. A
beam over the operator chain proposes ``candidates`` graphs, each is built
whole and scored with its operators chained per core, and the ``keep`` best
structurally different graphs are simulated; the fastest is the result.
Methods differ only in their spaces: ours/local/dram (``Method.boundary``)
choose each GEMM's boundary layout and reduction output contract jointly
with its neighbours; baselines keep the boundary their dataflow prescribes
(``--no-joint`` gives ours the same treatment as an ablation). ``atlas``
uses the upstream natively tuned TP GEMMs (a check of ``tp``). ``deadline``
caps the wall-clock seconds: no simulation starts after it. Weights are
stored once for every method: an M split must rotate B. Each run writes only
its own cycles; ``experiments.compare`` turns separate runs into speedups.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import pickle
from subprocess import TimeoutExpired
from pathlib import Path
import time

from ..gemm.plan import load
from ..gemm.space import Proxy, structure
from ..methods import METHODS, target
from ..output import variants as output_variants
from ..model.attention import AttentionPlan, KVLayout, build_attention
from ..model.schedule_proxy import ScheduleProxy
from ..model.kernels import Gemm
from ..model.transformer import (Workload, build, configuration, edges, norm_score, norms, port, relayout,
                                 column_groups, roles, tune_native)
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
    k = Gemm(workload.problems[name], load(entry['plan']), entry['native'], entry['direction'], entry['gated'],
             column_groups(workload, name))
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


def _graph_job(workload, chip, runs, unique, deadline, key):
    return (key, *simulate_graph(workload, chip, unique[key], runs/key, deadline))


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
    ``(A, C)`` boundary layouts; None if the plan does not compile (for every
    method, so a baseline drops an illegal tile like our spaces do)."""
    started = time.monotonic()
    entry = dict(plan=asdict(plan), native=False, **roles(workload, name))
    try:
        graph, kernel = compiled_operator(workload, chip, hw, name, entry)
    except ValueError:
        return None
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
    """``map`` over ``workers`` processes, or over a shared process pool."""
    if isinstance(workers, ProcessPoolExecutor):
        return list(workers.map(job, items)) if items else []
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


def _entry_score(workload, chip, hw, name, entry):
    """``(compiled score, (A, C) layouts)`` of a GEMM option; None if it does not compile.

    Upstream native kernels (``atlas``, a check of ``tp``) are one fixed option
    per GEMM in upstream's iteration-coded task format: their score is a
    constant (equal in every graph) and only their layouts matter."""
    if entry['native']:
        k = bind_kernel(workload, hw, name, entry)
        return 0.0, (k.input_layout(hw['cores'])[0], k.output_layout(hw['cores']))
    try:
        graph, kernel = compiled_operator(workload, chip, hw, name, entry)
    except ValueError:
        return None
    proxy = ScheduleProxy(chip)
    value = sum(proxy.tasks(op['raw'], graph.tensors) for op in graph.operations)
    return value, (kernel.input_layout(graph.cores)[0], kernel.output_layout(graph.cores))


def _entry_job(workload, chip, hw, item):
    name, entry = item
    if name == 'attention':
        return _attention_score(workload, chip, entry), None
    return _entry_score(workload, chip, hw, name, entry)


def program_digest(graph):
    """Digest of a graph's executable content: its tensors and every operator's
    tasks. Configurations that compile to equal digests run identically."""
    # The digest is an in-process identity used while selecting whole graphs.
    # Pickle preserves the generated task structure and is several times faster
    # than sorting and encoding hundreds of megabytes of JSON here.
    h = hashlib.sha256(pickle.dumps(graph.tensors, protocol=5))
    for op in graph.operations:
        h.update(pickle.dumps(op['raw'], protocol=5))
    return h.hexdigest()


def _whole_job(workload, chip, config, schedule=True):
    """``(whole-graph score or None, program digest)``; ``schedule=False`` for graphs
    the schedule proxy cannot read (upstream native kernels)."""
    graph = build(workload, chip, config)
    return (ScheduleProxy(chip).graph(graph) if schedule else None), program_digest(graph)


def signature(chosen):
    """A graph's structural decisions: per GEMM its spatial/rotation family and
    output contract, for attention its dataflow, partitions and contract
    (tiles and loop orders excluded)."""
    out = []
    for name in ORDER:
        c = chosen[name]
        if name == 'attention':
            out.append((c.dataflow, c.query_parts, c.context_parts, c.output))
        else:
            p = load(c['plan']); out.append((structure(p), getattr(p, 'output', None)))
    return tuple(out)


def pick(ranked, count, program=None):
    """``count`` graphs in rank order, each first with a structure no earlier
    pick has (``signature``); remaining slots, if any, go to the next in rank.
    Graphs whose ``program(item)`` equals an earlier one's would run
    identically and are never picked twice."""
    seen = set(); programs = set(); first = []; rest = []
    for item in ranked:
        if program is not None:
            digest = program(item)
            if digest in programs:
                continue
            programs.add(digest)
        key = signature(item[1])
        (rest if key in seen else first).append(item); seen.add(key)
    return (first+rest)[:count]


class Search:
    """Proxy search of one layer, shared by every method.

    Every option is compiled and scored by ``ScheduleProxy``. An operator meets
    its neighbours only through its boundary layouts (GEMM: A in, C out;
    attention: Q in, O out), so options are grouped by layout: the best
    implementation of each layout is a beam option and the next ``per`` best
    are its alternates. Joint-boundary methods (``ours``/``local``/``dram``)
    also expand reductions into their output contracts; baselines keep the
    contract their dataflow prescribes (GEMMs: their kernel-ranked proposals).
    Layout edges and RMSNorms are scored on their compiled tasks (``Edges``).
    A beam over the operator chain keeps ``candidates`` graphs by these
    additive scores; each is built whole and scored with its operators chained
    per core (``ScheduleProxy.graph``: cross-operator overlap, skew and
    contention). Implementations within a layout interact with the graph only
    through that overlap, so the best graphs then try each operator's
    alternates under the same whole-graph score. The ``keep`` best of
    structurally different graphs (``pick``) are simulated, never two that
    compile to the same program; the fastest is the result. Nothing else is
    simulated.
    """
    def __init__(self, exp, hw, method, workers, joint, per=2):
        self.exp = exp; self.hw = hw; self.method = METHODS[method]; self.workers = workers
        self.joint = joint; self.per = per
        self.edges = Edges(exp.workload, exp.chip, workers)
        self.scores = {}; self.ports = {}; self.alternates = {}; self.programs = {}; self.report = {}
        self.executor = None  # a shared process pool, while one is open

    def group(self, name, scored):
        """Beam options from ``[(choice, score, (in, out) layouts)]``: the best of every
        layout; the next ``per`` of each are recorded as its alternates."""
        layouts = {}
        for choice, value, ports in scored:
            layouts.setdefault((tuple(ports[0]), tuple(ports[1])), []).append((value, token(choice), choice, ports))
        options = []
        for found in layouts.values():
            found.sort(key=lambda x: x[:2]); kept = found[:1+self.per]
            for value, key, choice, ports in kept:
                self.scores[name, key] = value; self.ports[name, key] = ports
            options.append(kept[0][2])
            self.alternates[name, kept[0][1]] = [c for _, _, c, _ in kept[1:]]
        return sorted(options, key=lambda c: self.scores[name, token(c)])

    def attention(self, plans, executor=None):
        """Every attention plan (with output contracts for joint boundaries),
        compiled and scored, grouped by (Q, O) layout. ``executor`` (a process
        pool) may be shared with concurrent work."""
        started = time.monotonic(); workload = self.exp.workload; cores = self.hw['cores']
        if self.joint:
            plans = [c for p in plans for c in output_variants(p)]
        plans = list({token(c): c for c in plans}.values())
        job = partial(_attention_score, workload, self.exp.chip)
        if executor is None:
            values = _pool_map(job, plans, self.workers)
        else:
            values = list(executor.map(job, plans))
        options = self.group('attention', [(p, v, (port(workload, cores, 'attention.Q', p),
                                                   port(workload, cores, 'attention.O', p)))
                                          for p, v in zip(plans, values)])
        self.report['attention'] = dict(plans=len(plans), options=len(options), seconds=time.monotonic()-started)
        return options

    def gemm(self, name, proposals, pool):
        """Options of one GEMM, grouped by (A, C) layout: from the whole compiled
        pool with output contracts (joint), else from the method's proposals."""
        workload = self.exp.workload; job = partial(_entry_job, workload, self.exp.chip, self.hw)
        entry = lambda p: dict(plan=asdict(p), native=False, **roles(workload, name))
        if not self.joint or pool is None:
            found = _pool_map(job, [(name, e) for e in proposals], self.executor or self.workers)
            return self.group(name, [(e, *f) for e, f in zip(proposals, found) if f is not None])
        scored = [(entry(p), value, ports) for p, value, ports in pool]
        best = {}
        for p, value, ports in pool:
            key = (tuple(ports[0]), tuple(ports[1]))
            if key not in best or value < best[key][1]:
                best[key] = (p, value)
        variants = [entry(v) for p, _ in best.values() for v in output_variants(p) if v != p]
        scored += [(e, *f) for e, f in zip(variants, _pool_map(job, [(name, e) for e in variants], self.executor or self.workers))
                   if f is not None]
        return self.group(name, scored)

    def port(self, name, choice):
        op = name.split('.')[0]
        if op == 'kv':
            key = (name, None)
            if key not in self.ports:
                self.ports[key] = port(self.exp.workload, self.hw['cores'], name)
            return self.ports[key]
        return self.ports[op, token(choice)][name.endswith(('.C', '.O'))]

    def config(self, chosen):
        return dict(kernels={k: chosen[k] for k in self.exp.workload.problems}, attention=asdict(chosen['attention']))

    def whole(self, graphs):
        """``(whole score, chosen, configuration, additive score)`` of ``(additive, chosen)`` graphs."""
        configs = [self.config(c) for _, c in graphs]
        schedule = not self.method.native
        values = _pool_map(partial(_whole_job, self.exp.workload, self.exp.chip, schedule=schedule), configs, self.workers)
        for cfg, (_, digest) in zip(configs, values):
            self.programs[Experiment.key(cfg)] = digest
        # Without a whole-graph score (native kernels) graphs keep their additive one.
        return [(s if w is None else w, c, cfg, s) for (w, _), (s, c), cfg in zip(values, graphs, configs)]

    def propose(self, options, width, candidates, keep):
        """Configurations of the ``keep`` graphs to simulate, with the search record."""
        started = time.monotonic(); workload = self.exp.workload
        options = {name: options[name] for name in ORDER}
        graphs = joint_search(workload, options, lambda n, c: self.scores[n, token(c)], self.port, self.edges,
                              width, candidates)
        additive = time.monotonic()
        ranked = sorted(self.whole(graphs), key=lambda x: x[0]); seen = {Experiment.key(x[2]) for x in ranked}
        # Alternates of the best graphs' operators: same boundaries, so only
        # the whole-graph score can tell them apart.
        swaps = []
        program = lambda item: self.programs[Experiment.key(item[2])]
        for _, chosen, _, s in pick(ranked, keep, program):
            for name in ORDER:
                for other in self.alternates.get((name, token(chosen[name])), []):
                    variant = {**chosen, name: other}
                    if Experiment.key(self.config(variant)) not in seen:
                        seen.add(Experiment.key(self.config(variant)))
                        swaps.append((s-self.scores[name, token(chosen[name])]+self.scores[name, token(other)], variant))
        refined = self.whole(swaps)
        ranked = sorted(ranked+refined, key=lambda x: x[0]); chosen = pick(ranked, keep, program)
        alternate = {Experiment.key(x[2]) for x in refined}; simulated = {Experiment.key(x[2]) for x in chosen}
        self.report.update(options={n: len(c) for n, c in options.items()}, edges=len(self.edges.scores),
                           graphs=[dict(additive=s, whole=w, configuration=cfg, alternate=Experiment.key(cfg) in alternate,
                                        program=self.programs[Experiment.key(cfg)][:16],
                                        simulated=Experiment.key(cfg) in simulated) for w, _, cfg, s in ranked],
                           seconds=dict(additive=additive-started, whole=time.monotonic()-additive))
        return [cfg for _, _, cfg, _ in chosen]


def run(workload, chip, out, method='ours', budget=4, workers=WORKERS, tile_min=8, width=64, keep=8, tiles=2,
        deadline=None, joint=True, candidates=32):
    """Search one layer for ``method`` and simulate its ``keep`` proposed graphs (``Search``)."""
    exp = Experiment(workload, chip, out); started = time.monotonic(); stop = deadline and started+deadline
    joint = joint and METHODS[method].boundary
    label = method if joint == METHODS[method].boundary else method+'_nojoint'
    # Weights are preplaced once, like the native TP shards; replicating them
    # would trade unbilled DRAM capacity for speed.
    hw = dict(target(chip, tile_min), single_b=True); layout = KVLayout(workload.attention, hw['cores'])
    # Each method supplies its own fused dataflow on the shared execution backend.
    plans = METHODS[method].attention(layout, hw['sram'])
    if not plans:
        raise ValueError('No SRAM-feasible attention plan')
    search = Search(exp, hw, method, workers, joint, tiles); report = {}; clock = {}
    # The attention search and the GEMM scans are independent: they share one
    # pool, so attention plans fill workers that finished their scan pieces.
    executor = ProcessPoolExecutor(max(1, workers))
    try:
        if METHODS[method].native:
            options = {'attention': search.attention(plans, executor)}
        else:
            names = list(workload.problems)
            parts = max(1, workers//len(names))
            jobs = [(name, part) for name in names for part in range(parts)]
            scans = [executor.submit(_scan_job, workload, chip, method, hw, budget, tiles, parts, job) for job in jobs]
            options = {'attention': search.attention(plans, executor)}
            clock['attention'] = time.monotonic()-started
            found = [f.result() for f in scans]
    finally:
        executor.shutdown()
    clock.setdefault('attention', time.monotonic()-started)
    if METHODS[method].native:
        kernels = tune_native(workload, chip, exp.out/'native_tuning', workers)
        options.update({k: search.gemm(k, [v], None) for k, v in configuration(kernels, plans[0])['kernels'].items()})
    else:
        # Each GEMM scan splits exactly into pieces of whole structural
        # families (``search.diverse_part``), merged per GEMM below.
        clock['scan'] = time.monotonic()-started-clock['attention']  # its tail after attention
        # The four GEMMs compile concurrently into one pool (no per-GEMM tails).
        def nominate(name):
            pieces = [piece for (n, _), piece in zip(jobs, found) if n == name]
            nomination = {}; pool = [] if joint else None
            proposals, scanned = propose_gemm(workload, chip, method, hw, budget, tiles, name, pieces,
                                              workers=search.executor, report=nomination, pool=pool)
            return name, search.gemm(name, proposals, pool), dict(
                scanned=scanned, proposals=[e['plan'] for e in proposals], nomination=nomination)
        with ProcessPoolExecutor(max(1, workers)) as shared, ThreadPoolExecutor(len(names)) as threads:
            search.executor = shared
            for name, chosen, record in threads.map(nominate, names):
                options[name] = chosen; report[name] = dict(record, options=len(chosen))
                print(f'{name}: {record["nomination"]["fine_scored"]} compiled scores -> {len(chosen)} options',
                      flush=True)
        search.executor = None
        clock['nominate'] = time.monotonic()-started-clock['scan']-clock['attention']
        dump(exp.out/'nomination.json', report)
    configs = search.propose(options, width, candidates, keep)
    clock['propose'] = search.report['seconds']
    print(f'{len(search.report["graphs"])} graphs scored whole -> {len(configs)} simulated', flush=True)
    simulated = time.monotonic()
    rows = exp.measure_many(configs, label+('_joint' if joint else '_graph'), workers, stop, abort=False)
    best = min(rows, key=cycles); clock['simulate'] = time.monotonic()-simulated
    dump(exp.out/'search.json', dict(gemm=report, **search.report, budget=budget, tiles=tiles, width=width,
                                     candidates=candidates, keep=keep, joint_boundary=joint, clock=clock))
    summary = dict(experiment='transformer', workload=asdict(workload), chip=str(Path(chip).resolve()), method=label,
                   best=best, cycles=cycles(best), seconds=time.monotonic()-started, deadline=deadline,
                   hit_deadline=bool(stop and time.monotonic() > stop), sha256=exp.provenance, clock=clock,
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
                        ('ffn', 1536), ('budget', 4), ('workers', WORKERS), ('tile-min', 8),
                        ('width', 64), ('keep', 8), ('tiles', 2), ('candidates', 32)]:
        parser.add_argument('--'+name, type=int, default=value)
    parser.add_argument('--deadline', type=float, help='common wall-clock cap in seconds (no graph starts later)')
    parser.add_argument('--no-joint', action='store_true', help='ablation: no joint boundary search')
    a = parser.parse_args()
    if min(a.batch, a.context, a.query_heads, a.kv_heads, a.head_dim, a.ffn, a.budget, a.workers,
           a.tile_min, a.width, a.keep, a.tiles, a.candidates) < 1:
        parser.error('Counts must be positive')
    workload = Workload(a.batch, a.context if a.mode == 'prefill' else 1, a.context, a.query_heads, a.kv_heads,
                        a.head_dim, a.ffn)
    run(workload, a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.width, a.keep, a.tiles, a.deadline,
        not a.no_joint, a.candidates)


if __name__ == '__main__':
    main()
