"""One steady-state Transformer layer: one method's graph, measured by ATLAS.

Every method chooses its graph the same way. Each operator has a few options:
for a GEMM the method's proxy proposals (``tp``: the fixed TP grid along the
GEMM's mesh direction, tiles only), for attention the shared context splits
plus the method's additions. ``atlas`` uses the upstream natively tuned TP
GEMMs and collectives instead (a check of ``tp``). Every option is simulated
alone with its inputs in place, and every layout conversion between adjacent
choices is scored from its busiest port or mesh link. Beam search over this
chain proposes the ``keep`` best complete graphs, which are simulated whole.
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
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from math import isqrt
from subprocess import TimeoutExpired
from pathlib import Path
import time

from ..gemm.plan import load
from ..gemm.space import Proxy, structure
from ..methods import METHODS, target
from ..output import variants as output_variants
from ..model.attention import AttentionPlan, KVLayout
from ..model.attention_proxy import AttentionProxy
from ..model.kernels import Gemm
from ..model.transformer import (Workload, build, configuration, decode_configuration, edge_regions, edges,
                                 port, roles, tune_native)
from ..search import beam, diverse, measure_all
from ..sim import WORKERS, dump, fingerprint
from ..tasks import Graph, traffic
from . import attention as standalone_attention

cycles = lambda row: row['result']['e2e_cycles']


class Experiment:
    """Measures each distinct whole-graph configuration once per run."""
    def __init__(self, workload, chip, out):
        self.workload = workload; self.chip = Path(chip).resolve(); self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.provenance = fingerprint(chip); self.rows = []; self.seen = {}

    def measure(self, config, label, record=True, deadline=None):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        if key not in self.seen:
            folder = self.out/'runs'/key
            run = build(self.workload, self.chip, config).simulate(folder, deadline=deadline)
            self.seen[key] = (run['result'], run['graph'], folder)
        result, manifest, folder = self.seen[key]
        row = dict(label=label, key=key, configuration=config, result=result,
                   peak_dram_bytes=max(manifest['peak_dram_bytes']), path=str(folder),
                   layout_noc_bytes=sum(e['noc_data_bytes'] for e in manifest['edges']),
                   layout_write_bytes=sum(e['dram_write_bytes'] for e in manifest['edges']))
        if record:
            self.record(row)
        return row

    def record(self, row):
        self.rows.append(row); dump(self.out/'measurements.json', self.rows)
        print(f'{row["label"]}: {cycles(row)} cycles, peak {row["peak_dram_bytes"]} B/core', flush=True)

    def measure_many(self, configs, label, workers, deadline=None, abort=True):
        """Measure independent graphs in parallel, in order, until ``deadline``;
        with ``abort`` graphs still running at the deadline are dropped."""
        unique = {json.dumps(cfg, sort_keys=True): cfg for cfg in configs}
        rows = measure_all(unique, lambda key: self.measure(unique[key], label, False, deadline if abort else None),
                           workers, deadline=deadline)
        for row in rows:
            self.record(row)
        return rows


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


class Chain:
    """Joint proxy of a graph: standalone operator cycles plus layout-edge scores.

    Each option is simulated once with its inputs already in its own layout;
    each layout edge is scored from its exact per-core traffic.
    """
    def __init__(self, exp, hw, proxy, workers, attention_budget=32, deadline=None):
        self.exp = exp; self.hw = hw; self.proxy = proxy; self.workers = workers; self.deadline = deadline
        self.cycles = {}; self.ports = {}; self.edge_scores = {}
        self.attention_proxy = AttentionProxy(exp.chip); self.attention_budget = attention_budget
        self.attention_report = {}

    def kernel(self, name, entry):
        k = Gemm(self.exp.workload.problems[name], load(entry['plan']), entry['native'], entry['direction'],
                 entry['gated'])
        return k.bind(self.hw['cores'], self.hw['sram'], self.hw['dram'])

    def simulate(self, name, choice, deadline=None):
        w = self.exp.workload
        if name == 'attention':
            return standalone_attention.measure(w.attention, self.exp.chip, self.exp.out/'attention',
                                                choice, placed=True, deadline=deadline)['result']['e2e_cycles']
        key = hashlib.sha256(json.dumps([name, choice], sort_keys=True).encode()).hexdigest()[:20]
        folder = self.exp.out/'operators'/key
        graph = Graph(self.exp.chip); kernel = self.kernel(name, choice).prepare(graph, name)
        regions, tensors = kernel.input_layout(graph.cores)
        for core, t in enumerate(tensors):
            graph.tensor(core, 'x', t['shape'], t['strides'])
        kernel.compile(graph, [replace(r, name='x') for r in regions],
                       rope_width=w.attention.hidden if w.rope and name == 'qkv' else 0)
        return graph.simulate(folder, deadline=deadline)['result']['e2e_cycles']

    def measure(self, jobs, deadline):
        """Standalone cycles of ``(name, choice)`` jobs, in order, until ``deadline``."""
        run = lambda i: self.simulate(*jobs[i], deadline=deadline)
        try:
            rows = measure_all(range(len(jobs)), lambda i: (i, run(i)), self.workers, deadline=deadline)
        except TimeoutExpired:
            rows = []
        for i, value in rows:
            self.cycles[jobs[i][0], token(jobs[i][1])] = value

    def prepare(self, options):
        """Measure every distinct option once; drop illegal and behaviourally equal options."""
        def boundaries(name, choice):
            if name == 'attention':
                return output_variants(choice)
            if choice['native']:
                return [choice]  # upstream checker retains its original collective
            return [dict(choice,plan=asdict(p)) for p in output_variants(load(choice['plan']))]
        options = {name:[c for choice in choices for c in boundaries(name,choice)]
                   for name,choices in options.items()}
        def legal(name, c):
            try:
                return name == 'attention' or self.kernel(name, c) is not None
            except ValueError:
                return False
        options = {name: list({token(c): c for c in choices if legal(name, c)}.values())
                   for name, choices in options.items()}
        if 'attention' in options:
            candidates = options['attention']
            options['attention'] = self.attention_proxy.select(self.exp.workload.attention,candidates,
                                                               self.attention_budget,placed=True)
            self.attention_report = dict(plans=len(candidates),measured=len(options['attention']),
                                         proxy_budget=self.attention_budget)
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
            bound = choice if op == 'attention' or choice is None else self.kernel(op, choice)
            self.ports[key] = port(self.exp.workload, self.hw['cores'], name, bound)
        return self.ports[key]

    def edge(self, src, dst, alias, a, b):
        key = (src, dst, a and token(a), b and token(b))
        if key not in self.edge_scores:
            regions = edge_regions(self.exp.workload, self.hw['cores'], dst, self.port(src, a), self.port(dst, b))
            self.edge_scores[key] = self.proxy.edge(traffic(*regions, alias, isqrt(self.hw['cores'])))
        return self.edge_scores[key]

    def search(self, options, width, keep):
        """The ``keep`` best distinct complete assignments ``{operator: choice}`` by joint proxy."""
        graph_edges = [(s, d, alias, s.split('.')[0], d.split('.')[0]) for s, d, alias in edges(self.exp.workload)]
        def cost(name, index, partial):
            chosen = {**{o: options[o][i] for o, i in partial.items()}, name: options[name][index]}
            pick = lambda o: None if o == 'kv' else chosen.get(o, False)
            value = self.cycles[name, token(chosen[name])]
            # Charge each edge once, when its later endpoint is chosen.
            for src, dst, alias, a, b in graph_edges:
                if name in (a, b) and pick(a) is not False and pick(b) is not False:
                    value += self.edge(src, dst, alias, pick(a), pick(b))
            return value
        ranked = beam([(name, range(len(choices))) for name, choices in options.items()], cost, width, keep)
        return [(score, {name: options[name][i] for name, i in chosen.items()}) for score, chosen in ranked]

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
        """Measure the options alone; the ``keep`` best complete graphs of the chain as configurations."""
        options = self.prepare(options)
        proposals = self.search(options, width, keep)
        configs = [dict(kernels={k: c[k] for k in self.exp.workload.problems}, attention=asdict(c['attention']))
                   for _, c in proposals]
        return options, proposals, configs


ORDER = ('qkv', 'attention', 'output', 'expand', 'contract')


def run(workload, chip, out, method='ours', budget=4, workers=WORKERS, tile_min=8, passes=2, width=64, keep=8, tiles=2,
        refine=3, deadline=None, attention_budget=32):
    exp = Experiment(workload, chip, out); started = time.monotonic(); stop = deadline and started+deadline
    # Weights are preplaced once, like the native TP shards; replicating them
    # would trade unbilled DRAM capacity for speed.
    hw = dict(target(chip, tile_min), single_b=True); layout = KVLayout(workload.attention, hw['cores'])
    # Each method supplies its own fused dataflow on the shared execution backend.
    options = {'attention': METHODS[method].attention(layout, hw['sram'])}
    if not options['attention']:
        raise ValueError('No SRAM-feasible attention plan')
    proxy = Proxy(chip); report = {}
    if METHODS[method].native:
        kernels = tune_native(workload, chip, exp.out/'native_tuning', workers)
        options.update({k: [v] for k, v in configuration(kernels, options['attention'][0])['kernels'].items()})
    else:
        # Graph inputs arrive through layout edges whose cost depends on the
        # neighbours. Rank by the kernel alone and, in the same scan, by kernel
        # plus single-copy distribution, so plans that need no input
        # replication (the ones that make edges cheap) are always proposed too.
        for name, problem in workload.problems.items():
            score = lambda p: (k := proxy.kernel(problem, p), k+proxy.distribution(problem, p))
            bound = lambda q: (k := proxy.lower(problem, q), k+proxy.distribution(problem, q))
            space = lambda skip=None: METHODS[method].space(problem, skip=skip, direction=roles(workload, name)['direction'], **hw)
            chosen, scanned = diverse(space, score, budget, structure, tiles, bound)
            # Simulate in proxy (kernel score) order, so a deadline cuts the tail.
            ranked = sorted((p for p, _ in chosen), key=lambda p: proxy.kernel(problem, p))
            options[name] = [dict(plan=asdict(p), native=False, **roles(workload, name)) for p in ranked]
            report[name] = dict(scanned=scanned, chosen=[asdict(p) for p in ranked])
    chain = Chain(exp, hw, proxy, workers, attention_budget, stop and started+deadline/2)
    options, proposals, configs = chain.propose({k: options[k] for k in ORDER}, width, keep)
    best = min(exp.measure_many(configs, method+'_joint', workers, stop, abort=False), key=cycles)
    best = descend(exp, best, chain.neighbours(options, refine), method, passes, workers, stop)
    dump(exp.out/'search.json', dict(gemm=report, attention=chain.attention_report, measured=chain.measured, joint=[dict(score=score, configuration=cfg)
                                                          for (score, _), cfg in zip(proposals, configs)],
                                      budget=budget, tiles=tiles, passes=passes, width=width, keep=keep, refine=refine))
    summary = dict(experiment='transformer', workload=asdict(workload), chip=str(Path(chip).resolve()), method=method,
                   best=best, cycles=cycles(best), seconds=time.monotonic()-started, deadline=deadline,
                   hit_deadline=bool(stop and time.monotonic() > stop), sha256=exp.provenance)
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
    a = parser.parse_args()
    if min(a.batch, a.context, a.query_heads, a.kv_heads, a.head_dim, a.ffn, a.budget, a.workers,
           a.tile_min, a.width, a.keep, a.tiles) < 1 or min(a.passes, a.refine) < 0:
        parser.error('Counts must be positive')
    workload = Workload(a.batch, a.context if a.mode == 'prefill' else 1, a.context, a.query_heads, a.kv_heads,
                        a.head_dim, a.ffn)
    run(workload, a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.passes, a.width, a.keep, a.tiles, a.refine,
        a.deadline,a.attention_budget)


if __name__ == '__main__':
    main()
