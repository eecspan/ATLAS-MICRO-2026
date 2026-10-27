"""One steady-state Transformer layer: the ATLAS native graph against one method.

Every graph is chosen the same way. Each operator has a few options (for the
ATLAS incumbent: the native GEMMs and native fused-attention splits; for a
method: those plus the method's proxy proposals). Every option is simulated
alone with its inputs in place, and every layout conversion between adjacent
choices is scored from its busiest port or mesh link. Beam search over this
chain proposes the ``keep`` best complete graphs, which are simulated whole.
The chain score misses overlap between operators, so a coordinate descent
then simulates, in parallel, every graph that changes one decision to one of
its few best standalone options, keeps the best, and repeats up to
``passes`` times while it improves. Weights are stored once for every method:
an M split must rotate B. Each run searches one method; compare methods by
separate runs.
"""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from math import isqrt
from pathlib import Path
import time

from ..gemm.plan import Plan
from ..gemm.space import Proxy, structure
from ..methods import METHODS, propose, target, union
from ..model.attention import AttentionPlan, KVLayout, plans as attention_plans
from ..model.kernels import Gemm
from ..model.transformer import (Workload, build, configuration, decode_configuration, edge_regions, edges,
                                 port, tune_native)
from ..search import beam, diverse
from ..sim import dump, fingerprint
from ..tasks import Graph, traffic
from . import attention as standalone_attention

cycles = lambda row: row['result']['e2e_cycles']


class Experiment:
    """Measures each distinct whole-graph configuration once per run."""
    def __init__(self, workload, chip, out):
        self.workload = workload; self.chip = Path(chip).resolve(); self.out = Path(out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.provenance = fingerprint(chip); self.rows = []; self.seen = {}

    def measure(self, config, label, record=True):
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        if key not in self.seen:
            folder = self.out/'runs'/key
            run = build(self.workload, self.chip, config).simulate(folder)
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

    def measure_many(self, configs, label, workers):
        """Measure independent neighbors in parallel; each uses its own simulator process."""
        unique = {json.dumps(cfg, sort_keys=True): cfg for cfg in configs}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda cfg: self.measure(cfg, label, record=False), unique.values()))
        for row in rows:
            self.record(row)
        return rows

    def best_of(self, incumbent, configs, label, workers):
        return min([incumbent, *self.measure_many(configs, label, workers)], key=cycles)


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


def descend(exp, start, neighbours, label, passes, workers):
    """Coordinate descent over whole graphs; ``neighbours(best)`` gives each decision's alternatives."""
    best = start
    for sweep in range(passes):
        configs = [cfg for name, choices in neighbours(best).items()
                   for cfg in variants(best['configuration'], set_attention if name == 'attention' else set_kernel(name),
                                       choices)]
        candidate = exp.best_of(best, configs, f'{label}_descent{sweep}', workers)
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
    def __init__(self, exp, hw, proxy, workers):
        self.exp = exp; self.hw = hw; self.proxy = proxy; self.workers = workers
        self.cycles = {}; self.ports = {}; self.edge_scores = {}

    def kernel(self, name, entry):
        k = Gemm(self.exp.workload.problems[name], Plan(**{a: tuple(b) if isinstance(b, list) else b
                                                          for a, b in entry['plan'].items()}),
                 entry['native'], entry['direction'], entry['gated'])
        return k.bind(self.hw['cores'], self.hw['sram'], self.hw['dram'])

    def simulate(self, name, choice):
        w = self.exp.workload
        if name == 'attention':
            return standalone_attention.measure(w.attention, self.exp.chip, self.exp.out/'attention',
                                                choice, placed=True)['result']['e2e_cycles']
        key = hashlib.sha256(json.dumps([name, choice], sort_keys=True).encode()).hexdigest()[:20]
        folder = self.exp.out/'operators'/key
        graph = Graph(self.exp.chip); kernel = self.kernel(name, choice).prepare(graph, name)
        regions, tensors = kernel.input_layout(graph.cores)
        for core, t in enumerate(tensors):
            graph.tensor(core, 'x', t['shape'], t['strides'])
        kernel.compile(graph, [replace(r, name='x') for r in regions],
                       rope_width=w.attention.hidden if w.rope and name == 'qkv' else 0)
        return graph.simulate(folder)['result']['e2e_cycles']

    def prepare(self, options):
        """Measure every distinct option once; drop illegal and behaviourally equal options."""
        def legal(name, c):
            try:
                return name == 'attention' or self.kernel(name, c) is not None
            except ValueError:
                return False
        options = {name: list({token(c): c for c in choices if legal(name, c)}.values())
                   for name, choices in options.items()}
        jobs = list({(name, token(c)): (name, c) for name, choices in options.items() for c in choices
                     if (name, token(c)) not in self.cycles}.items())
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for (key, _), value in zip(jobs, pool.map(lambda job: self.simulate(*job[1]), jobs)):
                self.cycles[key] = value
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


def run(workload, chip, out, method='ours', budget=4, workers=4, tile_min=16, passes=2, width=64, keep=8, tiles=2,
        refine=3):
    exp = Experiment(workload, chip, out); started = time.monotonic()
    kernels = tune_native(workload, chip, exp.out/'native_tuning', workers)
    # Weights are preplaced once, like the native TP shards; replicating them
    # would trade unbilled DRAM capacity for speed.
    hw = dict(target(chip, tile_min), single_b=True); layout = KVLayout(workload.attention, hw['cores'])
    native_attention = list(attention_plans(layout, hw['sram'], False))
    if not native_attention:
        raise ValueError('No SRAM-feasible attention plan')
    native = configuration(kernels, native_attention[0])['kernels']
    proxy = Proxy(chip); chain = Chain(exp, hw, proxy, workers); joint = {}

    def choose(label, options, incumbent=None):
        options, proposals, configs = chain.propose({k: options[k] for k in ORDER}, width, keep)
        rows = exp.measure_many(configs, label+'_joint', workers)
        best = min([*([incumbent] if incumbent else []), *rows], key=cycles)
        joint[label] = [dict(score=score, configuration=cfg) for (score, _), cfg in zip(proposals, configs)]
        return descend(exp, best, chain.neighbours(options, refine), label, passes, workers)

    # The ATLAS incumbent is chosen like any method, over native options only.
    atlas = choose('atlas', {**{k: [native[k]] for k in workload.problems}, 'attention': native_attention})

    # Graph inputs arrive through layout edges whose cost depends on the
    # neighbours. Rank by the kernel alone and, in the same scan, by kernel plus
    # single-copy distribution, so plans that need no input replication (the
    # ones that make edges cheap) are always proposed too.
    report = {}; names = (*METHODS[method].includes, method)
    options = {'attention': list(dict.fromkeys(native_attention+[p for m in names
                                                                 for p in METHODS[m].attention(layout, hw['sram'])]))}
    for name, problem in workload.problems.items():
        score = lambda p: (k := proxy.kernel(problem, p), k+proxy.distribution(problem, p))
        bound = lambda q: (k := proxy.lower(problem, q), k+proxy.distribution(problem, q))
        ranked, scanned = propose(problem, hw, lambda space: diverse(space, score, budget, structure, tiles, bound),
                                  method)
        options[name] = [native[name]]+[dict(native[name], plan=asdict(p), native=False) for p in union(ranked, method)]
        report[name] = dict(scanned=scanned, chosen={k: [asdict(p) for p, _ in v] for k, v in ranked.items()})
    best = choose(method, options, atlas)
    dump(exp.out/'search.json', dict(gemm=report, joint=joint, budget=budget, tiles=tiles, passes=passes, width=width,
                                      keep=keep, refine=refine))
    summary = dict(workload=asdict(workload), chip=str(Path(chip).resolve()), method=method, atlas=atlas, best=best,
                   speedup=cycles(atlas)/cycles(best),
                   seconds=time.monotonic()-started, sha256=exp.provenance)
    dump(exp.out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'speedup', 'seconds')}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chip', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('prefill', 'decode'), default='prefill')
    parser.add_argument('--method', choices=list(METHODS), default='ours')
    for name, value in [('batch', 1), ('context', 128), ('query-heads', 8), ('kv-heads', 2), ('head-dim', 64),
                        ('ffn', 1536), ('budget', 4), ('workers', 4), ('tile-min', 16), ('passes', 2),
                        ('width', 64), ('keep', 8), ('tiles', 2), ('refine', 3)]:
        parser.add_argument('--'+name, type=int, default=value)
    a = parser.parse_args()
    if min(a.batch, a.context, a.query_heads, a.kv_heads, a.head_dim, a.ffn, a.budget, a.workers,
           a.tile_min, a.width, a.keep, a.tiles) < 1 or min(a.passes, a.refine) < 0:
        parser.error('Counts must be positive')
    workload = Workload(a.batch, a.context if a.mode == 'prefill' else 1, a.context, a.query_heads, a.kv_heads,
                        a.head_dim, a.ffn)
    run(workload, a.chip, a.output, a.method, a.budget, a.workers, a.tile_min, a.passes, a.width, a.keep, a.tiles, a.refine)


if __name__ == '__main__':
    main()
