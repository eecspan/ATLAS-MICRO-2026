"""Single fused attention: native ATLAS splits against each method's plans.

Q is stored once, row-sharded over all cores; KV uses the common static
head-partitioned layout. Moving or replicating Q to the cores a plan needs is
part of the measured graph, as is writing the output. Attention plans are few,
so every one is simulated; there is no proxy stage.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from ..methods import METHODS
from ..model.attention import Attention, AttentionPlan, KVLayout, compile_attention, endpoints, plans
from ..model.transformer import row_layout
from ..search import measure_all
from ..sim import dump, hardware
from ..tasks import Graph

cycles = lambda row: row['result']['e2e_cycles']


def measure(problem, chip, out, plan, placed=False):
    """Simulate one plan. Q starts row-sharded, or, if ``placed``, already where the plan reads it."""
    graph = Graph(chip); layout = KVLayout(problem, graph.cores)
    cache = layout.allocate(graph)
    if placed:
        query = []
        for i, r in enumerate(endpoints(layout, plan)[0]):
            graph.tensor(r.core, f'q{i}', [r.rows, r.cols]); query.append(replace(r, name=f'q{i}'))
    else:
        query = row_layout(graph, 'query', problem.batch*problem.queries, problem.hidden)
    compile_attention(graph, layout, plan, query, cache)
    name = '_'.join(map(str, asdict(plan).values()))+('_placed' if placed else '')
    return dict(plan=asdict(plan), result=graph.simulate(Path(out)/name)['result'])


def run(problem, chip, out, workers=4):
    chip = Path(chip).resolve(); out = Path(out).resolve(); started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    arch, _, _ = hardware(chip); sram = arch['core']['buffer']['buffer_size']*1024
    layout = KVLayout(problem, arch['core_num'])
    spaces = {'atlas': list(plans(layout, sram, False))}
    for name, method in METHODS.items():
        inherited = [p for m in method.includes for p in spaces[m]]
        spaces[name] = list(dict.fromkeys(spaces['atlas']+inherited+method.attention(layout, sram)))
    rows = measure_all(spaces['ours'], lambda p: measure(problem, chip, out/'runs', p), workers)
    dump(out/'results.json', rows)
    winners = {k: min((r for r in rows if AttentionPlan(**r['plan']) in v), key=cycles) for k, v in spaces.items()}
    summary = dict(problem=asdict(problem), chip=str(chip), **winners,
                   speedup={k: cycles(winners['atlas'])/cycles(v) for k, v in winners.items() if k != 'atlas'},
                   plans={k: len(v) for k, v in spaces.items()}, seconds=time.monotonic()-started)
    dump(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('speedup', 'plans')}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name, value in [('batch', 1), ('queries', 128), ('context', 128), ('query-heads', 8),
                        ('kv-heads', 2), ('head-dim', 64), ('workers', 4)]:
        ap.add_argument('--'+name, type=int, default=value)
    ap.add_argument('--chip', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    problem = Attention(a.batch, a.queries, a.context, a.query_heads, a.kv_heads, a.head_dim)
    run(problem, a.chip, a.output, a.workers)


if __name__ == '__main__':
    main()
