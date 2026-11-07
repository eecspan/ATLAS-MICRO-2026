"""Single fused attention: each method evaluates its own attention space.

Q is stored once, row-sharded over all cores; KV uses the common static
head-partitioned layout. Moving or replicating Q to the cores a plan needs is
part of the measured graph, as is writing the output. A task-based proxy
shortlists candidates before simulation; --budget 0 validates exhaustively.
Each method supplies its own fused dataflow over the shared static KV layout.
Each run writes only its own cycles; ``experiments.compare`` turns separate
runs into speedups.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from ..methods import METHODS
from ..model.attention import Attention, KVLayout, build_attention
from ..model.attention_proxy import AttentionProxy
from ..search import measure_all
from ..sim import WORKERS, dump, hardware

cycles = lambda row: row['result']['e2e_cycles']


def measure(problem, chip, out, plan, placed=False, deadline=None):
    """Simulate one plan. Q starts row-sharded, or, if ``placed``, already where the plan reads it."""
    graph = build_attention(problem,chip,plan,placed)
    name = '_'.join(map(str, asdict(plan).values()))+('_placed' if placed else '')
    return dict(plan=asdict(plan), result=graph.simulate(Path(out)/name, deadline=deadline)['result'])


def run(problem, chip, out, method='ours', workers=WORKERS, budget=32):
    chip = Path(chip).resolve(); out = Path(out).resolve(); started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    arch, _, _ = hardware(chip); sram = arch['core']['buffer']['buffer_size']*1024
    layout = KVLayout(problem, arch['core_num'])
    # No method inherits another method's attention candidates implicitly.
    space = METHODS[method].attention(layout, sram)
    if not space:
        raise ValueError(f'No SRAM-feasible fused attention plan for {method}')
    proxy = AttentionProxy(chip)
    candidates = proxy.select(problem,space,budget)
    rows = measure_all(candidates, lambda p: measure(problem, chip, out/'runs', p), workers)
    dump(out/'results.json', rows)
    selected = min(rows, key=cycles)
    summary = dict(experiment='attention', problem=asdict(problem), chip=str(chip), method=method, best=selected,
                   cycles=cycles(selected), plans=len(space), measured=len(candidates), proxy_budget=budget,
                   seconds=time.monotonic()-started)
    dump(out/'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('method', 'cycles', 'plans', 'measured', 'proxy_budget')}, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name, value in [('batch', 1), ('queries', 128), ('context', 128), ('query-heads', 8),
                        ('kv-heads', 2), ('head-dim', 64), ('workers', WORKERS), ('budget', 32)]:
        ap.add_argument('--'+name, type=int, default=value)
    ap.add_argument('--chip', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--method', choices=list(METHODS), default='ours')
    a = ap.parse_args()
    problem = Attention(a.batch, a.queries, a.context, a.query_heads, a.kv_heads, a.head_dim)
    run(problem, a.chip, a.output, a.method, a.workers, a.budget)


if __name__ == '__main__':
    main()
