"""Motivation data: where the time of simple (native ATLAS) compilation goes.

Reads finished ``experiments.gemm``/``attention``/``transformer`` outputs and
reports, for the ATLAS incumbent and for ours:

* input replication: its cycles and the DRAM bytes every replica occupies;
* resource idleness: the share of the critical path during which the matrix
  unit or the NoC is busy (per operator, ATLAS's busy counters of its busiest
  core, capped by the operator's span);
* the time per operator category (GEMM, split-K reduce, relayout, attention).

No search or simulation is run here; every number comes from saved results.
"""
import argparse
from collections import defaultdict
import json
from math import prod
from pathlib import Path

from ..sim import dump


def chip_name(path):
    path = Path(path)
    return path.parent.name if path.stem == 'chip' else path.stem


def replicated_bytes(problem, plan):
    """DRAM bytes of A and B including every replica, over their single-copy size."""
    m, n, k = problem['m'], problem['n'], problem['k']; ps = plan['ps']
    copies_a = ps[1]//prod(plan['a']); copies_b = ps[0]//prod(plan['b'])
    return (m*k*copies_a+k*n*copies_b)/(m*k+k*n)


def breakdown(total, ops):
    """``ops``: ``(category, span, matrix_busy, noc_busy)`` per operator."""
    by = defaultdict(int); mac = noc = idle = 0
    for kind, span, matrix, network in ops:
        by[kind] += span; mac += min(matrix, span); noc += min(network, span)
        idle += span if matrix == 0 and network > 0 else 0
    return dict(total=total, categories=dict(by), mac_busy=mac/total, noc_busy=noc/total, noc_only=idle/total)


def operators(result, kinds=None):
    kinds = kinds or {}
    return [(kinds.get(name, name), o.get('span', o['e2e_cycles']), o['matrix_cycles'], o['noc_cycles'])
            for name, o in result['operators'].items()]


def gemm(folder):
    s = json.loads((folder/'summary.json').read_text())
    row = dict(shape=[s['problem'][x] for x in 'mnk'], chip=chip_name(s['chip']))
    for name, key in (('atlas', 'atlas'), ('ours', 'best')):
        r = s[key]['result']; kernel = r['kernel_cycles']
        row[name] = dict(total=r['e2e_cycles'], kernel=kernel, replication=r['distribution_cycles'],
                         replication_share=r['distribution_cycles']/r['e2e_cycles'],
                         mac_busy_in_kernel=min(r['matrix_cycles'], kernel)/kernel,
                         noc_busy_in_kernel=min(r['noc_cycles'], kernel)/kernel,
                         input_footprint=replicated_bytes(s['problem'], s[key]['plan']),
                         plan={k: s[key]['plan'][k] for k in ('ps', 'tile', 'a', 'b', 'c', 'loop')})
    row['speedup'] = s['speedup']
    return row


def attention(folder):
    s = json.loads((folder/'summary.json').read_text()); row = dict(problem=s['problem'], speedup=s['speedup'])
    for name, key in (('atlas', 'atlas'), ('ours', 'best')):
        r = s[key]['result']
        row[name] = dict(plan=s[key]['plan'], **breakdown(r['e2e_cycles'], operators(r)))
    return row


def transformer(folder):
    s = json.loads((folder/'summary.json').read_text())
    row = dict(workload=s['workload'], chip=chip_name(s['chip']), speedup=s['speedup'])
    for name, key in (('atlas', 'atlas'), ('ours', 'best')):
        path = Path(s[key]['path']); r = json.loads((path/'result.json').read_text())
        kinds = {o['name']: o['metadata'].get('category', 'other')
                 for o in json.loads((path/'graph.json').read_text())['operators']}
        row[name] = dict(breakdown(r['e2e_cycles'], operators(r, kinds)), layout_noc_bytes=s[key]['layout_noc_bytes'],
                         peak_dram_bytes=s[key]['peak_dram_bytes'])
    return row


def plot(report, out):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if report['gemm']:
        rows = report['gemm']; fig, ax = plt.subplots(figsize=(1.3*len(rows)+2, 3))
        for i, r in enumerate(rows):
            base = r['atlas']['total']
            for j, name in enumerate(('atlas', 'ours')):
                x = i+0.38*(j-0.5)
                ax.bar(x, r[name]['kernel']/base, 0.36, color=('C0', 'C2')[j], label=f'{name} kernel' if i == 0 else None)
                ax.bar(x, r[name]['replication']/base, 0.36, bottom=r[name]['kernel']/base, color='C3',
                       label='input replication' if i == j == 0 else None)
        ax.set_xticks(range(len(rows)), ['%s\n%d×%d×%d' % (r['chip'], *r['shape']) for r in rows], fontsize=7)
        ax.set_ylabel('cycles / ATLAS total'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out/'gemm_breakdown.png', dpi=200); plt.close(fig)
    if report['transformer']:
        rows = report['transformer']; labels = []
        kinds = sorted({k for r in rows for n in ('atlas', 'ours') for k in r[n]['categories']})
        fig, ax = plt.subplots(figsize=(1.6*len(rows)+3, 3))
        for i, r in enumerate(rows):
            for j, name in enumerate(('atlas', 'ours')):
                bottom = 0
                for c, kind in enumerate(kinds):
                    value = r[name]['categories'].get(kind, 0)/r['atlas']['total']
                    ax.bar(2*i+j, value, 0.7, bottom=bottom, color=f'C{c}', label=kind if 2*i+j == 0 else None)
                    bottom += value
                w = r['workload']; labels.append(f"{r['chip']}\nB{w['batch']} q{w['queries']}\n{name}")
        ax.set_xticks(range(len(labels)), labels, fontsize=7); ax.set_ylabel('critical path / ATLAS total')
        ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(out/'transformer_breakdown.png', dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('gemm', 'attention', 'transformer'):
        ap.add_argument('--'+name, type=Path, nargs='*', default=[], help=f'experiments.{name} output folders')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    report = dict(gemm=[gemm(f) for f in a.gemm], attention=[attention(f) for f in a.attention],
                  transformer=[transformer(f) for f in a.transformer])
    dump(a.output/'motivation.json', report); plot(report, a.output)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
