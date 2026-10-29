"""Speedups from separate runs: every method's cycles over the TP baseline's.

Each ``experiments.gemm``/``attention``/``transformer`` run writes one method's
``summary.json``. Runs are matched by experiment, problem/workload and chip;
the ``tp`` run of a match is the denominator. GEMM rows also report the
Roofline ideal. Nothing is searched or simulated here.
"""
import argparse
import json
from pathlib import Path

from ..sim import dump


def chip_name(path):
    path = Path(path)
    return path.parent.name if path.stem == 'chip' else path.stem


def case(s):
    """Identity of the compared problem: experiment, shape/workload, chip."""
    shape = s.get('problem') or s['workload']
    return s['experiment'], json.dumps(shape, sort_keys=True), chip_name(s['chip'])


def runs(folders):
    """``{case: {method: summary}}`` of every summary under ``folders``."""
    table = {}
    for folder in folders:
        for path in sorted(Path(folder).glob('**/summary.json')):
            s = json.loads(path.read_text())
            if 'experiment' not in s:
                continue  # a nested tuning summary
            s['path'] = str(path.parent)
            if table.setdefault(case(s), {}).setdefault(s['method'], s) is not s:
                raise ValueError(f'Two {s["method"]} runs of one case: {path.parent}')
    return table


def compare(folders):
    rows = []
    for (experiment, shape, chip), methods in sorted(runs(folders).items()):
        base = methods.get('tp')
        row = dict(experiment=experiment, case=json.loads(shape), chip=chip,
                   cycles={m: s['cycles'] for m, s in sorted(methods.items())})
        if base:
            row['speedup'] = {m: base['cycles']/s['cycles'] for m, s in sorted(methods.items())}
            if 'roofline' in base:
                row['speedup']['roofline'] = base['cycles']/base['roofline']
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folders', type=Path, nargs='+', help='result folders, searched recursively')
    ap.add_argument('--output', type=Path, help='optional JSON file')
    a = ap.parse_args(); rows = compare(a.folders)
    if a.output:
        dump(a.output, rows)
    for r in rows:
        values = r.get('speedup', r['cycles'])
        print(r['experiment'], r['chip'], json.dumps(r['case'], separators=(',', ':')),
              ' '.join(f'{m}={v:.3f}' if isinstance(v, float) else f'{m}={v}' for m, v in values.items()))


if __name__ == '__main__':
    main()
