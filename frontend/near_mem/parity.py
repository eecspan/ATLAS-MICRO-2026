"""Check native GEMM and its general-task representation in the same full graph."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml

from frontend.auto.generalize import expand_gemm
from .backend import fingerprint, hardware
from .cloud_compare import dump, execute, freeze_graph
from .run import ROOT


def check(source, out):
    source = source.resolve(); out = out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True)
    chip = Path(yaml.safe_load((source/'system.yaml').read_text())['system']['chip']['config_path']).resolve()
    arch, _, _ = hardware(chip)
    graph, placement, hashes = freeze_graph(source/'native', out/'native_tasks')
    converted = deepcopy(graph)
    for op in converted['operator']:
        if op['type'] != 'gemm':
            continue
        folder = out/'general_tasks'/op['name']; folder.mkdir(parents=True)
        core = expand_gemm(op, placement['tensor'])
        for i in range(arch['core_num']):
            dump(folder/f'core_{i}.yaml', core)
        name = op['name']; op.clear()
        op.update(name=name, type='mpmd-general', iteration=core['iteration'], file_prefix=str(folder))
    provenance = fingerprint(chip)
    for rel in ('frontend/auto/generalize.py', 'frontend/auto/tiling_explorer_cloud.py'):
        p = ROOT/rel; provenance[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump(out/'manifest.json', dict(scope='representation equivalence of the saved native graph; no new search',
                                  source=str(source), chip=str(chip), sha256=provenance,
                                  task_sha256=hashes, libc_rand_seed=1))
    native = execute(chip, out/'native', graph, placement)
    general = execute(chip, out/'general', converted, placement)
    summary = dict(equal=native == general, native=native, general=general)
    dump(out/'summary.json', summary)
    print(json.dumps(dict(equal=summary['equal'], native_cycles=native['e2e_cycles'],
                          general_cycles=general['e2e_cycles']), indent=2), flush=True)
    if not summary['equal']:
        raise AssertionError('Native and general representations differ; inspect per-operator statistics')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    check(args.native, args.output)
