"""Replay a native cloud graph with ABI-compatible local GEMM replacements.

Local loop scheduling and QKV A rotation retain native tensor addresses and
collectives. Other global remappings still need explicit layout adapters. Attention/KV tasks are copied unchanged from the complete auto compiler.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import prod, isqrt
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import yaml

from .backend import fingerprint, hardware
from .plan import Plan, Problem
from frontend.auto.gemm_schedule import native_sram_bytes, schedule_gemm
from .run import ROOT, dump, execute, worker
from .search import candidates, Proxy, shortlist

def contract(operator, placement):
    """Infer and check the upstream GEMM boundary, without relabeling FFN ops."""
    if operator['type'] != 'gemm':
        raise ValueError('Expected a native GEMM')
    name = operator['name']
    tensors = {t['name']: t for t in placement['tensor']}
    names = dict(A=f'input_{name}', B=f'weight_{name}_b0', C=f'output_{name}')
    a, b, c = (tensors[names[v]] for v in 'ABC')
    m, k = a['shape']; bk, n = b['shape']
    if bk != k or c['shape'] != [m, n]:
        raise ValueError('Only the native unfactored single-weight GEMM boundary is supported')
    if any(t['element_size'] != 2 for t in (a, b, c)):
        raise ValueError('Expected native FP16 tensors')
    if a['strides'] != [k, 1] or b['strides'] != [1, k] or c['strides'] != [n, 1]:
        raise ValueError('Boundary must use row-major A/C and column-major B')
    if {d['name'] for d in operator['execution']['dram']} != set(names.values()):
        raise ValueError('Unexpected extra weight or DRAM operand')
    macs = sum(x['mac_count'] for x in operator['execution']['matrix'])*operator['iteration']
    if macs != m*n*k:
        raise ValueError('Native kernel does not match the inferred GEMM shape')
    return Problem(m, n, k), names


def replacement(operator, placement, plan, chip, out):
    if plan.cores > 1:
        from .cloud_mapping import ring_replacement
        return ring_replacement(operator, placement, plan, chip, out)
    problem, names = contract(operator, placement)
    if plan.ps != (1, 1, 1) or plan.layers:
        raise ValueError('Local replacement must preserve the native per-core ownership')
    arch, dram, _ = hardware(chip)
    plan.validate(problem)
    capacity = arch['core']['buffer']['buffer_size']*1024
    tm, tn, tk = plan.tile
    native_sram = native_sram_bytes(plan.tile, problem.dtype_bytes)
    if native_sram > capacity:
        raise ValueError('Native double-buffer SRAM capacity exceeded')
    out.mkdir(parents=True, exist_ok=True)
    if plan.loop == 'mnk':
        from frontend.auto.tiling_explorer_cloud import make_gemm_description
        from frontend.auto.generalize import expand_gemm
        # Compile every eligible MNK tile through the SAME native generator,
        # not just a saved winner as an after-the-fact fallback. Expansion has
        # native/general full-graph parity tests and preserves native overlap.
        shape = SimpleNamespace(name=operator['name'], M=problem.m, K=problem.k, N=problem.n)
        native = make_gemm_description(shape, problem.dtype_bytes, (tm, tk, tn),
                                       names['A'], names['C'], names['B'])
        core = expand_gemm(native, placement['tensor'])
        manifest = dict(problem=asdict(problem), plan=asdict(plan), sram_bytes=native_sram,
                        macs=prod(problem.shape), vector_ops=prod(problem.shape)//tk,
                        generator='shared_native_mnk')
    else:
        core, _ = schedule_gemm(problem.shape, plan.tile, plan.loop,
                                (names['A'], names['B'], names['C']), problem.dtype_bytes)
        manifest = dict(problem=asdict(problem), plan=asdict(plan), sram_bytes=native_sram,
                        macs=prod(problem.shape), vector_ops=prod(problem.shape)//tk,
                        generator='shared_native_pipeline')
    for phase in core['execution']:
        if phase['noc_tx'] or phase['noc_rx']:
            raise AssertionError('Local replacement cannot change communication')
        for task in phase['dram']:
            task['name'] = names.get(task['name'], task['name'])
    payload = json.dumps(core)
    for i in range(arch['core_num']):
        (out/f'core_{i}.yaml').write_text(payload)
    # Keep the original full placement, including every tensor base address.
    manifest.update(boundary=names, physical_cores=arch['core_num'], compiled_cores=1,
                    replicas=arch['core_num'],
                    scope='native spatial ownership; local schedule only')
    dump(out/'manifest.json', manifest)
    return dict(name=operator['name'], type='mpmd-general', iteration=core['iteration'],
                file_prefix=str(out)), manifest


def trial(args):
    operator, placement, plan, chip, out = args
    repl, manifest = replacement(operator, placement, plan, chip, out)
    result = execute(chip, out, dict(operator=[repl]), placement)
    expected = (2*manifest['macs']+manifest['vector_ops'])*manifest['replicas']
    if result['flop_count'] != expected:
        raise AssertionError('Replacement arithmetic count mismatch')
    return dict(plan=asdict(plan), operator=repl, result=result, path=str(out))


def freeze_graph(native, out):
    """Copy every referenced task, so the experiment is self-contained."""
    graph = yaml.safe_load((native/'operator_description.yaml').read_text())
    placement = yaml.safe_load((native/'data_placement.yaml').read_text())
    hashes = {}
    for path in (native/'operator_description.yaml', native/'data_placement.yaml'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for op in graph['operator']:
        if 'file_prefix' in op:
            src = Path(op['file_prefix']); dst = out/op['name']
            dst.mkdir(parents=True)
            for path in sorted(src.glob('core_*.yaml')):
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                shutil.copyfile(path, dst/path.name)
            op['file_prefix'] = str(dst)
        if 'attention_input' in op:
            src = Path(op['attention_input']); dst = out/'attn_input.yaml'
            hashes[str(src)] = hashlib.sha256(src.read_bytes()).hexdigest()
            shutil.copyfile(src, dst); op['attention_input'] = str(dst)
    return graph, placement, hashes


def compare(args):
    source = args.native.resolve(); out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True)
    source_manifest = json.loads((source/'manifest.json').read_text())
    # Fail closed if the saved full-compiler baseline no longer matches its sources.
    for path, digest in source_manifest['sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Baseline input changed; regenerate native cloud: {path}')
    chip = Path(yaml.safe_load((source/'system.yaml').read_text())['system']['chip']['config_path']).resolve()
    current_backend=fingerprint(chip)
    for key,digest in source_manifest.get('backend',{}).items():
        if key.startswith('binary:') and current_backend.get(key)!=digest:
            raise ValueError('Native tuning used another backend binary; regenerate the baseline')
    arch, dram, _ = hardware(chip)
    graph, placement, task_hashes = freeze_graph(source/'native', out/'tasks')
    manifest = dict(native=str(source), chip=str(chip), native_manifest=source_manifest,
                    scope=(source_manifest.get('evaluation_scope', 'complete native decode graph') +
                           ('; local scheduling' if args.mode == 'local' else
                            '; QKV local scheduling and A rotation within native N groups')),
                    unchanged='hardware, placements, pre-existing native kernels, attention, KV, collectives',
                    packing=False, budget=args.budget, libc_rand_seed=1,
                    shared_fixes='A/C address strides count accesses; shared native GEMM component work',
                    execution='each measurement in a fresh process; BookSim config unchanged',
                    sha256=current_backend, task_sha256=task_hashes)
    dump(out/'manifest.json', manifest)
    start = time.monotonic()
    baseline = execute(chip, out/'baseline', graph, placement)
    recorded = json.loads((source/'result.json').read_text())['layer_cycles']
    repeated = execute(chip, out/'baseline_repeat', graph, placement)
    if repeated != baseline:
        raise AssertionError('Identical native graph is not reproducible in fresh workers')
    dump(out/'baseline_replay.json', dict(compiler_process_cycles=recorded,
         fresh_process_cycles=baseline['e2e_cycles'], repeat_exact=True,
         note='Native MCRouter uses process-global libc rand(); previous compiler tuning consumes it. '
              'Compare both graphs in fresh workers with seed 1, not against compiler-process timing.'))
    joint = deepcopy(graph); selected = deepcopy(graph); searches = []; controls = {}
    for index, op in enumerate(graph['operator']):
        if op['type'] != 'gemm' or args.mode == 'qkv_ring' and op['name'] != 'qkv_proj':
            continue
        problem, _ = contract(op, placement)
        domain = candidates(problem, 1, arch['core']['buffer']['buffer_size']*1024,
                            tile_min=8, tile_policy='all', dram_capacity=dram)
        domain = (p for p in domain if native_sram_bytes(p.tile) <= arch['core']['buffer']['buffer_size']*1024)
        proposed, count = shortlist(problem, domain, Proxy(chip), args.budget)
        ring_count = 0
        if args.mode == 'qkv_ring':
            side = isqrt(arch['core_num'])
            ring_problem = Problem(problem.m, problem.n*side, problem.k)
            ring_domain = (p for p in candidates(ring_problem, side, arch['core']['buffer']['buffer_size']*1024,
                           tile_min=8, tile_policy='all', dram_capacity=dram)
                           if p.ps == (1, side, 1) and p.layers)
            ring_proposed, ring_count = shortlist(ring_problem, ring_domain, Proxy(chip), args.budget)
            proposed += ring_proposed
        # Always include the native tiling in all six orders as a useful anchor.
        extents = {d['name']: d['access_extent'] for d in op['execution']['dram']}
        tm, tk = extents[f'input_{op["name"]}']; _, tn = extents[f'weight_{op["name"]}_b0']
        plans = list(dict.fromkeys([p for p, _ in proposed] +
                     [Plan((1, 1, 1), (tm, tn, tk), loop=loop)
                      for loop in ('mnk', 'mkn', 'nmk', 'nkm', 'kmn', 'knm')]))
        folder = out/'tuning'/op['name']; folder.mkdir(parents=True)
        native_result = execute(chip, folder/'native', dict(operator=[op]), placement)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [(op, placement, p, chip, folder/f'candidate_{i:03d}') for i, p in enumerate(plans)]
            for row in pool.map(trial, jobs):
                rows.append(row); dump(folder/'results.json', rows)
        best = min(rows, key=lambda r: r['result']['e2e_cycles'])
        joint['operator'][index] = best['operator']
        use_joint = best['result']['e2e_cycles'] < native_result['e2e_cycles']
        if use_joint:
            selected['operator'][index] = best['operator']
        report = dict(name=op['name'], problem=asdict(problem), generated=count, ring_generated=ring_count,
                      evaluated=len(rows), native=native_result, best=best, selected_joint=use_joint)
        if args.mode == 'qkv_ring':
            for label, group in (('local_control', [r for r in rows if prod(r['plan']['ps']) == 1]),
                                 ('rotation', [r for r in rows if prod(r['plan']['ps']) > 1])):
                if not group:
                    continue
                choice = min(group, key=lambda r: r['result']['e2e_cycles'])
                controls[label] = deepcopy(graph)
                controls[label]['operator'][index] = choice['operator']
                report[label] = choice
        searches.append(report); dump(out/'searches.json', searches)
        print(op['name'], 'native', native_result['e2e_cycles'], 'joint', best['result']['e2e_cycles'], flush=True)
    # Identical graph/config/seed need only one execution. Record exact reuse;
    # different schedules are always simulated, including losing rotations.
    measured = {json.dumps(graph, sort_keys=True): ('baseline', baseline)}
    variants = {}
    for label, variant in dict(joint=joint, selected=selected, **controls).items():
        key = json.dumps(variant, sort_keys=True)
        if key in measured:
            previous, result = measured[key]
            folder = out/label; folder.mkdir()
            dump(folder/'operators.yaml', variant); dump(folder/'placement.yaml', placement)
            dump(folder/'result.json', result); dump(folder/'reuse.json', dict(identical_graph=previous))
        else:
            result = execute(chip, out/label, variant, placement)
            measured[key] = (label, result)
        variants[label] = result
    raw_result, selected_result = variants['joint'], variants['selected']
    # Standalone ranking does not guarantee graph ranking: DRAM state persists.
    winner = 'selected' if selected_result['e2e_cycles'] < baseline['e2e_cycles'] else 'baseline'
    summary = dict(scope=manifest['scope'], baseline=baseline, joint=raw_result,
                   selected=selected_result, winner=winner,
                   raw_speedup=baseline['e2e_cycles']/raw_result['e2e_cycles'],
                   selected_speedup=baseline['e2e_cycles']/selected_result['e2e_cycles'],
                   retained_speedup=max(1, baseline['e2e_cycles']/selected_result['e2e_cycles']),
                   wall_seconds=time.monotonic()-start)
    for label in controls:
        summary[label] = variants[label]
        summary[label+'_speedup'] = baseline['e2e_cycles']/variants[label]['e2e_cycles']
    dump(out/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if not isinstance(v, dict)}, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=['local', 'qkv_ring'], default='local')
    ap.add_argument('--budget', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--chip', type=Path)
    args = ap.parse_args()
    if args.worker:
        worker(args.chip, args.output)
    else:
        if not args.native or min(args.budget, args.workers) <= 0:
            ap.error('--native and positive budget/workers required')
        compare(args)


if __name__ == '__main__':
    main()
