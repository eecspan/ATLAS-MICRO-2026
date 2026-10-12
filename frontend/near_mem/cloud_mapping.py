"""QKV A-panel rotation inside the native N groups, with unchanged graph ABI.

Native QKV uses physical core = K_row * side + N_column. Each N group
already has identical input A replicas, so its initial rotating shard can be
read directly from the existing input tensor. Weights and partial C retain
native homes. The following original K reduction and KV communication remain.
"""
from copy import deepcopy
import json
from math import isqrt, prod
from pathlib import Path

from .backend import hardware, lower
from .compiler import compile_plan
from .plan import DEPENDENCIES, Problem


def ring_program(operator, placement, plan, chip):
    from .cloud_compare import contract
    local, names = contract(operator, placement)
    arch, dram, _ = hardware(chip)
    side = isqrt(arch['core_num'])
    if operator['name'] != 'qkv_proj' or side*side != arch['core_num']:
        raise ValueError('Only the audited square-mesh native qkv_proj contract is supported')
    if plan.ps != (1, side, 1) or plan.b != (1, 1) or plan.c != (1, 1) or not plan.layers:
        raise ValueError('This adapter rotates A only within each native QKV N group')
    problem = Problem(local.m, local.n*side, local.k, local.dtype_bytes)
    program = compile_plan(problem, plan, arch['core']['buffer']['buffer_size']*1024,
                           dram, overlap=True)
    # Convert compact homes to existing native tensor addresses. A is already
    # replicated across N columns; this does not create a free redistribution.
    tensors = {t['name']: t for t in placement['tensor']}
    for phase in program.phases:
        for col, actions in enumerate(phase):
            for action in actions:
                if action['op'] not in ('load', 'store'):
                    continue
                v = action['tensor']; axes = DEPENDENCIES[v]
                base = [index*plan.tile[d] for index, d in zip(action['key'], axes)]
                if v in ('B', 'C'):
                    base[1] -= col*local.n
                shape = tensors[names[v]]['shape']
                if any(x < 0 or x+y > n for x, y, n in zip(base, action['extent'], shape)):
                    raise AssertionError('Rotation changes a tensor home outside the native contract')
                action['base'] = base
    return program, names, arch, side


def ring_replacement(operator, placement, plan, chip, out):
    program, names, arch, side = ring_program(operator, placement, plan, chip)
    out = Path(out).resolve()
    manifest = lower(program, out, arch['noc']['flit_size'])
    templates = [json.loads((out/f'core_{i}.yaml').read_text()) for i in range(side)]
    id_span = manifest['noc_bytes']//arch['noc']['flit_size']
    if id_span*side >= 2**31:
        raise ValueError('Native flit identifiers would overflow')
    for row in range(side):
        for col, template in enumerate(templates):
            core = deepcopy(template)
            for phase in core['execution']:
                for task in phase['dram']:
                    task['name'] = names[task['name']]
                for key in ('noc_tx', 'noc_rx'):
                    for task in phase[key]:
                        task['src'] += row*side; task['dst'] += row*side
                        task['init_flit_id'] += row*id_span
            (out/f'core_{row*side+col}.yaml').write_text(json.dumps(core))
    manifest.update(boundary=names, physical_cores=arch['core_num'], compiled_cores=side,
                    replicas=side, generator='native_accounting_qkv_a_rotation',
                    scope='A rotation within native N groups; original K reduction and KV retained')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return dict(name=operator['name'], type='mpmd-general', iteration=templates[0]['iteration'],
                file_prefix=str(out)), manifest
