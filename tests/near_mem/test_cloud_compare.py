"""Boundary checks for substitution into the untouched native decode graph."""
from copy import deepcopy
import json
from itertools import permutations

import numpy as np
import pytest

from frontend.near_mem.cloud_compare import contract, replacement, execute
from frontend.near_mem.compiler import compile_plan
from frontend.near_mem.plan import Problem, Plan
from frontend.near_mem.run import make_hardware


def boundary():
    tensors = [dict(name='input_qkv_proj', shape=[8, 8], strides=[8, 1], base_addr=131072, element_size=2),
               dict(name='weight_qkv_proj_b0', shape=[8, 12], strides=[1, 8], base_addr=262144, element_size=2),
               dict(name='output_qkv_proj', shape=[8, 12], strides=[12, 1], base_addr=393216, element_size=2)]
    op = dict(name='qkv_proj', type='gemm', iteration=12,
              execution=dict(matrix=[dict(mac_count=64)],
                             dram=[dict(name=t['name']) for t in tensors]))
    return op, dict(tensor=tensors)


@pytest.mark.parametrize('loop', [''.join(p) for p in permutations('mnk')])
def test_local_result_is_native_split_k_partial(loop):
    # Native 2x2 mapping: K along rows, N along columns. The following native
    # reduction expects partial outputs, so a replacement must not reduce K twice.
    rng = np.random.default_rng(18)
    a = rng.normal(size=(8, 16)); b = rng.normal(size=(16, 24))
    result = np.zeros((8, 24))
    program = compile_plan(Problem(8, 12, 8), Plan((1, 1, 1), (4, 4, 4), loop=loop), 65536, 1<<28)
    for row in range(2):
        for col in range(2):
            result[:, col*12:(col+1)*12] += program.replay(a[:, row*8:(row+1)*8], b[row*8:(row+1)*8, col*12:(col+1)*12])
    np.testing.assert_allclose(result, a@b, atol=1e-12)


def test_native_addresses_and_names_survive_lowering(tmp_path):
    chip = make_hardware(tmp_path/'hardware', 4)
    op, placement = boundary(); before = deepcopy((op, placement))
    repl, manifest = replacement(op, placement, Plan((1, 1, 1), (4, 4, 4)), chip, tmp_path/'kernel')
    assert (op, placement) == before
    tensors = {t['name']: t for t in placement['tensor']}
    for core in range(4):
        tasks = json.loads((tmp_path/f'kernel/core_{core}.yaml').read_text())
        for phase in tasks['execution']:
            assert not phase['noc_tx'] and not phase['noc_rx']
            for task in phase['dram']:
                tensor = tensors[task['name']]
                assert all(0 <= base and base+extent <= size for base, extent, size in
                           zip(task['access_base'], task['access_extent'], tensor['shape']))
    result = execute(chip, tmp_path/'run', dict(operator=[repl]), placement)
    assert result['flop_count'] == 4*(2*manifest['macs']+manifest['vector_ops'])
    assert result['noc_cycles'] == 0
    assert json.loads((tmp_path/'run/placement.yaml').read_text()) == placement


def test_reject_incompatible_boundary():
    op, placement = boundary()
    placement['tensor'][1]['strides'] = [12, 1]
    with pytest.raises(ValueError, match='column-major'):
        contract(op, placement)
    op, placement = boundary()
    op['iteration'] = 1
    with pytest.raises(ValueError, match='shape'):
        contract(op, placement)
