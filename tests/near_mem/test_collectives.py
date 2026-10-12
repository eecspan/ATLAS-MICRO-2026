"""Replay native collective packets and DRAM windows, including gated FFN."""
import json
import math

import numpy as np
import pytest

from frontend.auto.collectives import allreduce_tasks
from frontend.near_mem.run import execute, make_hardware


def build(side, width, gated, direction):
    rows, factor = 3, 2 if gated else 1
    tensors = [dict(name='partial', shape=[rows, factor * width],
                    strides=[factor * width, 1], element_size=2, base_addr=0),
               dict(name='next_input', shape=[rows, width], strides=[width, 1],
                    element_size=2, base_addr=131072)]
    op, tasks = allreduce_tasks(*tensors, 'ffn', rows, 0, factor * width,
                               2, 1, direction, side, 64, [0] * (side * side), gated=gated)
    return tensors, op, tasks


def replay(tensors, op, tasks, partials, gated, side):
    width = tensors[1]['shape'][1]
    chunk_width = math.ceil(width / side)
    buffers = [dict() for _ in tasks]
    output = np.full((len(tasks), *tensors[1]['shape']), np.nan)
    writes = np.zeros(output.shape, dtype=int)
    factor = 2 if gated else 1
    for iteration in range(op['iteration'] + 2):
        for core, task in enumerate(tasks):
            for d in task['communication']['dram']:
                tensor = tensors[int(d['is_write'])]
                assert all(b + e <= s for b, e, s in zip(d['access_base'], d['access_extent'], tensor['shape']))
                if d['init_iter'] != iteration or d['is_write']:
                    continue
                row, col = d['access_base']; nr, nc = d['access_extent']
                chunk, half = (col % width) // chunk_width, col // width
                buffers[core].setdefault(chunk, np.zeros((nr, factor * nc)))
                buffers[core][chunk][:, half * nc:(half + 1) * nc] = partials[core, row:row+nr, col:col+nc]
        if 1 <= iteration <= op['iteration']:
            sends, receives = {}, []
            for core, task in enumerate(tasks):
                phase = task['communication']['on_chip'][iteration - 1]
                for v in phase['rx']['vector']:
                    if v['name'].startswith('gate_chunk'):
                        chunk = int(v['name'].split('chunk')[1])
                        if not v['vec_count']:
                            continue
                        gate, up = np.split(buffers[core][chunk], 2, axis=1)
                        buffers[core][chunk] = gate / (1 + np.exp(-gate)) * up
                for p in phase['tx']['noc']:
                    key = (p['src'], p['dst'], p['init_flit_id'], p['flit_num'])
                    chunk = int(p['name'].split('chunk')[1])
                    sends[key] = buffers[core][chunk].copy()
                for p in phase['rx']['noc']:
                    receives.append((core, p))
            assert len(sends) == len(receives)
            for core, p in receives:
                key = (p['src'], p['dst'], p['init_flit_id'], p['flit_num'])
                chunk = int(p['name'].split('chunk')[1])
                if p['name'].startswith('reduce'):
                    buffers[core][chunk] += sends[key]
                else:
                    buffers[core][chunk] = sends[key]
        for core, task in enumerate(tasks):
            for d in task['communication']['dram']:
                if d['init_iter'] == iteration and d['is_write']:
                    row, col = d['access_base']; nr, nc = d['access_extent']
                    output[core, row:row+nr, col:col+nc] = buffers[core][col // chunk_width]
                    writes[core, row:row+nr, col:col+nc] += 1
    np.testing.assert_array_equal(writes, 1)
    return output


@pytest.mark.parametrize('side', [2, 4])
@pytest.mark.parametrize('width', [1, 16, 17])
@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('direction', [0, 1])
def test_packets_reduce_before_nonlinearity_and_write_exact_replicas(side, width, gated, direction):
    tensors, op, tasks = build(side, width, gated, direction)
    partials = np.random.default_rng(9).normal(size=(side * side, *tensors[0]['shape']))
    actual = replay(tensors, op, tasks, partials, gated, side)
    for line in range(side):
        ids = [rank * side + line if direction == 0 else line * side + rank for rank in range(side)]
        expected = partials[ids].sum(axis=0)
        if gated:
            gate, up = np.split(expected, 2, axis=1)
            expected = gate / (1 + np.exp(-gate)) * up
        for core in ids:
            np.testing.assert_allclose(actual[core], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('gated', [False, True])
def test_collective_executes_in_backend_with_exact_memory_and_vector_work(tmp_path, gated):
    side, width = 2, 128  # Whole DRAM bursts: payload equals transferred bytes.
    tensors, op, tasks = build(side, width, gated, 1)
    directory = tmp_path / 'tasks'; directory.mkdir()
    op['file_prefix'] = str(directory)
    for core, task in enumerate(tasks):
        (directory / f'core_{core}.yaml').write_text(json.dumps(task))
    chip = make_hardware(tmp_path / 'hardware', side * side)
    result = execute(chip, tmp_path / 'run', dict(operator=[op]), dict(tensor=tensors))
    assert result['e2e_cycles'] > 0 and result['noc_cycles'] > 0
    assert result['matrix_cycles'] == 0
    expected_vec = sum(v['vec_count'] for t in tasks for phase in t['communication']['on_chip']
                       for v in phase['rx']['vector'])
    assert result['flop_count'] == expected_vec
    rows, factor = 3, 2 if gated else 1
    assert result['memory_access_bytes'] == side * side * rows * width * (factor + 1) * 2


def test_invalid_gated_shape_is_rejected():
    tensors, _, _ = build(2, 16, True, 0)
    with pytest.raises(ValueError, match='equally sized'):
        allreduce_tasks(*tensors, 'bad', 3, 0, 31, 2, 1, 0, 2, 64, [0]*4, gated=True)
