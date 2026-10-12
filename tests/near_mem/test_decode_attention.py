"""Attention request ownership must survive lowering into DRAM addresses."""
import json
from types import SimpleNamespace

import pytest

from frontend.auto.system import _build_dattn_input_payload
from frontend.auto.tiling_explorer_cloud import generate_attention_comp, generate_kv_proj_comm, generate_q_proj_comm
from frontend.model_parser import Operator, OperatorType
from frontend.near_mem.run import execute, make_hardware
from frontend.util import generate_context_slot_mapping


@pytest.mark.parametrize('contexts, cores, block', [([5], 4, 1), ([1001, 513, 7], 16, 16),
                                                   ([17]*3, 16, 1), ([256]*16, 16, 16)])
def test_context_partition_has_exact_token_counts_and_disjoint_slots(contexts, cores, block):
    cloud = SimpleNamespace(chip_config=SimpleNamespace(core_num=cores), block_size=block)
    mapping = generate_context_slot_mapping(cloud, contexts)
    for request, length in enumerate(contexts):
        assert sum(len(core.get(request, [])) for core in mapping) == length
    for core in mapping:
        slots = [slot for request_slots in core.values() for slot in request_slots]
        assert len(slots) == len(set(slots))
        assert sorted(slots) == list(range(len(slots)))


@pytest.mark.parametrize('requests, expected_bytes', [([0, 1], 512), ([0, 4], 768), ([4, 0], 768)])
def test_noncontiguous_request_rows_reach_actual_dram_backend(tmp_path, requests, expected_bytes):
    tensors = [dict(name='Q', shape=[5, 16], strides=[16, 1], element_size=2, base_addr=0),
               dict(name='O', shape=[5, 16], strides=[16, 1], element_size=2, base_addr=131072),
               dict(name='kv_cache', shape=[8, 32], strides=[32, 1], element_size=2, base_addr=262144)]
    operator = Operator(name='attention', op_type=OperatorType.ATTENTION, kv_head_num=1,
                        kv_group_num=1, head_dim=16, v_head_dim=16, is_mla=False)
    cloud = SimpleNamespace(chip_config=SimpleNamespace(buffer_config=SimpleNamespace(buffer_size=128)),
                            block_size=1)
    generated = generate_attention_comp(tensors[0], tensors[1], [tensors[2]], operator,
                                         2, [[[1]]], cloud)
    mapping = [{request: [slot] for slot, request in enumerate(requests)}, {}, {}, {}]
    payload = _build_dattn_input_payload(generated['attention_input_template'], mapping, 4)
    attn_file = tmp_path / 'attention_input.yaml'
    attn_file.write_text(json.dumps(dict(attention_input=payload)))
    op = generated['attention_comp_description']; op['attention_input'] = str(attn_file)
    result = execute(make_hardware(tmp_path/'hardware', 4), tmp_path/'run',
                     dict(operator=[op]), dict(tensor=tensors))
    # Q/O rows 0,1 share a 128-byte burst; rows 0,4 occupy different bursts.
    # The two explicitly addressed KV vectors add two more bursts in each case.
    assert result['memory_access_bytes'] == expected_bytes
    assert result['flop_count'] == 2 * (2 * (16 + 16) + 9 + 3 * 16)


@pytest.mark.parametrize('side', [2, 4])
def test_q_distribution_covers_each_global_query_row_once(tmp_path, side):
    rows, q_length = 2, 16
    source = dict(name='qkv_partial', shape=[rows, 32], strides=[32, 1], element_size=2, base_addr=0)
    target = dict(name='Q', shape=[rows, side*q_length], strides=[side*q_length, 1],
                  element_size=2, base_addr=131072)
    chip = make_hardware(tmp_path/'hardware', side*side)
    result = generate_q_proj_comm(source, target, 'qkv', rows, 0, q_length, 2, 1, 0,
                                  str(chip), side*side, side, 64, [0]*(side*side), str(tmp_path/'q'))
    for core in range(side*side):
        writes = [0] * target['shape'][1]
        for _, per_core in result[1]:
            for task in per_core[core]['communication']['dram']:
                if task['is_write']:
                    col = task['access_base'][1]; length = task['access_extent'][1]
                    assert col + length <= len(writes)
                    for i in range(col, col+length): writes[i] += 1
        assert writes == [1] * len(writes)


@pytest.mark.parametrize('side', [2, 4])
def test_kv_append_commits_even_when_final_scatter_has_zero_hops(tmp_path, side):
    source = dict(name='qkv_partial', shape=[2, 16+32//side], strides=[16+32//side, 1],
                  element_size=2, base_addr=0)
    target = dict(name='kv_cache', shape=[16, 32], strides=[32, 1], element_size=2, base_addr=131072)
    destinations = [[[] for _ in range(side)] for _ in range(side)]
    destinations[0][0] = [5]; destinations[-1][-1] = [7]
    chip = make_hardware(tmp_path/'hardware', side*side)
    result = generate_kv_proj_comm(source, target, 'qkv', 2, 16, 32//side, 16, 1, 2, 1,
                                   str(chip), side*side, side, 64, [0]*(side*side),
                                   destinations, str(tmp_path/'kv'))
    _, tasks, _ = result[1][0]
    writes = [(core, t['access_base'], t['access_extent']) for core, description in enumerate(tasks)
              for t in description['communication']['dram'] if t['is_write']]
    assert writes == [(0, [5, 0], [1, 32]), (side*side-1, [7, 0], [1, 32])]
    assert result[2] > 0  # The generated collective completed in the real backend.
