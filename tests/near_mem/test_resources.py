"""Resource counts come from the simulated payloads, by one rule for every method."""
import pytest

from frontend.near_mem.gemm.compiler import compile_plan
from frontend.near_mem.gemm.flash import FlashProgram
from frontend.near_mem.gemm.plan import FlashPlan, Problem
from frontend.near_mem.gemm.space import candidates
from frontend.near_mem.resources import count, report
from frontend.near_mem.sim import make_hardware


def programs():
    problem = Problem(32, 32, 32); seen = {}
    for receive in ('sram_consume', 'dram_land'):
        for layout in ('local', 'global'):
            for p in candidates(problem, 16, 1 << 20, 4, 'all', 'elk', 1 << 30, True, receive=receive, layout=layout):
                seen.setdefault((receive, layout, bool(p.layers), p.ps[2] > 1), p)
    yield from (compile_plan(problem, p, 1 << 20, 1 << 30) for p in seen.values())
    yield FlashProgram(problem, FlashPlan('os', 8, 8, False, 16), 1 << 20, 1 << 30)


@pytest.mark.parametrize('program', list(programs()))
def test_counts_match_the_compiled_program(program):
    raw, manifest = program.lower(64)
    tensors = [{t['name']: t for t in p['tensor']} for p in program.placements]
    c = count(raw, False, tensors, 64, 4)
    assert (c['macs'], c['vector_ops'], c['noc_bytes']) == (manifest['macs'], manifest['vector_ops'], manifest['noc_bytes'])
    assert (c['dram_read_bytes'], c['dram_write_bytes']) == (manifest['logical_dram_read_bytes'],
                                                             manifest['logical_dram_write_bytes'])
    assert c['noc_byte_hops'] >= c['noc_bytes']


def test_native_all_reduce_is_counted_like_our_tasks(tmp_path):
    from frontend.auto.collectives import allreduce_tasks
    side, rows, length = 4, 8, 64
    partial = dict(name='P', shape=[rows, length], element_size=2)
    result = dict(name='C', shape=[rows, length], element_size=2)
    _, raw = allreduce_tasks(partial, result, 'x', rows, 0, length, 2, 1, 0, side, 64, [0]*16)
    c = count(raw, True, [{'P': partial, 'C': result}]*16, 64, side)
    part = rows*length*2
    # A line reduce-scatter then all-gather: every rank sends (side-1) chunks in each, to a mesh neighbour.
    assert c['noc_bytes'] == c['noc_byte_hops'] == 16*2*(side-1)*part//side
    assert c['dram_read_bytes'] == c['dram_write_bytes'] == 16*part
    assert c['vector_ops'] == 16*(side-1)*rows*length//side
    chip = make_hardware(tmp_path/'hw', 16, 64)
    r = report(c, 1000, chip)
    assert r['energy_j']['total'] == pytest.approx(sum(v for k, v in r['energy_j'].items() if k != 'total'))
    assert 0 < r['utilization']['noc'] < 1
