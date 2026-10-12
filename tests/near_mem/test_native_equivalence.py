"""A larger scheduling space must retain native execution, not just its tiles."""
import json
from types import SimpleNamespace

import pytest

from frontend.auto.generalize import expand_gemm
from frontend.auto.tiling_explorer_cloud import make_gemm_description
from frontend.near_mem.cloud_compare import execute, replacement
from frontend.near_mem.plan import Plan
from frontend.near_mem.run import make_hardware


@pytest.mark.parametrize('tile', [(8, 16, 12), (4, 16, 12), (8, 8, 12), (8, 16, 4),
                                 (4, 8, 4), (2, 4, 4)])
@pytest.mark.parametrize('cores', [1, 4])
def test_native_gemm_and_general_tasks_execute_identically(tmp_path, tile, cores):
    m, k, n = 8, 16, 12
    tensors = [dict(name='input_gemm', shape=[m, k], strides=[k, 1], element_size=2, base_addr=0),
               dict(name='weight_gemm_b0', shape=[k, n], strides=[1, k], element_size=2, base_addr=131072),
               dict(name='output_gemm', shape=[m, n], strides=[n, 1], element_size=2, base_addr=262144)]
    operator = make_gemm_description(SimpleNamespace(name='gemm', M=m, K=k, N=n),
                                     2, tile, 'input_gemm', 'output_gemm', 'weight_gemm_b0')
    core = expand_gemm(operator, tensors)
    folder = tmp_path/'tasks'; folder.mkdir()
    for i in range(cores):
        (folder/f'core_{i}.yaml').write_text(json.dumps(core))
    general = dict(name=operator['name'], type='mpmd-general', iteration=core['iteration'], file_prefix=str(folder))
    chip = make_hardware(tmp_path/'hardware', cores)
    native = execute(chip, tmp_path/'native', dict(operator=[operator]), dict(tensor=tensors))
    expanded = execute(chip, tmp_path/'general', dict(operator=[general]), dict(tensor=tensors))
    assert expanded == native
    tm, tk, tn = tile
    adapted, manifest = replacement(operator, dict(tensor=tensors), Plan((1, 1, 1), (tm, tn, tk)),
                                    chip, tmp_path/'adapter_tasks')
    assert manifest['generator'] == 'shared_native_mnk'
    assert manifest['sram_bytes'] == 4*(tm*tk+tk*tn+2*tm*tn)
    assert execute(chip, tmp_path/'adapter', dict(operator=[adapted]), dict(tensor=tensors)) == native


def test_generalizer_rejects_unsupported_template():
    with pytest.raises(ValueError, match='native GEMM'):
        expand_gemm(dict(type='communication'), [])


def test_upstream_worker_still_searches_after_emitter_extraction(tmp_path):
    from frontend.auto.tiling_explorer_cloud import explore_gemm_comp_worker
    chip=make_hardware(tmp_path/'hardware',1)
    tensors=[dict(name='A',shape=[8,16],strides=[16,1],element_size=2,base_addr=0),
             dict(name='B',shape=[16,12],strides=[1,16],element_size=2,base_addr=131072),
             dict(name='C',shape=[8,12],strides=[12,1],element_size=2,base_addr=262144)]
    tiles=[(4,8,4),(8,16,12)]
    result=explore_gemm_comp_worker(tensors[0],tensors[2],[tensors[1]],
            SimpleNamespace(name='gemm',M=8,N=12,K=16,B=1),2,str(chip),tiles,0,str(tmp_path/'worker'))
    assert result['tiling_factors'] in tiles and result['latency']>0
    assert result['comp_description_list'][0]==make_gemm_description(
            SimpleNamespace(name='gemm',M=8,N=12,K=16),2,result['tiling_factors'],'A','C','B')


@pytest.mark.parametrize('tile',[(4,4,8),(8,4,16),(4,4,16),(8,12,16)])
def test_general_mapping_compiler_reproduces_native_mnk_without_rotation(tmp_path,tile):
    from frontend.near_mem.compiler import compile_plan
    from frontend.near_mem.backend import lower
    from frontend.near_mem.plan import Problem
    problem=Problem(8,12,16);plan=Plan((1,1,1),tile)
    program=compile_plan(problem,plan,131072,1<<29)
    out=tmp_path/'general';lower(program,out,64)
    native=make_gemm_description(SimpleNamespace(name='joint_gemm',M=8,N=12,K=16),
                                2,(tile[0],tile[2],tile[1]),'A','C','B')
    chip=make_hardware(tmp_path/'hardware',1)
    expected=execute(chip,tmp_path/'native',dict(operator=[native]),dict(core_tensor=program.placements))
    actual=execute(chip,out,json.loads((out/'operators.yaml').read_text()),json.loads((out/'placement.yaml').read_text()))
    assert actual==expected
