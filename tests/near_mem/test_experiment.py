"""Whole-graph configurations: deterministic placement and parallel measurement."""
from copy import deepcopy
from dataclasses import asdict
import json

from frontend.near_mem.model.attention import AttentionPlan
from frontend.near_mem.model.kernels import Gemm
from frontend.near_mem.experiments.transformer import Experiment
from frontend.near_mem.model.transformer import Workload, build, configuration
from frontend.near_mem.gemm.plan import Plan
from frontend.near_mem.sim import make_hardware


def small_configuration():
    workload = Workload(1,4,4,2,1,16,64)
    kernels = {name:Gemm(p,Plan((1,2,2),(4,8,16)),True,int(name=='expand'),name=='expand')
               for name,p in workload.problems.items()}
    return workload,configuration(kernels,AttentionPlan(1,1,1))


def test_configuration_dictionary_order_does_not_change_physical_placement(tmp_path):
    chip = make_hardware(tmp_path/'hardware',4)
    workload,cfg = small_configuration();reverse = deepcopy(cfg)
    reverse['kernels'] = dict(reversed(list(reverse['kernels'].items())))
    a,b = build(workload,chip,cfg),build(workload,chip,reverse)
    assert a.tensors == b.tensors and a.operations == b.operations


def test_parallel_neighbors_have_identical_serial_results(tmp_path):
    chip = make_hardware(tmp_path/'hardware',4)
    workload,first = small_configuration();second = deepcopy(first)
    second['attention'] = asdict(AttentionPlan(2,2))
    serial = Experiment(workload,chip,tmp_path/'serial')
    expected = [serial.measure(cfg,'serial')['result'] for cfg in (first,second)]
    parallel = Experiment(workload,chip,tmp_path/'parallel')
    actual = parallel.measure_many([first,second,deepcopy(first)],'parallel',2)
    assert len(actual) == 2
    assert [row['result'] for row in actual] == expected
    assert len(json.loads((tmp_path/'parallel'/'measurements.json').read_text())) == 2
