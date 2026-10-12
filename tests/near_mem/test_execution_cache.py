"""Exact-input cache and parallel search retain native simulator results."""
from copy import deepcopy
from dataclasses import asdict
import json

from frontend.near_mem.attention import AttentionPlan
from frontend.near_mem.backend import fingerprint
from frontend.near_mem.execution import ExecutionCache, configuration_key, task_digest
from frontend.near_mem.graph import Graph, Region
from frontend.near_mem.kernels import Gemm
from frontend.near_mem.experiment import Experiment
from frontend.near_mem.llm import Workload, build, configuration
from frontend.near_mem.plan import Plan
from frontend.near_mem.run import make_hardware


def copy_graph(chip, extra=0):
    graph = Graph(chip)
    graph.tensor(0,'source',[4,16]);graph.tensor(1,'target',[4,16])
    graph.convert('edge',[Region(0,'source',0,0,4,16)],[Region(1,'target',0,0,4,16)])
    if extra:
        timeline = graph.timeline();timeline.work(0,1,vectors=extra)
        graph.add_timeline('extra',timeline)
    return graph


def test_cache_checks_generated_tasks_and_backend_not_only_source(tmp_path,monkeypatch):
    import frontend.near_mem.graph as graph_module
    chip = make_hardware(tmp_path/'hardware',4)
    provenance = fingerprint(chip)
    signature = dict(workload={},configuration={},sha256=provenance)
    key = configuration_key(signature);root = tmp_path/'runs'
    first = root/'first';measured = copy_graph(chip).simulate(first)
    (first/'signature.json').write_text(json.dumps(signature))
    changed_source = dict(provenance,workflow='changed_proposal_policy')
    cache = ExecutionCache(chip,root,changed_source)
    original = graph_module.execute;calls = []
    def execute(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(graph_module,'execute',execute)
    second = copy_graph(chip).simulate(root/'second',cache,key)
    assert second['result'] == measured['result'] and calls == []
    assert task_digest(first,4) == task_digest(root/'second',4)
    assert json.loads((root/'second'/'execution.json').read_text())['reused_from'] == str(first)
    copy_graph(chip,extra=1024).simulate(root/'changed_tasks',cache,key)
    assert calls == [1]
    different_backend = dict(provenance)
    library = next(k for k in different_backend if k.startswith('binary:'))
    different_backend[library] = 'different_binary'
    incompatible = ExecutionCache(chip,root,different_backend)
    assert incompatible.get(task_digest(first,4),key) == (None,None)


def small_configuration():
    workload = Workload(1,4,4,2,1,16,64,1,4)
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
