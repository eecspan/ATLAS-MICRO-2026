"""Compiled scoring changes nominations, not measurement or diversity budgets."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from frontend.near_mem.experiments import transformer as search
from frontend.near_mem.gemm.plan import Plan, load
from frontend.near_mem.gemm.space import structure
from frontend.near_mem.methods import target
from frontend.near_mem.model.transformer import Workload
from frontend.near_mem.search import diverse
from frontend.near_mem.sim import make_hardware


def test_expanded_scan_only_uses_coarse_bounds(monkeypatch):
    score, bound, space = object(), object(), object()
    monkeypatch.setattr(search, '_gemm_scoring', lambda *a: (None, None, None, score, bound, space))
    calls = []
    monkeypatch.setattr(search, 'diverse_part', lambda *a: calls.append(a) or ([], 0))
    search.scan_gemm(None, None, None, None, 3, 2, 'output', 1, 4)
    assert calls == [(space, score, 12, structure, 8, bound, 1, 4)]


def test_fine_score_can_recover_coarse_reject_without_changing_diversity(monkeypatch):
    plans = [Plan((1, 2, 2), (8, n, 8), a=a) for a in ((1, 1), (2, 1)) for n in (8, 16, 32)]
    costs = {p: i+1 for i, p in enumerate(plans)}
    coarse = lambda p: (costs[p], costs[p])
    backend = SimpleNamespace(distribution=lambda *a: 0)
    method = SimpleNamespace(oriented=lambda n: True)
    monkeypatch.setattr(search, '_gemm_scoring', lambda *a: (None, backend, method, coarse, None, None))
    monkeypatch.setattr(search, 'roles', lambda *a: dict(direction=0, gated=False))
    calls = []
    def fine(*args, **kwargs):
        p = args[-1]; calls.append(p)
        return dict(kernel=10-costs[p], compile_seconds=0, proxy_seconds=0)
    monkeypatch.setattr(search, '_schedule_score', fine)
    pieces = [([(p, (i//3, i)) for i, p in enumerate(plans)], 1000)]
    report = {}
    entries, scanned = search.propose_gemm(None, None, 'ours', dict(cores=4), 1, 1, 'output',
                                            pieces, report=report)
    assert scanned == 1000
    assert [load(e['plan']) for e in entries] == [plans[-1], replace(plans[-1], transpose=True)]
    assert len(calls) == len(set(calls)) == report['fine_scored'] == 6
    assert report['selected_before_twins'] == 1
    assert diverse(lambda skip: iter(plans), coarse, 1, structure, 1)[0][0][0] == plans[0]


def test_real_nomination_parallel_matches_serial_and_never_simulates(tmp_path, monkeypatch):
    chip = make_hardware(tmp_path/'hw', 4, 64)
    w = Workload(1, 8, 8, 4, 1, 8, 32); hw = dict(target(chip), single_b=True)
    def forbidden(*a, **kw):
        pytest.fail('Proxy nomination must not invoke the simulator')
    monkeypatch.setattr(search.Graph, 'simulate', forbidden)
    whole = [search.scan_gemm(w, chip, 'ours', hw, 1, 1, 'output')]
    split = [search.scan_gemm(w, chip, 'ours', hw, 1, 1, 'output', p, 3) for p in range(3)]
    first, second = {}, {}
    a, _ = search.propose_gemm(w, chip, 'ours', hw, 1, 1, 'output', whole, report=first)
    b, _ = search.propose_gemm(w, chip, 'ours', hw, 1, 1, 'output', split, workers=2, report=second)
    assert a == b
    clean = lambda report: [(r['plan'], r['coarse'], r['fine'], r['selected']) for r in report['candidates']]
    assert clean(first) == clean(second)
    assert first['fine_scored'] == first['coarse_pool'] <= 2*4*4
    assert 0 < len(a) <= 2*1*1*2  # two scores, one family/tile each, both orientations
    assert all(r['kernel'] > 0 for r in first['candidates'])


def test_standalone_simulation_remains_the_measurement(monkeypatch, tmp_path):
    seen = []
    class Graph:
        def simulate(self, folder, deadline=None):
            seen.append((folder, deadline))
            return dict(result=dict(e2e_cycles=1234))
    monkeypatch.setattr(search, 'operator_graph', lambda *a: Graph())
    choice = dict(plan=asdict(Plan((1, 2, 2), (8, 8, 8))), native=False, direction=0, gated=False)
    assert search.simulate_operator(None, None, None, tmp_path, 'output', choice, deadline=99) == 1234
    assert len(seen) == 1 and seen[0][0].parent == tmp_path/'operators' and seen[0][1] == 99


@pytest.mark.parametrize('method', ['tp', 'local', 'dram', 't10', 'flash'])
def test_shared_nomination_supports_general_compiler_methods(tmp_path, method):
    chip = make_hardware(tmp_path/'hw', 4, 64)
    w = Workload(1, 8, 8, 4, 1, 8, 32); hw = dict(target(chip), single_b=True)
    report = {}
    entries, count = search.propose_gemm(w, chip, method, hw, 1, 1, 'output', report=report)
    assert count > 0 and entries and report['fine_scored'] > 0
    assert all(r['kernel'] > 0 for r in report['candidates'])


def test_joint_nomination_keeps_one_plan_per_layout_and_never_simulates(tmp_path, monkeypatch):
    from frontend.near_mem.experiments.transformer import Chain, Experiment, nominate_joint, ORDER
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.model.attention import KVLayout
    chip = make_hardware(tmp_path/'hw', 4, 64)
    w = Workload(1, 8, 8, 4, 1, 8, 32); hw = dict(target(chip), single_b=True)
    def forbidden(*a, **kw):
        pytest.fail('Proxy nomination must not invoke the simulator')
    monkeypatch.setattr(search.Graph, 'simulate', forbidden)
    pools = {}; kernel = {}
    for name in w.problems:
        pools[name] = []
        kernel[name], _ = search.propose_gemm(w, chip, 'ours', hw, 1, 1, name, pool=pools[name])
        layouts = [(tuple(a), tuple(c)) for _, _, (a, c) in pools[name]]
        assert len(pools[name]) > len(kernel[name])//2 and len(set(layouts)) > 1
    chain = Chain(Experiment(w, chip, tmp_path/'out'), hw, 1)
    attention = METHODS['ours'].attention(KVLayout(w.attention, hw['cores']), hw['sram'])
    picked, report = nominate_joint(chain, pools, attention, 16, 4)
    assert set(picked) == set(w.problems) and all(picked[n] for n in picked)
    assert 1 <= len(report['graphs']) <= 4 and report['edges'] > 0
    scores = [g['score'] for g in report['graphs']]
    assert scores == sorted(scores)


def test_compiled_edge_score_is_zero_for_a_bound_layout_and_positive_otherwise(tmp_path):
    from frontend.near_mem.model.transformer import relayout
    from frontend.near_mem.tasks import Region
    chip = make_hardware(tmp_path/'hw', 4, 64); w = Workload(1, 8, 8, 4, 1, 8, 32)
    rows = [Region(c, 'x', 2*c, 0, 2, 32) for c in range(4)]
    cols = [Region(c, 'y', 0, 8*c, 8, 8) for c in range(4)]
    assert relayout(w, chip, 'output.A', 'bind', rows, rows) == 0
    assert relayout(w, chip, 'output.A', 'bind', rows, cols) > 0


def test_norm_score_follows_the_producer_layout(tmp_path):
    from frontend.near_mem.model.transformer import norm_score
    from frontend.near_mem.tasks import Region
    chip = make_hardware(tmp_path/'hw', 4, 64); w = Workload(1, 8, 8, 4, 1, 8, 32)
    rows = [Region(c, 'x', 2*c, 0, 2, 32) for c in range(4)]
    cols = [Region(c, 'y', 0, 8*c, 8, 8) for c in range(4)]
    # Column shards reduce row statistics across cores; row shards do not.
    assert 0 < norm_score(w, chip, rows) < norm_score(w, chip, cols)
