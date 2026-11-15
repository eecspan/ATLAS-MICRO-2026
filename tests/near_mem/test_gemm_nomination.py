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


@pytest.mark.parametrize('method', ['ours', 'tp', 't10', 'flash'])
def test_search_proposes_distinct_graphs_without_simulation(tmp_path, monkeypatch, method):
    from frontend.near_mem.experiments.transformer import Experiment, Search, signature, token, ORDER
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.model.attention import AttentionPlan, KVLayout
    chip = make_hardware(tmp_path/'hw', 4, 64)
    w = Workload(1, 8, 8, 4, 1, 8, 32); hw = dict(target(chip), single_b=True)
    def forbidden(*a, **kw):
        pytest.fail('Proposal must not invoke the simulator')
    monkeypatch.setattr(search.Graph, 'simulate', forbidden)
    joint = METHODS[method].boundary
    s = search.Search(Experiment(w, chip, tmp_path/'out'), hw, method, 1, joint)
    options = {'attention': s.attention(METHODS[method].attention(KVLayout(w.attention, hw['cores']), hw['sram']))}
    for name in w.problems:
        pool = [] if joint else None
        proposals, _ = search.propose_gemm(w, chip, method, hw, 1, 1, name, pool=pool)
        options[name] = s.gemm(name, proposals, pool)
    configs = s.propose(options, 16, 8, 3)
    graphs = s.report['graphs']
    assert 1 <= len(configs) <= 3 and len(graphs) <= 8+3*5*2
    # Alternates share their option's boundary layouts and rank below it.
    for (name, key), others in s.alternates.items():
        for other in others:
            assert s.ports[name, token(other)] == s.ports[name, key]
            assert s.scores[name, token(other)] >= s.scores[name, key]
    assert [g['whole'] for g in graphs] == sorted(g['whole'] for g in graphs)
    assert sum(g['simulated'] for g in graphs) == len(configs)
    picked = [g['program'] for g in graphs if g['simulated']]
    assert len(picked) == len(set(picked))  # never two graphs of one program
    # Picks prefer structurally different graphs over near-duplicates.
    def sig(cfg):
        return signature({**cfg['kernels'], 'attention': AttentionPlan(**cfg['attention'])})
    distinct = len({sig(g['configuration']) for g in graphs})
    assert len({sig(c) for c in configs}) == min(len(configs), distinct)


def test_whole_graph_score_chains_operators_per_core(tmp_path):
    from frontend.near_mem.model.schedule_proxy import ScheduleProxy
    from frontend.near_mem.tasks import Timeline
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw', 4))
    def op(core):
        t = Timeline(4, 64); t.work(core, 1, macs=100*1024); return t.finish()
    one = proxy.tasks(op(0))
    # Two operators on one core run back to back; on two cores they overlap.
    assert proxy.run([op(0), op(0)]) >= 2*one-4
    assert proxy.run([op(0), op(1)]) < 2*one-4


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


def test_pick_skips_graphs_of_an_equal_program():
    from frontend.near_mem.experiments.transformer import pick
    from frontend.near_mem.model.attention import AttentionPlan
    entry = lambda ps: dict(plan=asdict(Plan(ps, (8, 8, 8))), native=False, direction=0, gated=False)
    graph = lambda ps: {**{n: entry(ps) for n in ('qkv', 'output', 'expand', 'contract')}, 'attention': AttentionPlan()}
    ranked = [(1, graph((1, 2, 2)), 'a'), (2, graph((1, 4, 1)), 'a'), (3, graph((1, 1, 4)), 'b')]
    assert [x[0] for x in pick(ranked, 2, lambda x: x[2])] == [1, 3]
    assert [x[0] for x in pick(ranked, 2)] == [1, 2]


def test_native_checker_follows_the_protocol(tmp_path):
    # Upstream native GEMMs (iteration-coded tasks) take a constant node score,
    # their real boundary layouts, and the additive graph score.
    from frontend.near_mem.experiments.transformer import run
    chip = make_hardware(tmp_path/'hw', 4, 64); w = Workload(1, 1, 8, 4, 1, 8, 32)
    s = run(w, chip, tmp_path/'atlas', 'atlas', workers=2, keep=2, candidates=4)
    assert s['cycles'] > 0 and s['method'] == 'atlas'
