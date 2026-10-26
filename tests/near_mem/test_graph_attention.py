"""Ownership, actual DRAM windows, and native execution of complete graphs."""
from copy import deepcopy
from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from frontend.near_mem.model.attention import Attention, AttentionPlan, KVLayout, compile_attention
from frontend.near_mem.tasks import Graph, Region, transfers, unique
from frontend.near_mem.model.kernels import Gemm, program_regions
from frontend.near_mem.model.transformer import Workload, build, configuration
from frontend.near_mem.gemm.plan import Plan, Problem
from frontend.near_mem.sim import make_hardware


def replay_copy(raw, arrays):
    """Replay emitted DRAM/NoC events with native slot semantics.

    Slot ``s`` reads DRAM, sends/receives one stage later and writes DRAM two
    stages later; each slot owns its read and receive buffers (pipelining).
    """
    read = {}; received = {}; packets = {}
    for s, _ in enumerate(raw[0]):
        for core, phases in enumerate(raw):
            for task in phases[s]['dram']:
                if not task['is_write']:
                    r,c = task['access_base']; h,w = task['access_extent']
                    read[core,s] = arrays[core,task['name']][r:r+h,c:c+w].copy()
    for s, _ in enumerate(raw[0]):
        for core, phases in enumerate(raw):
            for task in phases[s]['noc_tx']:
                packets[task['src'],task['init_flit_id']] = read[core,s]
    for s, _ in enumerate(raw[0]):
        for core, phases in enumerate(raw):
            for task in phases[s]['noc_rx']:
                received[core,s] = packets[task['src'],task['init_flit_id']]
    for s, _ in enumerate(raw[0]):
        for core, phases in enumerate(raw):
            for task in phases[s]['dram']:
                if task['is_write']:
                    r,c = task['access_base']; h,w = task['access_extent']
                    arrays[core,task['name']][r:r+h,c:c+w] = received.get((core,s), read.get((core,s)))


@pytest.fixture
def chip(tmp_path):
    return make_hardware(tmp_path/'hardware',4,sram_kib=16)


@pytest.mark.parametrize('source_parts,target_parts',[(1,4),(4,1),(2,4),(4,2)])
def test_actual_conversion_tasks_preserve_values_and_create_replicas(chip,source_parts,target_parts):
    graph = Graph(chip); full = np.arange(16*32).reshape(16,32); arrays = {}; source = []; target = []
    for core in range(source_parts):
        lo,hi = 16*core//source_parts,16*(core+1)//source_parts
        graph.tensor(core,'source',[hi-lo,32]); arrays[core,'source'] = full[lo:hi].copy()
        source.append(Region(core,'source',lo,0,hi-lo,32))
    for core in range(4):
        col = (core % target_parts)*(32//target_parts)
        graph.tensor(core,'target',[16,32//target_parts]); arrays[core,'target'] = np.full((16,32//target_parts),-1)
        target.append(Region(core,'target',0,col,16,32//target_parts))
    graph.convert('edge',source,target)
    replay_copy(graph.operations[0]['raw'],arrays)
    for r in target:
        np.testing.assert_array_equal(arrays[r.core,r.name],full[:,r.col:r.col+r.cols])
    assert graph.edges[0]['dram_write_bytes'] == sum(r.area for r in target)*2
    # Every unicast reads its own panel; SRAM keeps no panel across steps.
    assert graph.edges[0]['dram_read_bytes'] == graph.edges[0]['dram_write_bytes']


def test_missing_producer_data_is_rejected():
    with pytest.raises(ValueError,match='Uncovered'):
        transfers([Region(0,'x',0,0,4,4)],[Region(1,'y',0,0,4,5)])


def test_kv_request_shards_have_stable_owner(chip):
    p = Attention(3,4,17,4,2,8)
    layout = KVLayout(p,4)
    graph = Graph(chip)
    regions = layout.allocate(graph)
    assert len(regions) == p.kv_heads*p.batch*layout.shard_cores
    for r in regions:
        request = r.row//p.context
        head = r.col//(2*p.head_dim)
        start, length = layout.shard(request, head, r.core)
        assert r.row == request*p.context+start and r.rows == length
        assert r.cols == 2*p.head_dim


@pytest.mark.parametrize('queries,context',[(1,17),(7,19),(19,19)])
def test_append_emitted_addresses_match_future_attention_reads(chip,queries,context):
    p = Attention(2,queries,context,4,2,8)
    graph = Graph(chip); layout = KVLayout(p,4); cache = layout.allocate(graph)
    columns = p.hidden+2*p.kv_heads*p.head_dim
    projection = np.arange(p.batch*queries*columns).reshape(p.batch*queries,columns)
    graph.tensor(0,'qkv',list(projection.shape)); source = [Region(0,'qkv',0,0,*projection.shape)]
    arrays = {(c,t['name']):np.full(t['shape'],-1) for c,ts in enumerate(graph.tensors) for t in ts.values()}
    arrays[0,'qkv'] = projection.copy()
    layout.append(graph,source,cache)
    replay_copy(graph.operations[-1]['raw'],arrays)
    for r in cache:
        request = r.row//context; head = r.col//(2*p.head_dim)
        for j in range(r.rows):
            token = r.row+j-request*context
            actual = arrays[r.core,r.name][r.local_row+j]
            if token < context-queries:
                assert np.all(actual == -1)
            else:
                row = request*queries+token-(context-queries)
                expected = np.concatenate([projection[row,p.hidden+head*p.head_dim:p.hidden+(head+1)*p.head_dim],
                                           projection[row,p.hidden+(p.kv_heads+head)*p.head_dim:p.hidden+(p.kv_heads+head+1)*p.head_dim]])
                np.testing.assert_array_equal(actual,expected)
    assert graph.edges[-1]['dram_write_bytes'] == 2*p.batch*queries*p.kv_heads*p.head_dim*2


@pytest.mark.parametrize('queries,context,query_parts,kv_tile',[(1,19,1,0),(8,8,1,0),(8,8,2,0),(5,17,2,0),(5,17,1,4)])
def test_fused_attention_mac_coverage_bounds_and_no_history_write(chip,tmp_path,queries,context,query_parts,kv_tile):
    graph = Graph(chip); p = Attention(2,queries,context,4,2,8)
    layout = KVLayout(p,4); cache = layout.allocate(graph)
    graph.tensor(0,'Q',[p.batch*queries,p.hidden]); q = [Region(0,'Q',0,0,p.batch*queries,p.hidden)]
    plan = AttentionPlan(query_parts,1,1,kv_tile)
    out = compile_attention(graph,layout,plan,q,cache)
    info = graph.operations[-1]['metadata']
    scores = p.batch*p.query_heads*sum(range(context-queries+1,context+1))
    assert info['score_elements_issued'] == scores
    assert info['macs'] == 2*scores*p.head_dim
    assert sum(r.area for r in unique(out)) == p.batch*queries*p.hidden
    assert info['sram_bytes'] <= graph.sram
    for core,phases in enumerate(graph.operations[-1]['raw']):
        for phase in phases:
            for task in phase['dram']:
                tensor = graph.tensors[core][task['name']]
                assert all(x >= 0 and x+n <= size for x,n,size in zip(task['access_base'],task['access_extent'],tensor['shape']))
                assert not (task['is_write'] and task['name']=='kv')
    measured = graph.simulate(tmp_path/'sim')['result']['operators']['attention']
    assert measured['flop_count'] == 2*info['macs']+info['vector_ops']


def test_capacity_and_storage_reuse_are_enforced(chip):
    graph = Graph(chip); a = graph.tensor(0,'a',[4,32]); region = Region(0,'a',0,0,4,32)
    b = graph.tensor(0,'b',[4,32]); assert b['base_addr'] >= a['base_addr']+256
    graph.release([region]); c = graph.tensor(0,'c',[4,32]); assert c['base_addr']==a['base_addr']
    with pytest.raises(ValueError,match='capacity'):
        graph.tensor(0,'huge',[graph.capacity,2])


@pytest.mark.parametrize('queries,context,layers',[(8,8,1),(1,17,1),(4,4,2)])
def test_native_and_full_space_transformer_graphs_execute(chip,tmp_path,queries,context,layers):
    workload = Workload(1,queries,context,2,1,16,64,layers)
    native = {name:Gemm(p,Plan((1,2,2),(min(4,queries),8,min(16,p.k//2))),True,int(name=='expand'),name=='expand')
              for name,p in workload.problems.items()}
    baseline = configuration(native,AttentionPlan(1,1,1))
    configs = [baseline]
    if queries % 4 == 0:
        extended = deepcopy(baseline)
        extended['kernels']['qkv'].update(native=False,plan=Plan((1,1,4),(4,4,8),c=(1,4)).__dict__)
        extended['kernels']['expand'].update(native=False,plan=Plan((1,1,4),(4,8,8),c=(1,4)).__dict__)
        extended['attention'].update(query_parts=2)
        configs.append(extended)
    for i,cfg in enumerate(configs):
        graph = build(workload,chip,cfg)
        result = graph.simulate(tmp_path/f'graph{i}')['result']
        assert result['e2e_cycles'] > 0
        assert all(n <= graph.capacity for n in graph.peak)
        macs = sum(task['mac_count'] for op in graph.operations if op['kind']=='mpmd-general'
                   for core in op['raw'] for phase in core for task in phase['matrix'])
        expected = layers*(sum(p.m*p.n*p.k for p in workload.problems.values())
                           +2*workload.query_heads*workload.head_dim*sum(range(context-queries+1,context+1)))
        assert macs == expected
        for op in graph.operations:
            if op['kind'] != 'mpmd-general':
                continue
            flops = sum(2*t['mac_count'] for core in op['raw'] for phase in core for t in phase['matrix'])
            flops += sum(t['vec_count'] for core in op['raw'] for phase in core for t in phase['vector'])
            assert result['operators'][op['name']]['flop_count'] == flops


@pytest.mark.parametrize('native',[False,True])
def test_q_rope_is_fused_once_before_replication(chip,native):
    problem = Attention(2,4,19,2,1,16,)
    spec = Problem(8,64,32)
    plan = Plan((1,2,2),(4,8,16)) if native else Plan((1,1,4),(4,4,8),c=(1,4))
    graphs = []
    for rope in (False,True):
        graph = Graph(chip); kernel = Gemm(spec,plan,native).prepare(graph,'qkv')
        layout = KVLayout(problem,4); cache = layout.allocate(graph)
        graph.tensor(0,'input',[8,32]); source = [Region(0,'input',0,0,8,32)]
        kernel.qkv(graph,source,layout,cache,rope=rope)
        graphs.append(graph)
    def vectors(obj):
        if isinstance(obj,dict):
            return obj.get('vec_count',0)+sum(vectors(v) for v in obj.values())
        if isinstance(obj,list):
            return sum(vectors(v) for v in obj)
        return 0
    work = [sum(vectors(op['raw']) for op in graph.operations) for graph in graphs]
    assert work[1]-work[0] == 6*problem.batch*problem.queries*problem.hidden
    assert len(graphs[0].operations) == len(graphs[1].operations)


def test_request_level_kv_shards_have_stable_owner(chip):
    p = Attention(3,4,17,4,2,8)
    layout = KVLayout(p,4)
    graph = Graph(chip); cache = layout.allocate(graph)
    assert [r.core for r in cache] == ([0,1]*p.batch + [2,3]*p.batch)


def test_short_requests_use_request_owner(chip,tmp_path):
    graph = Graph(chip); p = Attention(4,1,1,2,1,8)
    layout = KVLayout(p,4);cache = layout.allocate(graph)
    assert [r.core for r in cache] == [0,1,2,3]
    graph.tensor(0,'Q',[4,16]);q = [Region(0,'Q',0,0,4,16)]
    out = compile_attention(graph,layout,AttentionPlan(),q,cache)
    assert [r.core for r in out] == [0,1,2,3]
    graph.simulate(tmp_path/'short_requests')


@pytest.mark.parametrize('plan',[Plan((1,4,1),(2,4,4)),Plan((1,2,2),(2,4,4)),
                                Plan((1,1,4),(2,4,4),c=(1,4))])
@pytest.mark.parametrize('loop',[''.join(p) for p in permutations('mnk')])
def test_gate_fusion_after_complete_reduction_has_correct_values_and_addresses(chip,tmp_path,plan,loop):
    graph = Graph(chip); spec = Problem(8,16,16); plan = replace(plan,loop=loop)
    kernel = Gemm(spec,plan,gated=True).prepare(graph,'ffn')
    graph.tensor(0,'input',[8,16]); source = [Region(0,'input',0,0,8,16)]
    output = kernel.compile(graph,source)
    rng = np.random.default_rng(7)
    a,wg,wu = rng.normal(size=(8,16)),rng.normal(size=(16,8)),rng.normal(size=(16,8))
    half = plan.tile[1]//2
    packed = np.concatenate([x for i in range(0,8,half) for x in (wg[:,i:i+half],wu[:,i:i+half])],axis=1)
    projected = kernel.program.replay(a,packed)
    arrays = {(c,t['name']):np.full(t['shape'],np.nan) for c,ts in enumerate(graph.tensors) for t in ts.values()}
    expected_stores = [[] for _ in range(graph.cores)]
    for phase in kernel.program.phases:
        for core,actions in enumerate(phase):
            for action in actions:
                if action['op'] != 'store' or not action.get('final'):
                    continue
                row,col = action['key']; tm,tn,_ = plan.tile
                values = projected[row*tm:(row+1)*tm,col*tn:(col+1)*tn]
                gated = values[:,:half]/(1+np.exp(-values[:,:half]))*values[:,half:]
                r,c = action['base']; c //= 2
                arrays[core,'ffn_G'][r:r+tm,c:c+half] = gated
                expected_stores[core].append(([r,c],[tm,half]))
    actual = np.full((8,8),np.nan); coverage = np.zeros((8,8),dtype=int)
    for r in output:
        actual[r.row:r.row+r.rows,r.col:r.col+r.cols] = arrays[r.core,r.name][r.local_row:r.local_row+r.rows,r.local_col:r.local_col+r.cols]
        coverage[r.row:r.row+r.rows,r.col:r.col+r.cols] += 1
    np.testing.assert_array_equal(coverage,1)
    gate_values = a@wg
    np.testing.assert_allclose(actual,gate_values/(1+np.exp(-gate_values))*(a@wu),atol=1e-11)
    op = graph.operations[-1]; assert op['name']=='ffn' and op['metadata']['fused_gate']
    for core,phases in enumerate(op['raw']):
        stores = [(t['access_base'],t['access_extent']) for phase in phases for t in phase['dram']
                  if t['is_write'] and t['name']=='ffn_G']
        assert stores == expected_stores[core]
        assert 'ffn_C' not in graph.live[core]
    vectors = sum(t['vec_count'] for phases in op['raw'] for phase in phases for t in phase['vector']
                  if t['name']=='output_epilogue')
    assert vectors == 5*8*8
    if loop == 'mkn':
        measured = graph.simulate(tmp_path/'gate')['result']['operators']['ffn']
        assert measured['flop_count']==2*8*16*16+op['metadata']['vector_ops']


def test_request_shards_have_no_hidden_capacity_contract():
    p = Attention(1,1,1,2,1,8)
    layout = KVLayout(p,4)
    assert layout.token_owner(0,0,0) == 0


def test_incomplete_fused_pairs_are_rejected(chip):
    graph = Graph(chip)
    with pytest.raises(ValueError,match='gate/up'):
        Gemm(Problem(4,15,16),Plan((1,1,4),(2,3,4)),gated=True).prepare(graph,'invalid')


@pytest.mark.parametrize('layers',[1,2])
def test_edge_traffic_model_matches_built_graph(tmp_path,layers):
    """The search's layout-edge traffic equals the DRAM writes the graph emits."""
    from frontend.near_mem.model.transformer import decode_configuration, edge_regions, edges, port
    from frontend.near_mem.tasks import traffic
    chip = make_hardware(tmp_path/'hardware',16,64)
    w = Workload(2,32,32,4,2,16,64,layers); m = w.batch*w.queries
    names = {('row','qkv.A'):'l0_qkv_A_layout',('contract.C','qkv.A'):'l1_qkv_A_layout',
             ('qkv.C','kv'):'l0_qkv_kv_append',('qkv.C','attention.Q'):'l0_attention_query_layout',
             ('attention.O','output.A'):'l0_output_A_layout',('row','output.C'):'l0_attention_residual_layout',
             ('contract.C','output.C'):'l1_attention_residual_layout',('output.C','expand.A'):'l0_expand_A_layout',
             ('expand.C','contract.A'):'l0_contract_A_layout',('output.C','contract.C'):'l0_ffn_residual_layout',
             ('contract.C','row'):'output_layout'}
    for plans,attention in [({},AttentionPlan(1,1,1)),
                            ({'qkv':Plan((16,1,1),(m//16,16,16)),'contract':Plan((4,4,1),(m//4,16,16)),
                              'expand':Plan((2,8,1),(m//2,16,64))},AttentionPlan(2,1,4)),
                            ({'qkv':Plan((1,16,1),(m,8,16)),'output':Plan((4,4,1),(m//4,4,16))},AttentionPlan(1,2,1))]:
        kernels = {n:Gemm(p,plans.get(n,Plan((1,4,4),(16,16,16))),n not in plans,int(n=='expand'),n=='expand')
                   for n,p in w.problems.items()}
        graph = build(w,chip,configuration(kernels,attention))
        written = {e['name']:e['dram_write_bytes'] for e in graph.edges}
        bound,_ = decode_configuration(w,configuration(kernels,attention))
        choice = {k:v.bind(16,graph.sram,graph.capacity) for k,v in bound.items()}; choice['attention'] = attention
        for src,dst,alias,_ in edges(w):
            regions = edge_regions(w,16,dst,port(w,16,src,choice.get(src.split('.')[0])),
                                   port(w,16,dst,choice.get(dst.split('.')[0])))
            assert 2*sum(x[1] for x in traffic(*regions,alias).values()) == written[names[src,dst]]
