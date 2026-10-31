"""Dependency deadlines, one DRAM resource, and bounded SRAM prefetch."""
import pytest

from frontend.near_mem.sim import make_hardware
from frontend.near_mem.tasks import Graph, Region
from frontend.near_mem.model.attention import Attention, AttentionPlan, KVLayout, compile_attention
from frontend.near_mem.gemm.compiler import compile_plan
from frontend.near_mem.gemm.plan import Plan, Problem


def stream(chip, out, deadline, work=100, reads=1, consumer=0):
    g = Graph(chip); g.tensor(0,'x',[1,16384])
    t = g.timeline()
    for i in range(reads):
        t.access(0,i+1,'x',[0,0],[1,16384],ready=deadline)
    for i in range(1,5):
        t.work(0,i,macs=work*1024)
    t.work(0,5,macs=consumer*1024,vectors=1)  # First consumer must wait.
    g.add_timeline('stream',t)
    return g.simulate(out)['result']


def test_dram_prefetch_overlaps_multiple_iterations_and_waits_at_use(tmp_path):
    chip = make_hardware(tmp_path/'hw',1)
    old = stream(chip,tmp_path/'blocking',None)
    new = stream(chip,tmp_path/'prefetch',5)
    assert new['e2e_cycles'] < old['e2e_cycles']
    for field in ('memory_access_bytes','dram_cycles','flop_count','matrix_cycles'):
        assert new[field] == old[field]
    assert new['e2e_cycles'] >= max(new['dram_cycles'],new['matrix_cycles'])
    # Longer independent compute hides more of the same DRAM request.
    large = stream(chip,tmp_path/'hidden',5,work=1000)
    assert large['e2e_cycles'] <= 4008
    use = stream(chip,tmp_path/'consumer',5,work=1,consumer=2000)
    assert use['e2e_cycles'] >= use['dram_cycles']+2000


def test_prefetches_share_bandwidth_and_drain_before_operator_exit(tmp_path):
    chip = make_hardware(tmp_path/'hw',1)
    one = stream(chip,tmp_path/'one',5,work=1)
    two = stream(chip,tmp_path/'two',5,work=1,reads=2)
    assert two['memory_access_bytes'] == 2*one['memory_access_bytes']
    assert two['e2e_cycles'] >= two['dram_cycles'] > one['dram_cycles']
    # An explicit immediate dependency is the original blocking semantics.
    immediate = stream(chip,tmp_path/'immediate',2)
    blocking = stream(chip,tmp_path/'blocking',None)
    assert immediate == blocking


def test_attention_prefetch_has_one_load_and_first_local_consumer(tmp_path):
    from frontend.near_mem.model.attention_schedule import Schedule
    p = Attention(1,16,64,4,1,8); layout = KVLayout(p,4)
    plan = AttentionPlan(4,1,2,2)
    s = Schedule(layout,plan)
    for e in s.events:
        if e.kind != 'kv_load':continue
        uses = [u.time for u in s.events if u.time>e.time and u.core==e.core and
                ((u.kind=='update' and u.other==e.key) or (u.kind in ('kv_move','kv_send') and u.key==e.key))]
        assert e.ready == min(uses)
    assert any(e.kind=='kv_load' and e.ready>e.time+4 for e in s.events)
    assert s.buffer_peak()<=plan.sram_bytes(p)
    chip = make_hardware(tmp_path/'hw',4)
    g = Graph(chip); kv = layout.allocate(g)
    g.tensor(0,'q',[16,32])
    compile_attention(g,layout,plan,[Region(0,'q',0,0,16,32)],kv)
    streamed = g.simulate(tmp_path/'streamed')['result']
    for op in g.operations:
        for core in op['raw']:
            for phase in core:
                for task in phase['dram']:task.pop('ready_iter',None)
    blocked = g.simulate(tmp_path/'blocked')['result']
    assert streamed['e2e_cycles']<=blocked['e2e_cycles']
    assert streamed['memory_access_bytes']==blocked['memory_access_bytes']


def test_gemm_prefetch_respects_bank_ownership_and_disabled_overlap():
    p = Problem(32,64,64); plan = Plan((1,4,1),(4,4,4),a=(1,4),group=(1,2,4))
    program = compile_plan(p,plan,1<<20,1<<30)
    raw,_ = program.lower(64)
    reads = [(i,d) for row in raw for i,phase in enumerate(row) for d in phase['dram']
             if 'ready_iter' in d and d['name'] in ('A','B')]
    assert reads and all(d['ready_iter']>i for i,d in reads)
    assert any(d['ready_iter']>i+1 for i,d in reads)
    raw,_ = compile_plan(p,plan,1<<20,1<<30,overlap=False).lower(64)
    assert not any('ready_iter' in d for row in raw for phase in row for d in phase['dram'])


def test_prefetch_rejects_invalid_dependencies(tmp_path):
    g = Graph(make_hardware(tmp_path/'hw',1)); t = g.timeline()
    with pytest.raises(ValueError,match='later consumer'):
        t.access(0,3,'x',[0,0],[1,1],ready=3)
    with pytest.raises(ValueError,match='read'):
        t.access(0,3,'x',[0,0],[1,1],write=True,ready=4)
