"""Schedule cost model: causal dependencies and physical resource costs."""
import pytest

from frontend.near_mem.model.schedule_proxy import ScheduleProxy
from frontend.near_mem.sim import make_hardware
from frontend.near_mem.tasks import Timeline


def test_compute_only_and_independent_core_progress(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw', 4))
    t = Timeline(4,64)
    t.work(0,1,macs=100*1024)
    t.work(1,2,macs=100*1024)
    assert proxy.tasks(t.finish()) == 103


def test_prefetch_deadline(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',1))
    def score(ready):
        t = Timeline(1,64)
        t.access(0,0,'x',[0,0],[1,16384],ready=ready)
        for i in range(1,5):
            t.work(0,i,macs=100*1024)
        return proxy.tasks(t.finish())
    assert score(4) < score(None)


def test_receive_waits_for_sender_compute(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    def score(macs):
        t = Timeline(4,64)
        t.work(0,1,macs=macs)
        t.send(2,0,3,64)
        return proxy.tasks(t.finish())
    assert score(100*1024)-score(0) == 99


def test_path_latency_and_shared_ejection_bandwidth(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    def score(dst,second=False):
        t = Timeline(4,64);t.send(1,0,dst,6400)
        if second:
            t.send(1,1,dst,6400)
        return proxy.tasks(t.finish())
    assert score(3) > score(1)
    assert score(3,True) > score(3)


def test_memory_deduplicates_bursts_and_uses_physical_axis(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    tensors = {'x':dict(element_size=2,base_addr=0,strides=[8,1]),
               'xt':dict(element_size=2,base_addr=0,strides=[1,8])}
    task = dict(name='x',access_base=[0,0],access_extent=[2,8],is_write=False)
    contiguous = dict(task,access_extent=[1,16])
    transposed = dict(task,name='xt',access_extent=[8,2])
    assert proxy.memory([task],tensors) == proxy.memory([contiguous],tensors)
    assert proxy.memory([task],tensors) == proxy.memory([transposed],tensors)


def test_missing_sender_fails_closed(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    t = Timeline(4,64);t.send(1,0,1,64);raw=t.finish()
    raw[0][0]['noc_tx']=[]
    with pytest.raises(ValueError,match='Unmatched'):
        proxy.tasks(raw)


def test_cache_keys_include_address_and_stride(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    t = dict(name='x',access_base=[0,0],access_extent=[4,8],is_write=False)
    tensor = dict(element_size=2,base_addr=0,strides=[8,1])
    original = proxy.memory([t],{'x':tensor})
    assert proxy.memory([t],{'x':dict(tensor,strides=[4096,1])}) > original
    assert proxy.memory([t],{'x':dict(tensor,base_addr=127)}) > original
    assert proxy.memory([t],{'x':tensor}) == original


def test_invalid_dma_deadline_fails_closed(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',1))
    t = Timeline(1,64);t.access(0,0,'x',[0,0],[1,64])
    raw = t.finish();raw[0][0]['dram'][0]['ready_iter'] = 0
    with pytest.raises(ValueError,match='deadline'):
        proxy.tasks(raw)


def test_batch_transaction_dedup_keeps_read_write_distinct(tmp_path):
    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',4))
    tensors = {'x':dict(element_size=2,base_addr=0,strides=[64,1])}
    task = dict(name='x',access_base=[0,0],access_extent=[1,64],is_write=False)
    one = proxy.transactions([task],tensors)
    assert proxy.transactions([task,task],tensors) == one
    both = proxy.transactions([task,dict(task,is_write=True)],tensors)
    assert both[False] == both[True] == one[False]


def test_calibrated_hop_latency_can_replace_pipeline_parameters(tmp_path):
    chip = make_hardware(tmp_path/'hw',4)
    t = Timeline(4,64);t.send(1,0,3,6400);raw=t.finish()
    assert ScheduleProxy(chip,hop_cycles=10).tasks(raw) > ScheduleProxy(chip,hop_cycles=1).tasks(raw)


def test_custom_memory_state_is_local_to_one_evaluation(tmp_path):
    class Service:
        def __init__(self,chip,backend):
            self.calls = 0

        def service(self,transactions):
            if not any(transactions.values()):
                return 0
            self.calls += 1
            return self.calls*100

    proxy = ScheduleProxy(make_hardware(tmp_path/'hw',1),memory_factory=Service)
    t = Timeline(1,64);t.access(0,0,'x',[0,0],[1,64]);raw=t.finish()
    tensors = [{'x':dict(element_size=2,base_addr=0,strides=[64,1])}]
    first = proxy.tasks(raw,tensors)
    assert first >= 100
    assert proxy.tasks(raw,tensors) == first
