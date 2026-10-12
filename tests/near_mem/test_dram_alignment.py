"""A burst is issued iff its byte interval intersects the access window."""
import json

import pytest

from frontend.near_mem.cloud_compare import execute
from frontend.near_mem.backend import KEYS
from frontend.near_mem.run import make_hardware


@pytest.mark.parametrize('rank',[2,4])
@pytest.mark.parametrize('write',[False,True])
@pytest.mark.parametrize('start,count,bursts',[(8,16,1),(32,96,2),(0,64,1),(60,8,2),(64,64,1),(8,0,0)])
def test_exact_burst_coverage(tmp_path,rank,write,start,count,bursts):
    chip=make_hardware(tmp_path/'hardware',1)
    # HBDRAM 1024 pins, prefetch 1: one 128-byte transaction.
    tensor=dict(name='X',base_addr=0,element_size=2,shape=[1]*(rank-1)+[256],strides=[256]*(rank-1)+[1])
    dram=dict(name='X',is_write=write,access_base=[0]*(rank-1)+[start],access_extent=[1]*(rank-1)+[count],
              access_stride_add=[1]*rank,access_offset_add=[0]*rank,init_iter=0,stride_iter=1,total_iter=1)
    folder=tmp_path/'tasks';folder.mkdir()
    phase={key:[] for key in KEYS};phase['dram']=[dram]
    (folder/'core_0.yaml').write_text(json.dumps(dict(iteration=1,execution=[phase])))
    op=dict(name='transfer',type='mpmd-general',iteration=1,file_prefix=str(folder))
    result=execute(chip,tmp_path/'run',dict(operator=[op]),dict(tensor=[tensor]))
    assert result['memory_access_bytes']==bursts*128
