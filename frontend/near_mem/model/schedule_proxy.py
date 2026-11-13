"""Event-level cost model for compiled GEMM nomination; no cycle/flit simulation.

Standalone measurements and attention proposal selection remain separate.
Whole messages reserve directed XY links and endpoint bandwidth. Independent
cores advance on their own compute, DMA deadlines and receive dependencies.
The fluid cut-through network omits finite buffers and VC arbitration.
Hardware rates and exposed topology parameters come from the architecture;
hop_cycles may instead be supplied by black-box latency calibration. The
default DRAM service reuses the pre-existing microbenchmark calibration, not
controller command timing. memory_factory is an optional per-core service
model accepting deduplicated read/write burst sets; no backend is imported.
"""
from collections import defaultdict
from heapq import heappop, heappush
from math import ceil, isqrt, prod
import re

from ..gemm.space import Proxy
from ..sim import hardware
from ..tasks import route


class ScheduleProxy:
    def __init__(self, chip, *, hop_cycles=None, memory_factory=None):
        self.chip = chip
        self.backend = Proxy(chip)
        self.arch = self.backend.arch
        _, _, paths = hardware(chip)
        config = paths[2].read_text()
        def setting(name, default):
            found = re.search(r'(?m)^\s*'+name+r'\s*=\s*(\d+)\s*;', config)
            return int(found[1]) if found else default
        self.hop = (setting('routing_delay', 1) + setting('vc_alloc_delay', 1)
                    + setting('sw_alloc_delay', 1) + setting('st_prepare_delay', 0)
                    + setting('st_final_delay', 1) + 1)
        if hop_cycles is not None:
            if hop_cycles < 0:
                raise ValueError('Hop latency must be nonnegative')
            self.hop = hop_cycles
        self.memory_factory = memory_factory
        if self.arch['noc']['topology'] != 'mesh':
            raise ValueError('Schedule proxy requires an XY mesh')
        self._memory_cache = {}
        self._transaction_cache = {}

    def memory(self, tasks, tensors):
        """Batched channel/row service, using actual tensor addresses/strides."""
        b = self.backend
        if tensors is None:
            return sum(prod(t['access_extent'][:-1])*ceil(t['access_extent'][-1]*2/b.burst)
                       *b.burst/b.dram_bw for t in tasks)
        signature = tuple((t['is_write'], tensors[t['name']]['element_size'], tensors[t['name']]['base_addr'],
                           tuple(tensors[t['name']]['strides']), tuple(t['access_base']),
                           tuple(t['access_extent'])) for t in tasks)
        if signature in self._memory_cache:
            return self._memory_cache[signature]
        bursts = defaultdict(int); rows = defaultdict(set)
        bits = b.channels.bit_length()-1; low = b.interleave
        transactions = self.transactions(tasks,tensors)
        for addresses in transactions.values():
            for address in addresses:
                channel = (address>>low)%b.channels
                row = (((address>>(low+bits))<<low)|(address&((1<<low)-1)))//b.columns
                bursts[channel] += 1; rows[channel].add(row)
        if not bursts:
            return 0.0
        value = max(sum(bursts.values())*b.burst/b.dram_bw,
                    max(n*b.burst/b.channel_bw+b.row_cost*len(rows[c]) for c,n in bursts.items())-b.overlap)
        # Cores often share identical physical access geometry. Cache only
        # service costs, never event timing or state-dependent dependencies.
        if len(self._memory_cache) >= 4096:
            self._memory_cache.clear()
        self._memory_cache[signature] = value
        return value

    def transactions(self, tasks, tensors):
        b = self.backend
        signature = tuple((t['is_write'], tensors[t['name']]['element_size'], tensors[t['name']]['base_addr'],
                           tuple(tensors[t['name']]['strides']), tuple(t['access_base']),
                           tuple(t['access_extent'])) for t in tasks)
        if signature in self._transaction_cache:
            return self._transaction_cache[signature]
        transactions = {False: set(), True: set()}
        for task in tasks:
            tensor = tensors[task['name']]
            dtype = tensor['element_size']; strides = tensor['strides']
            ext = task['access_extent']; base = task['access_base']
            if len(ext) != 2 or min(strides) != 1:
                raise ValueError('Schedule proxy currently requires contiguous-axis rank-2 tensors')
            axis = 0 if strides[0] == 1 else 1
            start = tensor['base_addr'] + sum(x*s for x,s in zip(base,strides))*dtype
            addresses = set()
            for line in range(ext[1-axis]):
                at = start+line*strides[1-axis]*dtype
                addresses.update(range(at//b.burst, (at+ext[axis]*dtype+b.burst-1)//b.burst))
            transactions[task['is_write']].update(addresses)
        # The target's batch interface deduplicates reads and writes separately;
        # a read and a write to the same burst remain two requests.
        if len(self._transaction_cache) >= 64:
            self._transaction_cache.clear()
        self._transaction_cache[signature] = transactions
        return transactions

    def tasks(self, raw, tensors=None, *, channel_memory=True, network=True):
        """Score one general operator; ``network=False`` removes contention only.

        Passing graph tensors enables the existing calibrated channel model.
        Without placements, DRAM falls back to the older peak-bandwidth model.
        Neither mode is a safe lower bound for pruning candidates.
        """
        if not raw or not raw[0]:
            raise ValueError('Schedule must contain at least one core and phase')
        cores = len(raw); length = len(raw[0])+2
        if isqrt(cores)**2 != cores or any(len(row)+2 != length for row in raw):
            raise ValueError('Schedule requires a square mesh and equal-length core rows')
        work = [[1.0]*length for _ in raw]
        memory = [[[] for _ in range(length)] for _ in raw]
        deadline = [[length]*length for _ in raw]
        sends = [[[] for _ in range(length)] for _ in raw]
        receives = [[[] for _ in range(length)] for _ in raw]
        arch = self.arch; buffer = arch['core']['buffer']
        for core, row in enumerate(raw):
            for t, phase in enumerate(row):
                work[core][t+1] = max(1,
                    ceil(sum(x['mac_count'] for x in phase['matrix'])/arch['core']['matrix']['mac_num']),
                    ceil(sum(x['vec_count'] for x in phase['vector'])/arch['core']['vector']['vec_num']),
                    ceil(sum(x['byte_count'] for x in phase['buffer_load'])/buffer['read_bw']),
                    ceil(sum(x['byte_count'] for x in phase['buffer_store'])/buffer['write_bw']))
                for task in phase['dram']:
                    at = t+2 if task['is_write'] else t
                    memory[core][at].append(task)
                    due = task.get('ready_iter',at+1)
                    if not at < due <= length:
                        raise ValueError('DRAM deadline must follow issue and stay within the operator')
                    deadline[core][at] = min(deadline[core][at],due)
                sends[core][t+1] = phase['noc_tx']
                receives[core][t+1] = phase['noc_rx']
        busy = [0.0]*cores; ready = [[0.0]*(length+1) for _ in raw]
        stage = [-1]*cores; pending = [set() for _ in raw]
        finished = [0.0]*cores; local_ready = [0.0]*cores
        arrived = set(); link_free = defaultdict(float)
        banks = None
        if self.memory_factory is not None:
            if tensors is None:
                raise ValueError('Custom memory service requires physical tensor placements')
            banks = [self.memory_factory(self.chip,self.backend) for _ in raw]
        events = []; counter = 0
        def push(at, kind, data):
            nonlocal counter
            counter += 1; heappush(events,(at,counter,kind,data))
        def key(task):
            return task['src'],task['dst'],task['init_flit_id']
        def advance(core, at):
            if not pending[core] and at >= local_ready[core]:
                push(at, 'start', (core,stage[core]+1))
        for core in range(cores):
            push(0.0,'start',(core,0))
        while events:
            at, _, kind, data = heappop(events)
            if kind == 'start':
                core,t = data
                if t <= stage[core]:
                    continue
                stage[core] = t
                if t == length:
                    finished[core] = max(at,busy[core]); continue
                duration = (banks[core].service(self.transactions(memory[core][t],tensors[core])) if banks else
                            self.memory(memory[core][t],tensors[core] if tensors is not None and channel_memory else None))
                if duration:
                    busy[core] = max(at,busy[core])+duration
                    due = deadline[core][t]
                    ready[core][due] = max(ready[core][due],busy[core])
                local_ready[core] = max(at+work[core][t],ready[core][t+1])
                pending[core] = {key(x) for x in receives[core][t]}-arrived
                push(local_ready[core], 'work', core)
                for task in sends[core][t]:
                    k = key(task)
                    path = [('tx',task['src']),*route(task['src'],task['dst'],isqrt(cores)),('rx',task['dst'])]
                    push(at,'packet',(k,task['flit_num'],path,0))
            elif kind == 'work':
                advance(data,at)
            elif kind == 'packet':
                k,n,path,i = data
                start = max(at,link_free[path[i]]) if network else at
                link_free[path[i]] = start+n
                if i+1 == len(path):
                    push(start+n+1,'receive',k)
                else:
                    push(start+(1 if i==0 else self.hop),'packet',(k,n,path,i+1))
            elif kind == 'receive':
                arrived.add(data); core = data[1]
                if data in pending[core]:
                    pending[core].remove(data); advance(core,at)
        if any(t != length for t in stage):
            raise ValueError('Unmatched NoC dependency in schedule proxy')
        return max(finished)
