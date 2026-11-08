"""Attention proposal score from the shared compiler's task schedule.

Compute/SRAM use hardware rates; NoC uses busiest ports and XY mesh links;
DRAM uses burst-rounded bytes at peak bandwidth on a per-core FIFO. Reads
wait at their declared first use, including cross-iteration prefetch. This
is a ranking proxy, not Ramulator/BookSim or a measured cycle count.
SRAM traffic comes only from explicit buffer tasks, like the ATLAS backend;
DRAM reads/writes do not infer extra staging-port accesses. This shared DMA
staging abstraction does not remove the compiler's buffer capacity checks.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial
from math import ceil, isqrt, prod

from ..gemm.space import Proxy
from ..search import shortlist
from ..tasks import route
from .attention import build_attention


def _score(chip, problem, placed, plan):
    return AttentionProxy(chip).score(problem, plan, placed)


class AttentionProxy:
    def __init__(self, chip):
        self.chip = chip
        self.backend = Proxy(chip)
        self.arch = self.backend.arch

    @lru_cache(maxsize=4096)
    def score(self, problem, plan, placed=False):
        graph = build_attention(problem, self.chip, plan, placed)
        return sum(self.tasks(op['raw']) for op in graph.operations)

    def __call__(self, problem, layout, plan):
        return self.score(problem, plan)

    def select(self, problem, candidates, budget=32, placed=False, workers=1):
        """Top budget plus one per spatial/output family, so edges retain choices.

        Zero budget is the exhaustive validation path. All methods use the
        same score and budget; final choices always come from the simulator.
        Scores are independent, so ``workers`` processes compute them; the
        selection is identical to the serial one.
        """
        candidates = list(dict.fromkeys(candidates))
        if budget <= 0:return candidates
        if workers > 1 and len(candidates) > 1:
            with ProcessPoolExecutor(min(workers, len(candidates))) as pool:
                scores = dict(zip(candidates, pool.map(partial(_score, self.chip, problem, placed), candidates)))
        else:
            scores = {p: self.score(problem, p, placed) for p in candidates}
        ranked,_ = shortlist(lambda _: iter(candidates),scores.__getitem__,budget,
                             family=lambda p:(p.dataflow,p.query_parts,p.context_parts,p.output))
        return [p for p,_ in ranked]

    def tasks(self, raw):
        arch = self.arch; cores = len(raw); length = len(raw[0])+2
        matrix = arch['core']['matrix']['mac_num']; vector = arch['core']['vector']['vec_num']
        buffer = arch['core']['buffer']
        # Same offsets as the backend: reads at raw t, compute at t+1,
        # writes at t+2. Iterations conservatively share a phase boundary;
        # exact per-core skew and router arbitration are left to ATLAS.
        work = [1.0]*length; links = [Counter() for _ in range(length)]
        ports = [Counter() for _ in range(length)]
        memory = [[0.0]*length for _ in raw]
        deadlines = [[length]*length for _ in raw]
        for core,row in enumerate(raw):
            for t,phase in enumerate(row):
                work[t+1] = max(work[t+1],
                    ceil(sum(x['mac_count'] for x in phase['matrix'])/matrix),
                    ceil(sum(x['vec_count'] for x in phase['vector'])/vector),
                    ceil(sum(x['byte_count'] for x in phase['buffer_load'])/buffer['read_bw']),
                    ceil(sum(x['byte_count'] for x in phase['buffer_store'])/buffer['write_bw']))
                for task in phase['dram']:
                    at = t+2 if task['is_write'] else t
                    # Widths/row tails round to physical bursts. Padding and
                    # channel conflicts are resolved by Ramulator, not fitted.
                    ext = task['access_extent']; burst = self.backend.burst
                    size = prod(ext[:-1])*ceil(ext[-1]*2/burst)*burst
                    memory[core][at] += size/self.backend.dram_bw
                    deadlines[core][at] = min(deadlines[core][at],task.get('ready_iter',at+1))
                for task in phase['noc_tx']:
                    n = task['flit_num']; a,b = task['src'],task['dst']
                    ports[t+1]['tx',a] += n; ports[t+1]['rx',b] += n
                    for edge in route(a,b,isqrt(cores)):links[t+1][edge] += n
        clock = 0.0; busy = [0.0]*cores; ready = [0.0]*(length+1)
        for t in range(length):
            for core in range(cores):
                if memory[core][t]:
                    busy[core] = max(clock,busy[core])+memory[core][t]
                    due = deadlines[core][t]
                    ready[due] = max(ready[due],busy[core])
            duration = max(work[t],max(ports[t].values(),default=0),max(links[t].values(),default=0))
            clock = max(clock+duration,ready[t+1])
        return max(clock,*busy)
