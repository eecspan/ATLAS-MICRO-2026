"""FlashGEMM's dataflow on the shared ATLAS backend.

A, B and C keep one DRAM copy in the standard block layout (``GlobalLayout``):
core (r, c) of the ``s x s`` mesh holds block (r, c) of each. A K tile takes
an equal slice of every K block, so each core contributes one slab and the
mesh lines all-gather the tile into SRAM with the linear, bidirectional
schedule of ATLAS's collectives while the previous step computes. ``os``
keeps C stationary and gathers A along rows and B along columns, per M panel
of the C block. ``ws`` keeps B stationary: A slabs are transposed to the row
of their K block and gathered along it, and partial C is reduce-scattered
along columns. As in the paper, a step holds the whole local N extent; only
when no such plan fits the SRAM (small-SRAM chips) do we split N into panels,
re-gathering A for each, which is the minimal change that makes it run. The
paper's skipped ring and DRAM bypass are hardware and are not modelled.
"""
from dataclasses import asdict, replace
from itertools import product
from math import ceil, isqrt

from ..tasks import Timeline, coalesce
from .layout import GlobalLayout
from .plan import FlashPlan, Problem, execution_problem
from .space import divisors


class Geometry:
    """Per-step extents. ``ms`` rows and ``ks`` K columns are one core's slab."""
    def __init__(self, problem, plan):
        problem = execution_problem(problem, plan)
        s = self.side = isqrt(plan.cores); m, n, k = problem.shape; self.dtype = problem.dtype_bytes
        if s*s != plan.cores or m % s or n % s or k % s:
            raise ValueError('FlashGEMM needs a square mesh and mesh-divisible dimensions')
        self.lm, self.ln, self.lk = m//s, n//s, k//s
        mt, kt = plan.m_tile, plan.k_tile; self.nt = nt = plan.n_tile or self.ln
        if self.ln % nt or plan.n_tile and plan.b_resident:
            raise ValueError('FlashGEMM N panels must divide the local N and not keep B resident')
        self.npanels = self.ln//nt
        if plan.dataflow == 'os':
            ok = self.lm % mt == 0 and kt % s == 0 and self.lk % (kt//s) == 0
            self.ms, self.ks, self.panels, self.ktiles = mt, kt//s, self.lm//mt, k//kt
        elif plan.dataflow == 'ws':
            ok = mt % s == 0 and self.lm % (mt//s) == 0 and self.lk % kt == 0
            self.ms, self.ks, self.panels, self.ktiles = mt//s, kt, self.lm//(mt//s), self.lk//kt
        else:
            raise ValueError('Unknown FlashGEMM dataflow')
        if not ok or plan.b_resident and plan.dataflow != 'ws':
            raise ValueError('FlashGEMM tiles must divide the per-core extents')
        # One round per gather hop (+ the WS transpose); the compute of a step
        # is spread over a round so every hop overlaps a share of it.
        self.round = max(1, s-1+(plan.dataflow == 'ws'))
        t = self.dtype; a, b, c = mt*kt*t, kt*nt*t, mt*nt*t
        resident = self.lk*self.ln*t if plan.b_resident else 2*b
        self.sram = 2*a+(2*b if plan.dataflow == 'os' else resident)+2*c+(2*c//s if plan.dataflow == 'ws' else 0)


def candidates(problem, cores, sram, tile_min=16, tile_policy='powers'):
    """FlashGEMM plans; zero-pad mesh tails to keep its regular equal blocks.

    Padded MACs and DRAM traffic are included in both proxy ranking and the
    simulator. Padding is only used to make an otherwise invalid mesh shape
    executable; the dataflow and tile search remain FlashGEMM's own.
    """
    side = isqrt(cores)
    if side*side != cores:
        return iter(())
    logical = problem.shape
    execution = tuple(((n+side-1)//side)*side for n in logical)
    work = Problem(*execution, problem.dtype_bytes)
    plans = list(_candidates(work, cores, sram, tile_min, tile_policy, [None]))
    if plans or work.n % side:
        selected = plans
    else:
        local_n = work.n//side
        values = [v for v in divisors(local_n) if v >= min(local_n, tile_min) and v < local_n]
        panels = values if tile_policy == 'all' else [v for v in values if v & (v-1) == 0]
        selected = list(_candidates(work, cores, sram, tile_min, tile_policy, panels))
    if execution != logical:
        selected = [replace(p, logical_shape=logical, execution_shape=execution) for p in selected]
    return iter(selected)


def _candidates(problem, cores, sram, tile_min, tile_policy, panels):
    s = isqrt(cores); m, n, k = problem.shape
    def tiles(n):
        values = [v for v in divisors(n) if v >= min(n, tile_min)]
        return values if tile_policy == 'all' else [v for v in values if v & (v-1) == 0 or v == n]
    if s*s != cores or m % s or n % s or k % s:
        return
    for dataflow, mts, kts in (('os', tiles(m//s), [s*x for x in tiles(k//s)]),
                               ('ws', [s*x for x in tiles(m//s)], tiles(k//s))):
        for resident in (False, True) if dataflow == 'ws' else (False,):  # families arrive contiguously
            for mt, kt, nt in product(mts, kts, panels):
                plan = FlashPlan(dataflow, mt, kt, resident, cores, n_tile=nt)
                try:
                    g = Geometry(problem, plan)
                except ValueError:
                    continue
                if g.sram <= sram and not (resident and g.panels == 1):
                    yield plan


def score(problem, plan, mac, dram_bw, flit):
    """Proxy: per round the slowest of compute share, DRAM slab reads and one gather hop."""
    problem = execution_problem(problem, plan)
    g = Geometry(problem, plan); t = g.dtype; s = g.side; R = g.round
    mt, kt, nt = plan.m_tile, plan.k_tile, g.nt
    compute = ceil(mt*nt*kt/mac)
    if plan.dataflow == 'os':
        reads, hop = (g.ms*g.ks+g.ks*nt)*t, max(g.ms*g.ks, g.ks*nt)*t
    else:
        reads, hop = g.ms*g.ks*t+(0 if plan.b_resident else kt*nt*t), g.ms*g.ks*t
    step = R*max(compute/R, reads/R/dram_bw, hop/flit)+R
    panels = g.panels*g.npanels
    output = g.lm*g.ln*t/dram_bw+(panels*(s-1)*g.ms*nt*t/flit if plan.dataflow == 'ws' else 0)
    resident = g.lk*g.ln*t/dram_bw if plan.b_resident else 0
    return (panels*g.ktiles+1)*step+output+resident


class FlashProgram:
    """Placement, boundary regions and native tasks of one plan."""
    def __init__(self, problem, plan, sram, dram):
        from ..sim import span, strides
        problem = execution_problem(problem, plan)
        self.problem = problem; self.plan = plan; g = self.g = Geometry(problem, plan)
        if g.sram > sram:
            raise ValueError(f'SRAM capacity exceeded: {g.sram} > {sram}')
        self.sram_bytes = g.sram; self.layout = GlobalLayout(problem, plan.cores)
        self.placements = []
        for core in range(plan.cores):
            address = 0; tensors = []
            for name in 'ABC':
                shape = list(self.layout.block(name)); address = (address+127)//128*128
                pitch = strides(shape, g.dtype, name == 'B')
                tensors.append(dict(name=name, shape=shape, strides=pitch, element_size=g.dtype, base_addr=address))
                address += span(shape, pitch, g.dtype)
            if address > dram:
                raise ValueError('Aligned DRAM placement exceeds capacity')
            self.placements.append(dict(core_id=core, tensor=tensors))

    suffix = ''

    @property
    def gate_tile(self):
        return self.g.nt

    def regions(self, tensor):
        regions=self.layout.regions(tensor)
        if self.plan.logical_shape is None:
            return regions
        m,n,k=self.plan.logical_shape;rows,cols={'A':(m,k),'B':(k,n),'C':(m,n)}[tensor]
        clipped=[]
        for r in regions:
            nr=min(r.rows,max(0,rows-r.row));nc=min(r.cols,max(0,cols-r.col))
            if nr and nc:clipped.append(r.crop(r.row,r.col,nr,nc))
        return coalesce(clipped)

    def steps(self):
        """``(q, j, x, t)``: M panel, N panel, K tile and the round its slabs are read."""
        g = self.g
        for q in range(g.panels):
            for j in range(g.npanels):
                for x in range(g.ktiles):
                    yield q, j, x, ((q*g.npanels+j)*g.ktiles+x)*g.round

    def slabs(self, q, j, x):
        """Local A and B windows ``(base, extent)`` every core reads for a step.

        ``os``: K tile x is slice x of every K block; ``ws``: slice x of the
        core's own K block. Either way the windows are local offsets.
        """
        g = self.g
        return ([q*g.ms, x*g.ks], [g.ms, g.ks]), ([x*g.ks, j*g.nt], [g.ks, g.nt])

    def lower(self, flit, final_ops=None):
        p = self.plan; g = self.g; s = g.side; t = g.dtype; R = g.round
        tl = Timeline(p.cores, flit); mt, kt, nt = p.m_tile, p.k_tile, g.nt
        row = lambda r: [r*s+x for x in range(s)]; col = lambda c: [y*s+c for y in range(s)]
        def gather(time, members, size, name):
            # Hop h: member j forwards to each neighbour the slab of origin j-+(h-1).
            for h in range(1, s):
                for j, src in enumerate(members):
                    for d in (1, -1):
                        if 0 <= j+d < s and 0 <= j-d*(h-1) < s:
                            tl.send(time+h, src, members[j+d], size, name)
        def store(core, time, base, extent, key):
            action = dict(op='store', tensor='C', final=True, key=key, shape=list(extent), base=list(base),
                          extent=list(extent))
            if final_ops is not None:
                action, ops, read, written = final_ops(action)
                tl.work(core, time-1, vectors=ops, read=read, write=written)
            tl.access(core, time, action['tensor'], action['base'], action['extent'], True, t)
        work = dict(macs=mt*nt*kt, vectors=mt*nt, read=(mt*kt+kt*nt+mt*nt)*t, write=2*mt*nt*t)
        share = lambda value, i: value*(i+1)//R-value*i//R
        lead = int(p.dataflow == 'ws')  # the transpose round precedes the row gather
        for q, j, x, start in self.steps():
            a, b = self.slabs(q, j, x)
            for core in range(p.cores):
                tl.access(core, start, 'A', *a, dtype=t)
                if not (p.b_resident and q):
                    tl.access(core, start, 'B', *b, dtype=t)
                for i in range(R):
                    tl.work(core, start+R+1+i, **{key: share(v, i) for key, v in work.items()})
            slab = g.ms*g.ks*t
            if p.dataflow == 'os':
                for r in range(s):
                    gather(start, row(r), slab, 'A'); gather(start, col(r), g.ks*nt*t, 'B')
            else:
                for core in range(p.cores):
                    r, c = divmod(core, s)
                    tl.send(start+1, core, c*s+r, slab, 'A')  # to the row of its K block
                for r in range(s):
                    gather(start+lead, row(r), slab, 'A')
        for q, j in product(range(g.panels), range(g.npanels)):
            done = ((q*g.npanels+j+1)*g.ktiles+1)*R  # the panel's last compute round
            for core in range(p.cores):
                r, c = divmod(core, s)
                store(core, done+1+s*(p.dataflow == 'ws'), [q*g.ms, j*nt], [g.ms, nt],
                      (r*g.panels+q, c*g.npanels+j))
            if p.dataflow == 'ws':
                # Linear reduce-scatter of the panel's s chunks along each column.
                chunk = g.ms*nt
                for c in range(s):
                    members = col(c)
                    for h in range(1, s):
                        for j in range(h):
                            for src, dst in ((members[j], members[j+1]), (members[s-1-j], members[s-2-j])):
                                tl.send(done+h, src, dst, chunk*t, 'C')
                                tl.work(dst, done+h+1, vectors=chunk, read=2*chunk*t, write=chunk*t)
        raw = tl.finish()
        assert tl.counts['macs'] == self.problem.m*self.problem.n*self.problem.k
        manifest = dict(problem=asdict(self.problem), plan=asdict(p), sram_bytes=self.sram_bytes,
                        phases=len(raw[0]), macs=tl.counts['macs'], vector_ops=tl.counts['vector_ops'],
                        logical_dram_read_bytes=tl.counts['dram_read_bytes'],
                        logical_dram_write_bytes=tl.counts['dram_write_bytes'], noc_bytes=tl.counts['noc_data_bytes'])
        return raw, manifest
