"""Physical tensor regions and explicit graph edges for native ATLAS tasks.

A Region names actual per-core DRAM storage. Replication is represented by
multiple regions, never an implicit broadcast. The allocator reuses dead
activations; all weights and KV allocations remain live through the graph.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from math import isqrt, prod
from pathlib import Path

from .sim import KEYS, hardware, span, strides as layout
from .sim import execute, dump


@dataclass(frozen=True)
class Region:
    core: int
    name: str
    row: int
    col: int
    rows: int
    cols: int
    local_row: int = 0
    local_col: int = 0

    @property
    def area(self):
        return self.rows * self.cols

    def crop(self, row, col, rows, cols):
        assert self.row <= row and row+rows <= self.row+self.rows
        assert self.col <= col and col+cols <= self.col+self.cols
        return replace(self, row=row, col=col, rows=rows, cols=cols,
                       local_row=self.local_row+row-self.row,
                       local_col=self.local_col+col-self.col)

    def intersect(self, other):
        r, c = max(self.row, other.row), max(self.col, other.col)
        nr = min(self.row+self.rows, other.row+other.rows)-r
        nc = min(self.col+self.cols, other.col+other.cols)-c
        return self.crop(r, c, nr, nc) if min(nr, nc) > 0 else None

    def subtract(self, other):
        x = self.intersect(other)
        if x is None:
            return [self]
        r, c, h, w = self.row, self.col, self.rows, self.cols
        pieces = [(r, c, x.row-r, w), (x.row+x.rows, c, r+h-x.row-x.rows, w),
                  (x.row, c, x.rows, x.col-c), (x.row, x.col+x.cols, x.rows, c+w-x.col-x.cols)]
        return [self.crop(*p) for p in pieces if min(p[2:]) > 0]


def window(regions, row, col, rows, cols):
    """A logical view: change global coordinates without moving physical data."""
    box = Region(0, '', row, col, rows, cols)
    return [replace(x, row=x.row-row, col=x.col-col) for r in regions
            if (x := r.intersect(box)) is not None]


def unique(regions):
    """Choose one physical copy of each logical element, preserving rectangles."""
    result = []
    for region in regions:
        pieces = [region]
        for covered in result:
            pieces = [x for p in pieces for x in p.subtract(covered)]
        result.extend(pieces)
    return result


def transfers(source, target):
    """Partition destination exactly once; prefer a local replica when present."""
    result = []
    for dst in target:
        missing = [dst]
        for src in sorted(source, key=lambda x: (x.core != dst.core, abs(x.core-dst.core), x.core)):
            remaining = []
            for part in missing:
                overlap = part.intersect(src)
                if overlap is not None:
                    result.append((src.crop(overlap.row, overlap.col, overlap.rows, overlap.cols), overlap))
                remaining.extend(part.subtract(src))
            missing = remaining
            if not missing:
                break
        if missing:
            raise ValueError(f'Uncovered graph input: {missing[0]}')
    return result


def traffic(source, target, alias='bind', side=None):
    """Per-core element counts ``convert`` moves: dram read/write, NoC sent/received.

    With the mesh ``side``, each directed link's load is added under
    ``('link', (a, b))`` in the sent slot, since links, not only ports, bound
    a relayout's NoC time.

    Each moved piece is read where it is sent, as ``convert`` does. Same-core pieces
    are free when ``alias`` is ``'cover'`` and one local source region covers
    the whole target region (the consumer reads it in place), or ``'bind'``
    and a core's whole target comes from one local tensor at the same offsets
    (the consumer reuses it, as ``bind_input`` does); otherwise they are a
    local DRAM copy.
    """
    if alias == 'cover':
        target = [t for t in target if not any(r.core == t.core and (x := r.intersect(t)) is not None
                                               and x.area == t.area for r in source)]
    pairs = transfers(source, target); load = defaultdict(lambda: [0, 0, 0, 0]); links = Counter()
    aliased = set()
    if alias == 'bind':
        names = defaultdict(set)
        for src, dst in pairs:
            same = src.core == dst.core and (src.local_row, src.local_col) == (dst.local_row, dst.local_col)
            names[dst.core].add(src.name if same else None)
        aliased = {core for core, n in names.items() if len(n) == 1 and None not in n}
    for src, dst in pairs:
        if src.core == dst.core and dst.core in aliased:
            continue
        load[src.core][0] += src.area; load[dst.core][1] += src.area
        if src.core != dst.core:
            load[src.core][2] += src.area; load[dst.core][3] += src.area
            for link in (route(src.core, dst.core, side) if side else ()):
                links[link] += src.area
    for link, area in links.items():
        load.setdefault(('link', link), [0, 0, 0, 0])[2] = area
    return dict(load)


def route(src, dst, side):
    """Directed mesh links of BookSim's X-then-Y route."""
    (sy, sx), (dy, dx) = divmod(src, side), divmod(dst, side)
    path = []
    while sx != dx:
        nx = sx+(1 if dx > sx else -1); path.append((sy*side+sx, sy*side+nx)); sx = nx
    while sy != dy:
        ny = sy+(1 if dy > sy else -1); path.append((sy*side+sx, ny*side+sx)); sy = ny
    return path


def panelize(box, limit):
    """Split a (row, col, rows, cols) box into panels of at most limit elements."""
    row, col, rows, cols = box; width = min(cols, limit); height = max(1, limit//width)
    return [(row+y, col+x, min(height, rows-y), min(width, cols-x))
            for x in range(0, cols, width) for y in range(0, rows, height)]


class Timeline:
    """Use simulator times, hiding the native read/compute/write slot offsets."""
    def __init__(self, cores, flit_size):
        self.raw = [[] for _ in range(cores)]
        self.flit_size = flit_size
        self.fid = 0
        self.counts = dict(macs=0, vector_ops=0, dram_read_bytes=0, dram_write_bytes=0,
                           noc_data_bytes=0, noc_credit_bytes=0)

    def add(self, core, time, key, task):
        slot = time if key == 'dram' and not task['is_write'] else time-2 if key == 'dram' else time-1
        if slot < 0:
            raise ValueError('Task precedes its available inputs')
        while len(self.raw[core]) <= slot:
            self.raw[core].append({k: [] for k in KEYS})
        self.raw[core][slot][key].append(task)

    def memory(self, time, region, write=False, dtype=2):
        self.add(region.core, time, 'dram', dict(name=region.name, is_write=write,
                 access_base=[region.local_row, region.local_col], access_extent=[region.rows, region.cols],
                 access_stride_add=[1, 1], access_offset_add=[0, 0], init_iter=0, stride_iter=1, total_iter=1))
        self.counts['dram_write_bytes' if write else 'dram_read_bytes'] += region.area*dtype

    def buffer(self, core, time, size, write=False):
        if size:
            self.add(core, time, 'buffer_store' if write else 'buffer_load',
                     dict(name='sram', byte_count=size, is_write=write))

    def work(self, core, time, macs=0, vectors=0, read=0, write=0):
        if macs:
            if macs >= 2**31:
                raise ValueError('Matrix task exceeds native signed count')
            self.add(core, time, 'matrix', dict(name='matrix', mac_count=macs))
            self.counts['macs'] += macs
        if vectors:
            self.add(core, time, 'vector', dict(name='vector', vec_count=vectors))
            self.counts['vector_ops'] += vectors
        self.buffer(core, time, read)
        self.buffer(core, time, write, True)

    def send(self, time, src, dst, size, credit=False):
        if src == dst or not size:
            return
        flits = (size+self.flit_size-1)//self.flit_size
        packet = dict(name='credit' if credit else 'data', src=src, dst=dst,
                      flit_num=flits, init_flit_id=self.fid)
        self.fid += flits
        self.add(src, time, 'noc_tx', dict(packet, is_send=True))
        self.add(dst, time, 'noc_rx', dict(packet, is_send=False))
        self.counts['noc_credit_bytes' if credit else 'noc_data_bytes'] += flits*self.flit_size
        if not credit:
            self.buffer(src, time, size)
            self.buffer(dst, time, size, True)

    def finish(self):
        length = max(1, *(len(x) for x in self.raw))
        for core in self.raw:
            core.extend({k: [] for k in KEYS} for _ in range(length-len(core)))
        return self.raw


class Graph:
    def __init__(self, chip):
        self.chip = Path(chip).resolve()
        self.arch, self.capacity, _ = hardware(chip)
        self.cores = self.arch['core_num']
        self.sram = self.arch['core']['buffer']['buffer_size']*1024
        self.flit = self.arch['noc']['flit_size']
        self.tensors = [dict() for _ in range(self.cores)]
        self.live = [dict() for _ in range(self.cores)]
        self.peak = [0]*self.cores
        self.operations = []
        self.edges = []
        self.next_flit = 0

    def tensor(self, core, name, shape, strides=None, dtype=2, persistent=False, column=False):
        if name in self.tensors[core]:
            raise ValueError(f'Duplicate physical tensor {core}/{name}')
        strides = strides or layout(shape, dtype, column)
        size = span(shape, strides, dtype)
        address = (self.peak[core]+127)//128*128 if persistent else 0
        for start, end in sorted(self.live[core].values()):
            if address+size <= start:
                break
            address = max(address, (end+127)//128*128)
        if address+size > self.capacity:
            raise ValueError(f'Per-core DRAM capacity exceeded on core {core}')
        tensor = dict(name=name, shape=list(shape), strides=list(strides),
                      element_size=dtype, base_addr=address)
        self.tensors[core][name] = tensor
        self.live[core][name] = (address, address+size)
        self.peak[core] = max(self.peak[core], address+size)
        return tensor

    def release(self, regions, keep=()):
        protected = {(r.core, r.name) for r in keep}
        for r in regions:
            if (r.core, r.name) not in protected:
                self.live[r.core].pop(r.name, None)

    def timeline(self):
        return Timeline(self.cores, self.flit)

    def add(self, name, raw, metadata=None, kind='mpmd-general', operator=None):
        """Append one operator, renumbering its flits into a graph-unique range.

        Core/src is part of BookSim's identity; operators, including native
        collectives and extended kernels, still use disjoint ranges. Only
        packet dictionaries are copied; other tasks are shared, read-only.
        """
        largest = -1; offset = self.next_flit
        def copy(obj):
            nonlocal largest
            if isinstance(obj, dict):
                if 'init_flit_id' in obj:
                    largest = max(largest, obj['init_flit_id']+obj['flit_num']-1)
                    return dict(obj, init_flit_id=obj['init_flit_id']+offset)
                return {k: copy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [copy(v) for v in obj]
            return obj
        raw = copy(raw)
        self.next_flit += largest+1
        if self.next_flit >= 2**31:
            raise ValueError('Native flit identifier overflow')
        self.operations.append(dict(name=name, kind=kind, raw=raw, metadata=metadata or {}, operator=operator))

    def add_timeline(self, name, timeline, **metadata):
        self.add(name, timeline.finish(), dict(timeline.counts, **metadata))

    def allgather(self, name, groups):
        """ATLAS-style linear all-gather: first along mesh rows, then columns.

        Each group lists the replicas of one block on a mesh sub-grid. Replica
        (r, c) initially owns slab r*C+c of the block. As in ATLAS's Q
        all-gather, slabs move hop by hop in both directions, one per link and
        step; each step is cut into SRAM-bounded panels. All groups run together.
        """
        side = isqrt(self.cores); timeline = self.timeline()
        load = Counter(r.core for g in groups for r in g)
        limit = max(1, self.sram//(16*max(load.values(), default=1)))
        phases = ([], [])
        for group in groups:
            grid = {divmod(r.core, side): r for r in group}
            rows = sorted({y for y, _ in grid}); cols = sorted({x for _, x in grid})
            if len(grid) != len(group) or len(grid) != len(rows)*len(cols):
                raise ValueError('Replica group is not a mesh sub-grid')
            first = group[0]; axis = int(first.rows < first.cols)
            length = (first.rows, first.cols)[axis]; n, width = len(group), len(cols)
            bound = [length*i//n for i in range(n+1)]
            def slab(lo, hi):
                return ((first.row+lo, first.col, hi-lo, first.cols) if axis == 0 else
                        (first.row, first.col+lo, first.rows, hi-lo))
            for i, y in enumerate(rows):
                phases[0].append(([grid[y, x] for x in cols],
                                  [slab(bound[i*width+j], bound[i*width+j+1]) for j in range(width)]))
            # A one-column group has no row phase; let its column lines start at once.
            for j, x in enumerate(cols):
                phases[int(width > 1)].append(([grid[y, x] for y in rows],
                                  [slab(bound[i*width], bound[(i+1)*width]) for i in range(len(rows))]))
        start = 0
        for lines in phases:
            end = start
            for members, slabs in lines:
                if len(members) < 2:
                    continue
                panels = [panelize(box, limit) for box in slabs]
                count = max(map(len, panels))
                for origin, src in enumerate(members):
                    for q, box in enumerate(panels[origin]):
                        timeline.memory(start+q, src.crop(*box))
                        for direction in (1, -1):
                            prev = src
                            for hop in range(1, len(members)):
                                j = origin+direction*hop
                                if not 0 <= j < len(members):
                                    break
                                t = start+1+(hop-1)*count+q
                                timeline.send(t, prev.core, members[j].core, box[2]*box[3]*2)
                                timeline.memory(t+1, members[j].crop(*box), True)
                                prev = members[j]; end = max(end, t+2)
            start = end
        self.add_timeline(name, timeline, category='distribution', groups=len(groups))
        self.edges.append(dict(name=name, **timeline.counts, aliased=False))

    def convert(self, name, source, target):
        """Relayout: every target element is read once from its nearest source copy.

        Transfers are cut into SRAM-bounded panels and pipelined like the
        linear all-gather: read, send and write of successive panels overlap,
        and each core sends at most one and receives at most one panel per
        step, so a step costs one panel on every port. Each unicast is placed
        at the earliest step where both ends are free and reads its panel
        from DRAM itself (DRAM is far faster than one NoC port; SRAM keeps no
        panel across steps).
        """
        pairs = transfers(source, target)
        timeline = self.timeline(); limit = max(1, self.sram//16)
        moves = []
        for src, dst in pairs:
            if src.core == dst.core and src.name == dst.name and (src.local_row, src.local_col) == (dst.local_row, dst.local_col):
                continue
            for box in panelize((src.row, src.col, src.rows, src.cols), limit):
                moves.append((src.crop(*box), dst.crop(*box)))
        sending = defaultdict(set); receiving = defaultdict(set)
        for src, dst in moves:
            step = 0
            while step in sending[src.core] or step in receiving[dst.core]:
                step += 1
            sending[src.core].add(step); receiving[dst.core].add(step)
            timeline.memory(step, src)
            timeline.send(step+1, src.core, dst.core, src.area*2)
            timeline.memory(step+2, dst, True)
        if moves:
            self.add_timeline(name, timeline, category='layout', regions=len(pairs))
        record = dict(name=name, **timeline.counts, aliased=not moves)
        self.edges.append(record)
        return target

    def simulate(self, out, cache=None, config_key=None, timeout=7200):
        out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
        operators = []
        for i, entry in enumerate(self.operations):
            folder = out/f'{i:02}_{entry["name"]}'; folder.mkdir(exist_ok=True)
            raw = entry['raw']
            op = deepcopy(entry['operator']) or dict(type=entry['kind'], iteration=len(raw[0]))
            op.update(name=entry['name'], file_prefix=str(folder))
            for core, task in enumerate(raw):
                payload = dict(iteration=len(task), execution=task) if entry['kind'] == 'mpmd-general' else task
                (folder/f'core_{core}.yaml').write_text(json.dumps(payload))
            operators.append(op)
        manifest = dict(peak_dram_bytes=self.peak, dram_capacity_per_core=self.capacity,
                        edges=self.edges, operators=[{k:v for k,v in op.items() if k not in ('raw','operator')}
                                                   for op in self.operations])
        (out/'graph.json').write_text(json.dumps(manifest, indent=2))
        placement = dict(core_tensor=[dict(core_id=i, tensor=list(t.values())) for i,t in enumerate(self.tensors)])
        dump(out/'operators.yaml',dict(operator=operators)); dump(out/'placement.yaml',placement)
        if cache is None:
            result = execute(self.chip,out,dict(operator=operators),placement,timeout)
        else:
            from .cache import task_digest
            digest = task_digest(out,self.cores)
            result,source = cache.get(digest,config_key)
            if result is None:
                result = execute(self.chip,out,dict(operator=operators),placement,timeout)
            else:
                dump(out/'result.json',result)
            cache.put(digest,out,source)
        return dict(result=result, graph=manifest)
