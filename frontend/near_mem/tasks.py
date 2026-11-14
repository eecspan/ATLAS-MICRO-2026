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


def coalesce(regions):
    """Merge regions that are adjacent both logically and in local storage."""
    regions = list(regions)
    for vertical in (False, True):
        result = []
        def key(r):
            return ((r.core, r.name, r.col, r.cols, r.local_col, r.local_row-r.row, r.row) if vertical else
                    (r.core, r.name, r.row, r.rows, r.local_row, r.local_col-r.col, r.col))
        for r in sorted(regions, key=key):
            if result:
                p = result[-1]
                if key(p)[:-1] == key(r)[:-1] and ((p.row+p.rows == r.row) if vertical else (p.col+p.cols == r.col)):
                    result[-1] = replace(p, rows=p.rows+r.rows) if vertical else replace(p, cols=p.cols+r.cols)
                    continue
            result.append(r)
        regions = result
    return regions


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


def transfers(source, target, order=None):
    """Partition destination exactly once; prefer a local replica when present.

    ``order`` (``sources_by_core(source)``) may be shared by repeated calls
    over one source."""
    order = order or sources_by_core(source)
    result = []
    for dst in target:
        missing = [dst]
        top, bottom, left, right = dst.row, dst.row+dst.rows, dst.col, dst.col+dst.cols
        for src in order(dst.core):
            sr, sc = src.row, src.col; er, ec = sr+src.rows, sc+src.cols
            if sr >= bottom or er <= top or sc >= right or ec <= left:
                continue  # misses the destination, hence every missing piece
            remaining = []
            for part in missing:
                r, c = max(part.row, sr), max(part.col, sc)
                nr, nc = min(part.row+part.rows, er)-r, min(part.col+part.cols, ec)-c
                if min(nr, nc) <= 0:
                    remaining.append(part)
                    continue
                overlap = part.crop(r, c, nr, nc)
                result.append((src.crop(r, c, nr, nc), overlap))
                pr, pc, h, w = part.row, part.col, part.rows, part.cols
                pieces = [(pr, pc, r-pr, w), (r+nr, pc, pr+h-r-nr, w), (r, pc, nr, c-pc), (r, c+nc, nr, pc+w-c-nc)]
                remaining.extend(part.crop(*p) for p in pieces if min(p[2:]) > 0)
            missing = remaining
            if not missing:
                break
        if missing:
            raise ValueError(f'Uncovered graph input: {missing[0]}')
    return result


def sources_by_core(source):
    """``core -> sources`` in ``transfers`` preference: local first, then by core distance."""
    cache = {}
    def order(core):
        if core not in cache:
            cache[core] = sorted(source, key=lambda x: (x.core != core, abs(x.core-core), x.core))
        return cache[core]
    return order


def traffic(source, target, alias='bind', side=None):
    """Per-phase, per-core element counts ``convert`` moves: dram read/write, NoC sent/received.

    The first phase is the unicast scatter; with the mesh ``side``, each
    directed link's load is added under ``('link', (a, b))`` in the sent
    slot. Then come the hops of the row and column phases of the replica
    all-gather, whose line schedule keeps every link at one slab per hop.

    Same-core pieces are free when ``alias`` is ``'cover'`` and one local
    source region covers the whole target region (the consumer reads it in
    place), or ``'bind'`` and a core's whole target comes from one local
    tensor at the same offsets (the consumer reuses it, as ``bind_input``
    does); otherwise they are a local DRAM copy.
    """
    if alias == 'cover':
        target = [t for t in target if not any(r.core == t.core and (x := r.intersect(t)) is not None
                                               and x.area == t.area for r in source)]
    aliased = set()
    if alias == 'bind':
        names = defaultdict(set)
        for src, dst in transfers(source, target):
            same = src.core == dst.core and (src.local_row, src.local_col) == (dst.local_row, dst.local_col)
            names[dst.core].add(src.name if same else None)
        aliased = {core for core, n in names.items() if len(n) == 1 and None not in n}
    pairs, groups = conversion(source, [t for t in target if t.core not in aliased], side)
    load = defaultdict(lambda: [0, 0, 0, 0]); links = Counter()
    for src, dst in pairs:
        load[src.core][0] += src.area; load[dst.core][1] += src.area
        if src.core != dst.core:
            load[src.core][2] += src.area; load[dst.core][3] += src.area
            for link in (route(src.core, dst.core, side) if side else ()):
                links[link] += src.area
    for link, area in links.items():
        load.setdefault(('link', link), [0, 0, 0, 0])[2] = area
    phases = [dict(load)]
    for lines in gather_lines(groups, side):
        # Lockstep hops: at hop t member j receives the slabs of j-t and j+t.
        for hop in range(1, max((len(m) for m, _ in lines), default=1)):
            load = defaultdict(lambda: [0, 0, 0, 0])
            for members, slabs in lines:
                areas = [box[2]*box[3] for box in slabs]
                for j, member in enumerate(members):
                    got = sum(areas[i] for i in (j-hop, j+hop) if 0 <= i < len(members))
                    load[member.core] = [x+y for x, y in zip(load[member.core],
                                                             (areas[j]*(hop == 1), got, got, got))]
            phases.append(dict(load))
    return phases


def subgrid(group, side):
    """``(grid, rows, cols)`` when the regions sit on a full mesh sub-grid, else None."""
    grid = {divmod(r.core, side): r for r in group}
    rows = sorted({y for y, _ in grid}); cols = sorted({x for _, x in grid})
    return (grid, rows, cols) if len(grid) == len(group) == len(rows)*len(cols) else None


def gather_lines(groups, side):
    """Row and column phases of the ATLAS linear all-gather: ``[(members, slabs)]`` each.

    Each group lists the replicas of one block on a mesh sub-grid. Replica
    (r, c) initially owns slab r*C+c; the row phase completes each row's
    slabs, the column phase the block. A one-column group has no row phase,
    so its column lines run in the first phase.
    """
    phases = ([], [])
    for group in groups:
        found = subgrid(group, side)
        if found is None:
            raise ValueError('Replica group is not a mesh sub-grid')
        grid, rows, cols = found
        first = group[0]; axis = int(first.rows < first.cols)
        length = (first.rows, first.cols)[axis]; n, width = len(group), len(cols)
        bound = [length*i//n for i in range(n+1)]
        def slab(lo, hi):
            return ((first.row+lo, first.col, hi-lo, first.cols) if axis == 0 else
                    (first.row, first.col+lo, first.rows, hi-lo))
        if width > 1:
            for i, y in enumerate(rows):
                phases[0].append(([grid[y, x] for x in cols],
                                  [slab(bound[i*width+j], bound[i*width+j+1]) for j in range(width)]))
        for x in cols if len(rows) > 1 else ():
            phases[int(width > 1)].append(([grid[y, x] for y in rows],
                                           [slab(bound[i*width], bound[(i+1)*width]) for i in range(len(rows))]))
    return phases


def conversion(source, target, side):
    """Plan of a relayout: unicast ``(source, target)`` pairs and replica groups.

    A target its core cannot fill from local copies is a replica when other
    such targets hold the same block on a mesh sub-grid. Like the native
    collectives, each block then crosses the NoC once: every replica receives
    one slab by unicast and the linear all-gather completes the block. All
    other targets are unicast from their nearest copy.
    """
    pairs = []; pending = defaultdict(list); groups = []; order = sources_by_core(source)
    for t in target:
        own = transfers(source, [t], order)
        if all(s.core == t.core for s, _ in own):
            pairs += own
        else:
            pending[t.row, t.col, t.rows, t.cols].append(t)
    for members in pending.values():
        if len(members) > 1 and side and subgrid(members, side) and max(members[0].rows, members[0].cols) >= len(members):
            groups.append(members)
        else:
            pairs += transfers(source, members, order)
    for lines in gather_lines(groups, side)[:1]:
        for members, slabs in lines:
            pairs += transfers(source, [m.crop(*box) for m, box in zip(members, slabs)], order)
    return pairs, groups


def route(src, dst, side):
    """Directed mesh links of BookSim's X-then-Y route."""
    (sy, sx), (dy, dx) = divmod(src, side), divmod(dst, side)
    path = []
    while sx != dx:
        nx = sx+(1 if dx > sx else -1); path.append((sy*side+sx, sy*side+nx)); sx = nx
    while sy != dy:
        ny = sy+(1 if dy > sy else -1); path.append((sy*side+sx, ny*side+sx)); sy = ny
    return path


def panelize(box, limit, column=False):
    """Split a (row, col, rows, cols) box into panels of at most ``limit`` elements.

    Panels follow the tensor's storage order (whole rows of a row-major
    tensor, whole columns of a column-major one), so each panel is read and
    written as long contiguous runs instead of a strided scatter.
    """
    row, col, rows, cols = box
    if rows <= 0 or cols <= 0:
        return []
    if column:
        height = min(rows, limit); width = max(1, limit//height)
    else:
        width = min(cols, limit); height = max(1, limit//width)
    return [(row+y, col+x, min(height, rows-y), min(width, cols-x))
            for x in range(0, cols, width) for y in range(0, rows, height)]


class Timeline:
    """Use simulator times, hiding the native read/compute/write slot offsets."""
    def __init__(self, cores, flit_size):
        self.raw = [[] for _ in range(cores)]
        self.flit_size = flit_size
        self.fid = 0
        self.counts = dict(macs=0, vector_ops=0, dram_read_bytes=0, dram_write_bytes=0,
                           noc_data_bytes=0)

    def add(self, core, time, key, task):
        slot = time if key == 'dram' and not task['is_write'] else time-2 if key == 'dram' else time-1
        if slot < 0:
            raise ValueError('Task precedes its available inputs')
        while len(self.raw[core]) <= slot:
            self.raw[core].append({k: [] for k in KEYS})
        self.raw[core][slot][key].append(task)

    def access(self, core, time, name, base, extent, write=False, dtype=2, ready=None):
        """Local DRAM access; a prefetched read waits only at ``ready`` time."""
        rank = len(base)
        if ready is not None and (write or ready <= time):
            raise ValueError('Prefetch requires a read with a later consumer')
        self.add(core, time, 'dram', dict(name=name, is_write=write, access_base=list(base),
                 access_extent=list(extent), access_stride_add=[1]*rank, access_offset_add=[0]*rank,
                 init_iter=0, stride_iter=1, total_iter=1, **({} if ready is None else dict(ready_iter=ready))))
        count = dtype
        for n in extent:
            count *= n
        self.counts['dram_write_bytes' if write else 'dram_read_bytes'] += count

    def memory(self, time, region, write=False, dtype=2, ready=None):
        self.access(region.core, time, region.name, [region.local_row, region.local_col],
                    [region.rows, region.cols], write, dtype, ready)

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

    def send(self, time, src, dst, size, name='data'):
        """Data arrival synchronises neighbours, as in native ATLAS collectives."""
        if src == dst or not size:
            return
        flits = (size+self.flit_size-1)//self.flit_size
        packet = dict(name=name, src=src, dst=dst, flit_num=flits, init_flit_id=self.fid)
        self.fid += flits
        self.add(src, time, 'noc_tx', dict(packet, is_send=True))
        self.add(dst, time, 'noc_rx', dict(packet, is_send=False))
        self.counts['noc_data_bytes'] += flits*self.flit_size
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
        if kind in ('mpmd-general','spmd-general'):
            # General tasks have a fixed shallow schema. Only packets need
            # renumbering; traversing every scalar in compute/DRAM tasks is
            # expensive when a fused schedule has thousands of iterations.
            raw = [[dict(phase,noc_tx=[copy(p) for p in phase['noc_tx']],
                               noc_rx=[copy(p) for p in phase['noc_rx']]) for phase in row] for row in raw]
        else:
            raw = copy(raw)
        self.next_flit += largest+1
        if self.next_flit >= 2**31:
            raise ValueError('Native flit identifier overflow')
        self.operations.append(dict(name=name, kind=kind, raw=raw, metadata=metadata or {}, operator=operator))

    def add_timeline(self, name, timeline, **metadata):
        self.add(name, timeline.finish(), dict(timeline.counts, **metadata))

    def column(self, region):
        """Whether ``region``'s physical tensor is stored column-major."""
        return self.tensors[region.core][region.name]['strides'][0] == 1

    def gather(self, timeline, groups, start=0):
        """Emit the linear all-gather of replica ``groups`` from step ``start``.

        As in ATLAS's Q all-gather, slabs move hop by hop in both directions,
        one per link and step, cut into SRAM-bounded panels shared by the
        groups on a core; all lines of a phase run together. Returns the
        first free step.
        """
        load = Counter(r.core for g in groups for r in g)
        limit = max(1, self.sram//(16*max(load.values(), default=1)))
        for lines in gather_lines(groups, isqrt(self.cores)):
            end = start
            for members, slabs in lines:
                panels = [panelize(box, limit, self.column(m)) for m, box in zip(members, slabs)]
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
        return start

    def allgather(self, name, groups):
        """ATLAS-style linear all-gather of replica ``groups``, rows then columns."""
        timeline = self.timeline()
        self.gather(timeline, groups)
        self.add_timeline(name, timeline, category='distribution', groups=len(groups))
        self.edges.append(dict(name=name, **timeline.counts, aliased=False))

    def convert(self, name, source, target):
        """Relayout ``source`` into ``target`` as planned by ``conversion``.

        Unicasts are cut into SRAM-bounded panels and pipelined: read, send
        and write of successive panels overlap. Each panel is placed at the
        earliest step where its sender, its receiver and every mesh link on
        its X-Y route are free, so a step carries at most one panel per port
        and link; it reads its panel from DRAM itself (SRAM keeps no panel
        across steps). Replicas then complete by the linear all-gather.
        """
        side = isqrt(self.cores)
        pairs, groups = conversion(source, target, side)
        timeline = self.timeline(); limit = max(1, self.sram//16)
        busy = defaultdict(set); end = 0
        for src, dst in pairs:
            if src.core == dst.core and src.name == dst.name and (src.local_row, src.local_col) == (dst.local_row, dst.local_col):
                continue
            uses = [('tx', src.core), ('rx', dst.core), *route(src.core, dst.core, side)]
            for box in panelize((src.row, src.col, src.rows, src.cols), limit, self.column(src)):
                step = 0
                while any(step in busy[u] for u in uses):
                    step += 1
                for u in uses:
                    busy[u].add(step)
                timeline.memory(step, src.crop(*box))
                timeline.send(step+1, src.core, dst.core, box[2]*box[3]*2)
                timeline.memory(step+2, dst.crop(*box), True); end = max(end, step+3)
        end = self.gather(timeline, groups, end)
        moved = bool(end)
        if moved:
            self.add_timeline(name, timeline, category='layout', regions=len(pairs), groups=len(groups))
        self.edges.append(dict(name=name, **timeline.counts, aliased=not moved))
        return target

    def simulate(self, out, timeout=7200, deadline=None):
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
        from .resources import add, count, empty
        # Resource counts from the exact payloads, native collectives included.
        per = {e['name']: count(e['raw'], e['kind'] == 'communication', self.tensors, self.flit, isqrt(self.cores))
               for e in self.operations}
        total = empty()
        for part in per.values():
            add(total, part)
        manifest = dict(peak_dram_bytes=self.peak, dram_capacity_per_core=self.capacity,
                        edges=self.edges, operators=[{k:v for k,v in op.items() if k not in ('raw','operator')}
                                                   for op in self.operations],
                        resources=dict(total=total, operators=per))
        (out/'graph.json').write_text(json.dumps(manifest, indent=2))
        placement = dict(core_tensor=[dict(core_id=i, tensor=list(t.values())) for i,t in enumerate(self.tensors)])
        dump(out/'operators.yaml',dict(operator=operators)); dump(out/'placement.yaml',placement)
        result = execute(self.chip,out,dict(operator=operators),placement,timeout,deadline)
        return dict(result=result, graph=manifest)
