"""Physical tensor regions and explicit graph edges for native ATLAS tasks.

A Region names actual per-core DRAM storage. Replication is represented by
multiple regions, never an implicit broadcast. The allocator reuses dead
activations; all weights and KV allocations remain live through the graph.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from math import prod
from pathlib import Path

from .backend import KEYS, hardware
from .run import execute, dump


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

    def tensor(self, core, name, shape, strides=None, dtype=2, persistent=False):
        if name in self.tensors[core]:
            raise ValueError(f'Duplicate physical tensor {core}/{name}')
        size = prod(shape)*dtype
        address = (self.peak[core]+127)//128*128 if persistent else 0
        for start, end in sorted(self.live[core].values()):
            if address+size <= start:
                break
            address = max(address, (end+127)//128*128)
        if address+size > self.capacity:
            raise ValueError(f'Per-core DRAM capacity exceeded on core {core}')
        tensor = dict(name=name, shape=list(shape), strides=strides or [shape[1], 1],
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
        # Core/src is part of BookSim's identity; still use disjoint ranges
        # across operators, including native collectives and extended kernels.
        raw = deepcopy(raw)
        largest = -1
        def packets(obj):
            if isinstance(obj, dict):
                if 'init_flit_id' in obj:
                    yield obj
                else:
                    for value in obj.values():
                        yield from packets(value)
            elif isinstance(obj, list):
                for value in obj:
                    yield from packets(value)
        for packet in packets(raw):
            largest = max(largest, packet['init_flit_id']+packet['flit_num']-1)
            packet['init_flit_id'] += self.next_flit
        self.next_flit += largest+1
        if self.next_flit >= 2**31:
            raise ValueError('Native flit identifier overflow')
        self.operations.append(dict(name=name, kind=kind, raw=raw, metadata=metadata or {}, operator=operator))

    def add_timeline(self, name, timeline, **metadata):
        self.add(name, timeline.finish(), dict(timeline.counts, **metadata))

    def convert(self, name, source, target):
        pairs = transfers(source, target)
        timeline = self.timeline()
        pending = []
        for src, dst in pairs:
            if src.core == dst.core and src.name == dst.name and (src.local_row, src.local_col) == (dst.local_row, dst.local_col):
                continue
            # One send/receive panel per endpoint; bound even very wide rows.
            max_elements = self.sram//8
            for c in range(0, src.cols, max_elements):
                width = min(max_elements, src.cols-c)
                height = max(1, max_elements//width)
                for r in range(0, src.rows, height):
                    box = (src.row+r, src.col+c, min(height, src.rows-r), width)
                    pending.append((src.crop(*box), dst.crop(*box)))
        # Multicast through ordinary unicasts: retain one source panel and read
        # DRAM once when several consumers need the identical physical window.
        groups = {}
        for src,dst in pending:
            groups.setdefault(src,[]).append(dst)
        pending = list(groups.items())
        step = 0
        while pending:
            used = set(); rest = []
            for src,destinations in pending:
                endpoints = {src.core, *(dst.core for dst in destinations)}
                if used & endpoints:
                    rest.append((src,destinations)); continue
                used.update(endpoints)
                timeline.memory(step,src)
                for dst in destinations:
                    if src.core != dst.core:
                        timeline.send(step+1,dst.core,src.core,1,credit=True)
                        timeline.send(step+2,src.core,dst.core,src.area*2)
                    timeline.memory(step+3,dst,True)
            pending = rest; step += 4
        if step:
            self.add_timeline(name, timeline, category='layout', regions=len(pairs))
        record = dict(name=name, **timeline.counts, aliased=not step)
        self.edges.append(record)
        return target

    def simulate(self, out, cache=None, config_key=None):
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
            result = execute(self.chip,out,dict(operator=operators),placement)
        else:
            from .execution import task_digest
            digest = task_digest(out,self.cores)
            result,source = cache.get(digest,config_key)
            if result is None:
                result = execute(self.chip,out,dict(operator=operators),placement)
            else:
                dump(out/'result.json',result)
            cache.put(digest,out,source)
        return dict(result=result, graph=manifest)
