"""Native GEMM component work and dependency-safe extensions of its pipeline."""
from copy import deepcopy
from itertools import product
from math import prod


def gemm_work(tile, element_size=2):
    """Identical component accounting to the original ATLAS GEMM template."""
    m, n, k = tile
    c = m*n*element_size
    return dict(matrix=[dict(name='gemm_tile', mac_count=m*n*k)],
                vector=[dict(name='gemm_tile_accumulation', vec_count=m*n)],
                buffer_load=[dict(name='gemm_tile_load', byte_count=(m*k+k*n)*element_size, is_write=False),
                             dict(name='gemm_tile_accumulation_load', byte_count=c, is_write=False)],
                buffer_store=[dict(name='gemm_tile_store', byte_count=c, is_write=True),
                              dict(name='gemm_tile_accumulation_store', byte_count=c, is_write=True)])


def native_sram_bytes(tile, element_size=2):
    m, n, k = tile
    return 2*element_size*(m*k+k*n+2*m*n)


def schedule_gemm(shape, tile, loop='mnk', names=('A', 'B', 'C'), element_size=2):
    """Six loop orders using native prefetch, two output banks and accounting.

    Compute at t, prefetch at t-1, and write the previous output at t+1.
    A revisited partial output must be stored before its reload; insert a gap
    when a loop order would otherwise read and write that tile in one DRAM batch.
    Returns native general tasks and compute coordinates for numerical replay.
    """
    if sorted(loop) != list('kmn') or any(n <= 0 or t <= 0 or n % t for n, t in zip(shape, tile)):
        raise ValueError('Exact positive tiles and a permutation of mnk required')
    dims = tuple('mnk'.index(x) for x in loop)
    counts = tuple(n//t for n, t in zip(shape, tile))
    computes = []; memory = []; last_write = {}
    previous = None; clock = 0
    for indices in product(*(range(counts[d]) for d in dims)):
        point = [0, 0, 0]
        for d, value in zip(dims, indices):
            point[d] = value
        m, n, k = point
        c = (m, n); changed_c = previous is None or c != previous[:2]
        if previous is not None and changed_c:
            prev_c = previous[:2]
            memory.append((clock+1, 'C', prev_c, True)); last_write[prev_c] = clock+1
        clock += 1
        if changed_c and c in last_write:
            clock = max(clock, last_write[c]+2)
            memory.append((clock-1, 'C', c, False))
        if previous is None or (m, k) != (previous[0], previous[2]):
            memory.append((clock-1, 'A', (m, k), False))
        if previous is None or (k, n) != (previous[2], previous[1]):
            memory.append((clock-1, 'B', (k, n), False))
        computes.append((clock, tuple(point))); previous = tuple(point)
    memory.append((clock+1, 'C', previous[:2], True))
    keys = ('matrix', 'vector', 'buffer_load', 'buffer_store', 'dram', 'noc_tx', 'noc_rx')
    raw = [{key: [] for key in keys} for _ in range(clock)]
    work = gemm_work(tile, element_size)
    for time, point in computes:
        raw[time-1].update(deepcopy(work))
    axes = dict(A=(0, 2), B=(2, 1), C=(0, 1))
    name_map = dict(zip('ABC', names))
    # Native HBDRAM has separate read/write queues; A then B reproduces its
    # read request ordering. C reloads are extra dependencies of non-OS orders.
    for time, tensor, key, write in sorted(memory, key=lambda x: (x[0], 'ABC'.index(x[1]))):
        extent = [tile[d] for d in axes[tensor]]
        task = dict(name=name_map[tensor], is_write=write,
                    access_base=[i*t for i, t in zip(key, extent)], access_extent=extent,
                    access_stride_add=[1, 1], access_offset_add=[0, 0],
                    init_iter=0, stride_iter=1, total_iter=1)
        raw[time-2 if write else time]['dram'].append(task)
    assert len(computes)*prod(tile) == prod(shape)
    return dict(iteration=clock, execution=raw), computes
