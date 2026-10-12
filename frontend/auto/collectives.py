"""Native line collectives, including a post-reduction gated FFN epilogue."""
import math

from frontend.util import make_dram_task


def allreduce_tasks(input_placement, output_placement, name, rows, offset, length,
                    element_size, reduction_ops, direction, side, flit_size,
                    flit_indices, *, gated=False):
    """Reduce partials, optionally apply SiLU(gate)*up, then gather replicas.

    Gated shards store [gate_slice, up_slice]. Each reduce-scatter chunk carries
    matching slices from BOTH halves; its owner applies the nonlinearity only
    after all K contributions arrive. All-gather carries the contracted result.
    Scalar vector accounting for SiLU+gate is neg, exp, add, div, mul (5 ops).
    """
    if side < 2 or direction not in (0, 1):
        raise ValueError('Line collective requires side >= 2 and direction 0 or 1')
    factor = 2 if gated else 1
    if length <= 0 or length % factor:
        raise ValueError('Gated projection must contain equally sized gate/up halves')
    width = length // factor
    if (rows > input_placement['shape'][0] or rows > output_placement['shape'][0]
            or offset < 0 or offset + length > input_placement['shape'][1]
            or width > output_placement['shape'][1]):
        raise ValueError('Collective input/output window exceeds its tensor shape')
    chunk_width = math.ceil(width / side)
    iterations = 2 * (side - 1) + int(gated)
    op = dict(name=name + '_comm_1d_all_reduce', type='communication',
              iteration=iterations, core_num=side * side,
              file_prefix=name + '_comm_1d_all_reduce/')
    cores = [dict(communication=dict(on_chip=[dict(
        tx=dict(buffer_load=[], noc=[]),
        rx=dict(buffer_load=[], buffer_store=[], vector=[], noc=[]))
        for _ in range(iterations)], dram=[])) for _ in range(side * side)]
    loaded, stored = set(), set()

    def core(line, rank):
        return rank * side + line if direction == 0 else line * side + rank

    def extent(chunk):
        return max(0, min(chunk_width, width - chunk * chunk_width))

    def dram(core_id, chunk, iteration, write):
        seen = stored if write else loaded
        if (core_id, chunk) in seen or not extent(chunk):
            return
        seen.add((core_id, chunk))
        tensor = output_placement if write else input_placement
        # Separate gate/up windows preserve the existing row-major DRAM layout.
        for half in range(1 if write else factor):
            col = chunk * chunk_width + (0 if write else offset + half * width)
            cores[core_id]['communication']['dram'].append(make_dram_task(
                name=tensor['name'], is_write=write, access_base=[0, col],
                access_extent=[rows, extent(chunk)], access_stride_add=[1, 1],
                access_offset_add=[tensor['shape'][0], chunk_width],
                init_iter=iteration, stride_iter=1, total_iter=1))

    def transfer(src, dst, chunk, iteration, reduce):
        elements = rows * extent(chunk) * (factor if reduce else 1)
        if not elements:
            return
        size = elements * element_size
        label = ('reduce' if reduce else 'gather') + f'_chunk{chunk}'
        tx = cores[src]['communication']['on_chip'][iteration]['tx']
        rx = cores[dst]['communication']['on_chip'][iteration]['rx']
        tx['buffer_load'].append(dict(name=label, byte_count=size, is_write=False))
        packet = dict(name=label, src=src, dst=dst,
                      flit_num=math.ceil(size / flit_size), init_flit_id=flit_indices[src])
        tx['noc'].append(dict(packet, is_send=True))
        rx['noc'].append(dict(packet, is_send=False))
        rx['buffer_store'].append(dict(name=label, byte_count=size, is_write=True))
        if reduce:
            rx['buffer_load'].append(dict(name=label, byte_count=2 * size, is_write=False))
            rx['buffer_store'].append(dict(name=label, byte_count=size, is_write=True))
            rx['vector'].append(dict(name=label, vec_count=elements * reduction_ops))
        flit_indices[src] += packet['flit_num']

    # Preserve the native bidirectional line schedule, including DRAM staging.
    for step in range(side - 1):
        for line in range(side):
            for rank in range(side):
                edges = []
                if rank <= step:
                    edges.append((rank + 1, abs(rank - step)))
                if rank >= side - 1 - step:
                    edges.append((rank - 1, side - 1 - abs(rank + step - (side - 1))))
                src = core(line, rank)
                for dest, chunk in edges:
                    dram(src, chunk, step, False)
                    if 0 <= dest < side:
                        dst = core(line, dest)
                        transfer(src, dst, chunk, step, True)
                        dram(dst, chunk, step, False)

    if gated:
        for line in range(side):
            for rank in range(side):
                chunk = side - 1 - rank
                count = rows * extent(chunk)
                rx = cores[core(line, rank)]['communication']['on_chip'][side - 1]['rx']
                rx['buffer_load'].append(dict(name=f'gate_chunk{chunk}',
                                             byte_count=2 * count * element_size, is_write=False))
                rx['buffer_store'].append(dict(name=f'gate_chunk{chunk}',
                                              byte_count=count * element_size, is_write=True))
                rx['vector'].append(dict(name=f'gate_chunk{chunk}', vec_count=5 * count))

    for step in range(side - 1):
        iteration = side - 1 + int(gated) + step
        for line in range(side):
            for rank in range(side):
                edges = []
                if rank >= step:
                    edges.append((rank + 1, side - 1 - abs(rank - step)))
                if rank <= side - 1 - step:
                    edges.append((rank - 1, abs(rank + step - (side - 1))))
                src = core(line, rank)
                for dest, chunk in edges:
                    dram(src, chunk, iteration + 1, True)
                    if 0 <= dest < side:
                        dst = core(line, dest)
                        transfer(src, dst, chunk, iteration, False)
                        dram(dst, chunk, iteration + 1, True)
    return op, cores
