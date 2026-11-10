"""Resource counts, utilization and an energy estimate from the simulated tasks.

``count`` walks the exact task payloads ATLAS executes, for our general tasks
and the native collectives alike, so every method is counted by one rule:
MACs, vector ops, SRAM and DRAM bytes, NoC bytes and byte-hops (XY routes on
the square mesh: Manhattan distance). DRAM bytes are logical (accessed
elements); row-buffer effects stay inside the simulated cycles.

``report`` adds utilization against chip peaks over the simulated cycles, a
time breakdown and an event-energy estimate (per-event energy times counts,
plus static power times time; published coefficients in ``ENERGY``, reported
only, never used by search).

Time breakdown (``breakdown``), per the simulator's per-cycle counts on the
critical core: in a cycle the core either still runs its on-chip work
(compute, possibly overlapping DRAM and NoC transfers), or waits only for
DRAM, only for the NoC, or for both; ``busy`` gives each resource's total
busy time, overlapped or not.
"""
from math import isqrt, prod

from .sim import hardware

FIELDS = ('macs', 'vector_ops', 'sram_read_bytes', 'sram_write_bytes', 'dram_read_bytes', 'dram_write_bytes',
          'noc_bytes', 'noc_byte_hops')

# pJ per event. DRAM: stacked hybrid-bonded DRAM (VLSI'23, used by ATLAS and
# Helios); SRAM and compute: Helios (0.019 pJ/bit, 0.43 pJ/FLOP; a MAC is two
# FLOPs); NoC: Tangram (0.61 pJ/bit/hop). Static power: 10% of the chip's
# configured component power, as DeepStack takes 10% of TDP.
ENERGY = dict(dram_pj_per_bit=0.66, sram_pj_per_bit=0.019, flop_pj=0.43, noc_pj_per_bit_hop=0.61,
              static_fraction=0.1)


def empty():
    return dict.fromkeys(FIELDS, 0)


def add(total, part):
    for k in FIELDS:
        total[k] += part.get(k, 0)
    return total


def _phase(phase):
    """(matrix, vector, SRAM reads, SRAM writes, sent packets) of one task phase."""
    if 'tx' in phase:  # an iteration of a native communication operator
        tx, rx = phase['tx'], phase['rx']
        return ([], rx.get('vector', []), tx.get('buffer_load', [])+rx.get('buffer_load', []),
                rx.get('buffer_store', []), [p for p in tx.get('noc', []) if p.get('is_send')])
    return (phase.get('matrix', []), phase.get('vector', []), phase.get('buffer_load', []),
            phase.get('buffer_store', []), [p for p in phase.get('noc_tx', []) if p.get('is_send', True)])


def count(raw, communication, tensors, flit, side):
    """Counts of one operator: ``raw`` per-core tasks, ``tensors`` each core's placements by name."""
    out = empty()
    hops = lambda a, b: abs(a % side-b % side)+abs(a//side-b//side)
    for core, task in enumerate(raw):
        if communication:
            phases, drams = task['communication']['on_chip'], task['communication']['dram']
        else:
            phases, drams = task, [d for phase in task for d in phase.get('dram', [])]
        for phase in phases:
            matrix, vector, reads, writes, packets = _phase(phase)
            out['macs'] += sum(t['mac_count'] for t in matrix)
            out['vector_ops'] += sum(t['vec_count'] for t in vector)
            out['sram_read_bytes'] += sum(t['byte_count'] for t in reads)
            out['sram_write_bytes'] += sum(t['byte_count'] for t in writes)
            for p in packets:
                size = p['flit_num']*flit
                out['noc_bytes'] += size; out['noc_byte_hops'] += size*hops(p['src'], p['dst'])
        for d in drams:
            size = prod(d['access_extent'])*tensors[core][d['name']]['element_size']*d.get('total_iter', 1)
            out['dram_write_bytes' if d['is_write'] else 'dram_read_bytes'] += size
    return out


def breakdown(result):
    """Time of a simulator ``result`` on its critical core: on-chip work, then waits
    for DRAM only, the NoC only, or both; ``busy`` per resource, overlapped or not."""
    c = result['critical']
    dram, noc, both = (c[f'matrix_bubble_{k}_cycles'] for k in ('dram', 'noc', 'dram_noc'))
    return dict(cycles=c['e2e_cycles'], on_chip=c['e2e_cycles']-dram-noc-both, exposed_dram=dram,
                exposed_noc=noc, exposed_dram_noc=both,
                busy=dict(matrix=c['matrix_cycles'], vector=c['vector_cycles'], dram=c['dram_cycles'], noc=c['noc_cycles']))


def report(resources, cycles, chip, energy=ENERGY):
    """Utilization of compute, DRAM and NoC over ``cycles``, and the energy estimate in joules."""
    from .gemm.space import Proxy
    arch, _, _ = hardware(chip); cores = arch['core_num']; side = isqrt(cores); flit = arch['noc']['flit_size']
    links = 4*side*(side-1)  # directed mesh links
    utilization = dict(compute=resources['macs']/(cycles*cores*arch['core']['matrix']['mac_num']),
                       dram=(resources['dram_read_bytes']+resources['dram_write_bytes'])/(cycles*cores*Proxy(chip).dram_bw),
                       noc=resources['noc_byte_hops']/(cycles*links*flit))
    pj = dict(compute=(2*resources['macs']+resources['vector_ops'])*energy['flop_pj'],
              sram=8*(resources['sram_read_bytes']+resources['sram_write_bytes'])*energy['sram_pj_per_bit'],
              dram=8*(resources['dram_read_bytes']+resources['dram_write_bytes'])*energy['dram_pj_per_bit'],
              noc=8*resources['noc_byte_hops']*energy['noc_pj_per_bit_hop'])
    joules = {k: v*1e-12 for k, v in pj.items()}
    power = arch['dram']['power']+arch['noc']['power']+sum(arch['core'][k]['power']
                                                          for k in ('controller', 'matrix', 'vector', 'buffer'))
    joules['static'] = energy['static_fraction']*power*cores*cycles/(arch['frequency']*1e6)
    joules['total'] = sum(joules.values())
    return dict(resources=resources, utilization=utilization, energy_j=joules)
