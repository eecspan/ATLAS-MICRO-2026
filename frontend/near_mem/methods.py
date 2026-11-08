"""Compilation methods compared on one ATLAS backend.

A method is a named mapping space over the same compiler and simulator.
Hardware, input accounting, lowering and measurement are shared, so methods
differ only in what they may choose. Every method runs on its own and writes
its own cycles; ``experiments.compare`` computes speedups over ``tp``.

``tp`` is the tensor-parallel baseline: inputs replicated, no rotation, the
native ``mnk`` kernel and ATLAS's native tile rule. A single GEMM may split M
or N; in a graph every GEMM uses the fixed K/N mesh grid of ATLAS's TP
deployment along its ``direction``. ``atlas`` runs the unmodified upstream
GEMM generator and tile explorer (``gemm.native``); it only checks that
``tp`` loses nothing and is not a baseline of its own. ``space`` yields a
method's whole space. ``ours`` is the full Ps/Pt space, Pt = 1 included, with
both mesh orientations of every partition, so the TP grid is one of its
points; ``local`` is the same space restricted to Pt = 1 (an ablation).
Only our spaces (``boundary``) move DRAM residency per plan, so only they
choose reduction outputs jointly with their neighbours in a graph; every
baseline keeps the boundary its dataflow prescribes.
Baselines keep only their own dataflow: T10 rotates on chip over one
plan-independent DRAM layout with its fixed ``mnk`` loop; FlashGEMM
all-gathers single-copy blocks.
"""
from dataclasses import dataclass, replace

from math import isqrt

from .gemm import flash
from .gemm.plan import Problem
from .gemm.space import LOOPS, candidates, divisors
from .model.attention import plans as attention_plans
from .sim import hardware


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    rotation: bool = False
    # Joint boundary search in a graph: output contracts and neighbour-aware selection.
    boundary: bool = False
    receive: str = 'sram_consume'
    layout: str = 'local'
    flash: bool = False
    native: bool = False
    tp: bool = False
    # Each method owns its fused dataflow; proxy/lowering/output policies are shared.
    attention_mode: str = 'native'
    loops: tuple = LOOPS

    def oriented(self, cores):
        """Whether this method's plans come in both mesh orientations."""
        return not (self.flash or self.tp) and self.layout == 'local' and isqrt(cores)**2 == cores

    def space(self, problem, cores, sram, dram, tile_min=8, tile_policy='all', layer_policy='elk', skip=None,
              single_b=False, direction=None, orient=True):
        """This method's plans; see ``gemm.space.candidates`` for ``skip`` and ``single_b``.
        ``direction`` is a graph GEMM's TP mesh direction (None for a single GEMM)."""
        if self.flash:
            return flash.candidates(problem, cores, sram, tile_min, tile_policy)
        if self.tp:
            execution = problem.shape
            if direction is None:
                parts, transpose = [(pm, cores//pm, 1) for pm in divisors(cores)], False
                if not any(problem.m % pm == 0 and problem.n % (cores//pm) == 0
                           for pm in divisors(cores)):
                    # Standalone TP uses M/N replication groups. If no group
                    # covers a small tail exactly, pad only M/N to the nearest
                    # legal group; the plan records and simulates that work.
                    choices = []
                    for pm in divisors(cores):
                        pn = cores//pm
                        m = ((problem.m+pm-1)//pm)*pm
                        n = ((problem.n+pn-1)//pn)*pn
                        choices.append(((m*n, m-problem.m+n-problem.n), (m,n,problem.k)))
                    execution = min(choices)[1]
            elif isqrt(cores)**2 == cores:
                side = isqrt(cores)
                parts, transpose = [(1, side, side)], direction == 0
                # The graph TP baseline fixes K/N to the mesh axes. Pad just
                # those axes; M is not split and keeps its logical extent.
                execution = (problem.m,
                             ((problem.n+side-1)//side)*side,
                             ((problem.k+side-1)//side)*side)
            else:
                raise ValueError('The TP grid needs a square mesh')
            adapted_skip = _padding_skip(skip, problem.shape, execution)
            work = Problem(*execution, problem.dtype_bytes)
            plans = candidates(work, cores, sram, TP_TILE_MIN, 'all', layer_policy, dram, False, adapted_skip,
                               loops=('mnk',), parts=parts, transpose=transpose)
            if execution == problem.shape:
                return plans
            return (_padding_metadata(p, problem.shape, execution) for p in plans)
        execution = problem.shape
        if self.layout == 'global':
            side = isqrt(cores)
            if side*side != cores:
                raise ValueError('T10 requires a square mesh')
            if problem.m == 1:
                # A remains single-copy and is striped by K; B/C use N to
                # occupy every DRAM bank. A's remote reads are charged later.
                execution = (1, ((problem.n+cores-1)//cores)*cores,
                             ((problem.k+cores-1)//cores)*cores)
            elif any(n % side for n in problem.shape):
                execution = tuple(((n+side-1)//side)*side for n in problem.shape)
        adapted_skip = _padding_skip(skip, problem.shape, execution)
        work = Problem(*execution, problem.dtype_bytes)
        # Both mesh orientations of a partition: the TP grid's transposed
        # numbering is a point of our space, not a separate method. With
        # ``orient=False`` only the row-major numbering is enumerated; the
        # proxy ignores the orientation, so callers add twins after ranking
        # (``oriented``).
        plans = (p for t in ((False, True) if orient and self.oriented(cores) else (False,)) for p in candidates(
            work, cores, sram, tile_min, tile_policy, layer_policy, dram, self.rotation, adapted_skip, single_b,
            self.receive, self.layout, self.loops, transpose=t))
        if execution == problem.shape:
            return plans
        return (_padding_metadata(p, problem.shape, execution) for p in plans)

    def attention(self, layout, sram):
        """Method-local fused space; no implicit TP fallback for the DRAM ring."""
        if self.attention_mode == 'ours':
            return list(attention_plans(layout, sram, True))
        if self.attention_mode in ('t10', 'dram'):
            return list(attention_plans(layout, sram, dataflow=self.attention_mode))
        if self.attention_mode == 'flash':
            return [p for flow in ('flash_os','flash_ws') for p in attention_plans(layout,sram,dataflow=flow)]
        if self.attention_mode == 'native':
            return list(attention_plans(layout, sram, False))
        raise ValueError(f'Unknown attention mode: {self.attention_mode}')


TP_TILE_MIN = (8, 8, 512)  # ATLAS's native minimum tile (m, n, k)


def _padding_metadata(plan, logical, execution):
    return replace(plan, logical_shape=logical, execution_shape=execution)


def _padding_skip(skip, logical, execution):
    if skip is None or logical == execution:
        return skip
    return lambda prefix: skip(_padding_metadata(prefix, logical, execution))

METHODS = {m.name: m for m in (
    Method('tp', 'tensor parallel: replicated inputs, M/N (single GEMM) or fixed K/N TP grid, mnk, native tiles',
           tp=True),
    Method('atlas', 'unmodified upstream ATLAS GEMM generator and tile explorer (a check of tp)', native=True),
    Method('local', 'ablation: our space at Pt = 1 (Ps incl. split-K and mesh orientation, tiles, six loop orders)',
           boundary=True),
    Method('ours', 'our full space: Ps and temporal rotation Pt of A/B/C (Pt = 1 included) and of fused-attention '
           'Q/KV tiles', True, True, attention_mode='ours'),
    Method('dram', 'DRAM-level rotation: rotated A/B/C blocks are forwarded DRAM to DRAM and re-read',
           True, True, receive='dram_land', attention_mode='dram'),
    Method('t10', 'improved T10: on-chip Ps/Pt rotation over one plan-independent global DRAM layout, mnk loop',
           True, layout='global', loops=('mnk',), attention_mode='t10'),
    Method('flash', 'FlashGEMM: single-copy blocks, per-K-tile row/column all-gathers overlapped with compute',
           flash=True, attention_mode='flash'),
)}


def target(chip, tile_min=8, tile_policy='all', layer_policy='elk'):
    """Keyword arguments of ``Method.space`` for one chip."""
    arch, dram, _ = hardware(chip)
    return dict(cores=arch['core_num'], sram=arch['core']['buffer']['buffer_size']*1024, dram=dram,
                tile_min=tile_min, tile_policy=tile_policy, layer_policy=layer_policy)
