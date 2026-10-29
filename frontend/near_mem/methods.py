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
``tp`` loses nothing and is not a baseline of its own. ``space`` yields the
plans a method adds; ``includes`` lists the smaller methods whose plans it
also proposes, so spaces nest (tp ⊂ local ⊂ ours) by construction.
Baselines keep only their own dataflow: T10 rotates on chip over one
plan-independent DRAM layout with its fixed ``mnk`` loop; FlashGEMM
all-gathers single-copy blocks.
"""
from dataclasses import dataclass

from math import isqrt

from .gemm import flash
from .gemm.space import LOOPS, candidates, divisors
from .model.attention import plans as attention_plans
from .sim import hardware


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    rotation: bool = False
    includes: tuple = ()
    receive: str = 'sram_consume'
    layout: str = 'local'
    flash: bool = False
    native: bool = False
    tp: bool = False
    loops: tuple = LOOPS

    def space(self, problem, cores, sram, dram, tile_min=8, tile_policy='all', layer_policy='elk', skip=None,
              single_b=False, direction=None):
        """Plans this method adds; see ``gemm.space.candidates`` for ``skip`` and ``single_b``.
        ``direction`` is a graph GEMM's TP mesh direction (None for a single GEMM)."""
        if self.flash:
            return flash.candidates(problem, cores, sram, tile_min, tile_policy)
        if self.tp:
            if direction is None:
                parts, transpose = [(pm, cores//pm, 1) for pm in divisors(cores)], False
            elif isqrt(cores)**2 == cores:
                parts, transpose = [(1, isqrt(cores), isqrt(cores))], direction == 0
            else:
                raise ValueError('The TP grid needs a square mesh')
            return candidates(problem, cores, sram, TP_TILE_MIN, 'all', layer_policy, dram, False, skip,
                              loops=('mnk',), parts=parts, transpose=transpose)
        # A local method with rotation adds only rotating plans (no-rotation
        # plans belong to ``local``); a global layout keeps both.
        own = self.rotation and self.layout == 'local'
        drop = (lambda q: not q.layers or bool(skip and skip(q))) if own else skip
        return candidates(problem, cores, sram, tile_min, tile_policy, layer_policy, dram, self.rotation, drop, single_b,
                          self.receive, self.layout, self.loops)

    def attention(self, layout, sram):
        """Fused-attention plans this method adds to the native context splits."""
        if self.name != 'ours':
            return []
        return [p for p in attention_plans(layout, sram, True) if (p.query_parts, p.context_parts) != (1, 1)]


TP_TILE_MIN = (8, 8, 512)  # ATLAS's native minimum tile (m, n, k)

METHODS = {m.name: m for m in (
    Method('tp', 'tensor parallel: replicated inputs, M/N (single GEMM) or fixed K/N TP grid, mnk, native tiles',
           tp=True),
    Method('atlas', 'unmodified upstream ATLAS GEMM generator and tile explorer (a check of tp)', native=True),
    Method('local', 'our spatial/tile/loop space without rotation: Ps incl. split-K, tiles, six loop orders',
           includes=('tp',)),
    Method('ours', 'adds temporal rotation (Pt) of A/B/C and of fused-attention KV tiles', True, ('tp', 'local')),
    Method('dram', 'DRAM-level rotation: rotated A/B/C blocks are forwarded DRAM to DRAM and re-read',
           True, ('tp', 'local'), receive='dram_land'),
    Method('t10', 'improved T10: on-chip Ps/Pt rotation over one plan-independent global DRAM layout, mnk loop',
           True, layout='global', loops=('mnk',)),
    Method('flash', 'FlashGEMM: single-copy blocks, per-K-tile row/column all-gathers overlapped with compute',
           flash=True),
)}


def target(chip, tile_min=8, tile_policy='all', layer_policy='elk'):
    """Keyword arguments of ``Method.space`` for one chip."""
    arch, dram, _ = hardware(chip)
    return dict(cores=arch['core_num'], sram=arch['core']['buffer']['buffer_size']*1024, dram=dram,
                tile_min=tile_min, tile_policy=tile_policy, layer_policy=layer_policy)


def propose(problem, hw, pick, name):
    """Ranked proposals of method ``name`` and of every method it includes.

    ``pick(space, method)`` maps a space ``skip -> plans`` to
    ``([(plan, score)], scanned)``, as ``search.shortlist`` does. Returns the
    lists and the plans scored, per method.
    """
    if METHODS[name].native:
        raise ValueError('The ATLAS space is generated by gemm.native')
    ranked = {}; scanned = {}
    for m in (METHODS[x] for x in (*METHODS[name].includes, name)):
        ranked[m.name], scanned[m.name] = pick(lambda skip=None, m=m: m.space(problem, skip=skip, **hw), m)
    return ranked, scanned


def union(ranked, name, count=None):
    """Candidates of a method: the first ``count`` of it and of every included method."""
    names = (*METHODS[name].includes, name)
    return list(dict.fromkeys(p for x in names for p, _ in ranked[x][:count]))
