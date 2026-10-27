"""Compilation methods compared on one ATLAS backend.

A method is a named mapping space over the same compiler and simulator.
Hardware, input accounting, lowering and measurement are shared, so methods
differ only in what they may choose. ``space`` yields the plans a method adds;
``includes`` lists every smaller method whose space it contains, so the
spaces nest (atlas ⊂ local ⊂ ours) by construction. Each experiment runs one
method on its own; comparing methods means running each separately. A new
baseline (e.g. ELK/T10 or FlashGEMM) is one more entry with its own ``space``.

The original ATLAS compiler is the common incumbent of every method. It is
tuned by ``gemm.native`` and the native attention plans, outside this registry.
"""
from dataclasses import dataclass

from .gemm.space import candidates
from .model.attention import plans as attention_plans
from .sim import hardware


@dataclass(frozen=True)
class Method:
    name: str
    description: str
    rotation: bool = False
    includes: tuple = ()

    def space(self, problem, cores, sram, dram, tile_min=16, tile_policy='powers', layer_policy='elk', skip=None,
              single_b=False):
        """Plans this method adds; see ``gemm.space.candidates`` for ``skip`` and ``single_b``."""
        drop = (lambda q: not q.layers or bool(skip and skip(q))) if self.rotation else skip
        return candidates(problem, cores, sram, tile_min, tile_policy, layer_policy, dram, self.rotation, drop, single_b)

    def attention(self, layout, sram):
        """Fused-attention plans this method adds to the native context splits."""
        if not self.rotation:
            return []
        return [p for p in attention_plans(layout, sram, True) if (p.query_parts, p.context_parts) != (1, 1)]


METHODS = {m.name: m for m in (
    Method('local', 'our spatial/tile/loop space without rotation: Ps incl. split-K, tiles, six loop orders'),
    Method('ours', 'adds temporal rotation (Pt) of A/B/C and of fused-attention KV tiles', True, ('local',)),
)}


def target(chip, tile_min=16, tile_policy='powers', layer_policy='elk'):
    """Keyword arguments of ``Method.space`` for one chip."""
    arch, dram, _ = hardware(chip)
    return dict(cores=arch['core_num'], sram=arch['core']['buffer']['buffer_size']*1024, dram=dram,
                tile_min=tile_min, tile_policy=tile_policy, layer_policy=layer_policy)


def propose(problem, hw, pick, name):
    """Ranked proposals of method ``name`` and of every method it includes.

    ``pick`` maps a space ``skip -> plans`` to ``([(plan, score)], scanned)``,
    as ``search.shortlist`` does. Returns the lists and the plans scored, per method.
    """
    ranked = {}; scanned = {}
    for m in (METHODS[x] for x in (*METHODS[name].includes, name)):
        ranked[m.name], scanned[m.name] = pick(lambda skip=None, m=m: m.space(problem, skip=skip, **hw))
    return ranked, scanned


def union(ranked, name, count=None, seed=()):
    """Candidates of a method: ``seed``, then the first ``count`` of it and every included method."""
    names = (*METHODS[name].includes, name)
    return list(dict.fromkeys([*seed, *(p for x in names for p, _ in ranked[x][:count])]))
