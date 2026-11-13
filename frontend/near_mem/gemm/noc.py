"""Maximum channel load of a GEMM plan's NoC traffic on BookSim's XY mesh.

A bandwidth-bound phase lasts at least as long as its busiest injection
port, ejection port or directed link (Dally & Towles' channel-load bound).
Everything here depends only on a plan's structure (spatial parts, temporal
periods, mesh numbering, the head of its gcd-layer order and its output
contract), never on its tiles, so each structure is evaluated once. Rotation
loads are in units of the volume every core sends; ``peak`` scales them.
A shift phase moves all tensors of its gcd layer together (C included), so
its frontier spans all of them.
"""
from collections import Counter, defaultdict
from dataclasses import replace
from functools import lru_cache
from itertools import permutations
from math import isqrt

from ..output import reduce_steps, root_index
from ..tasks import route
from .ownership import Ownership
from .plan import Problem


def mesh(cores):
    """Mesh side of a square chip (XY links are charged), else None (ports only)."""
    side = isqrt(cores)
    return side if side*side == cores else None


def load(flows, side):
    """Per-key load of ``(src, dst, amount)`` flows: tx/rx ports and XY links."""
    out = Counter()
    for src, dst, amount in flows:
        if src == dst:
            continue
        out['tx', src] += amount; out['rx', dst] += amount
        for edge in (route(src, dst, side) if side else ()):
            out[edge] += amount
    return out


def frontier(loads):
    """``(names, vectors)``: the non-dominated per-key loads of tensors sharing a phase."""
    names = tuple(sorted(loads))
    keys = set().union(*(l.keys() for l in loads.values()))
    vectors = {tuple(loads[n][k] for n in names) for k in keys}
    kept = [v for v in vectors if not any(u != v and all(x >= y for x, y in zip(u, v)) for u in vectors)]
    return names, tuple(sorted(kept))


def peak(front, weights):
    """Busiest key of a phase whose tensors each send ``weights[name]`` per core.
    Monotone and homogeneous in ``weights``, so bounds scale with volumes."""
    names, vectors = front
    return max((sum(weights[n]*x for n, x in zip(names, v)) for v in vectors), default=0)


def shell(p):
    """``p`` reduced to the fields the loads depend on."""
    return replace(p, tile=None, loop='mnk', group=(1, 1, 1), layer_order=(), panel=None,
                   logical_shape=None, execution_shape=None)


def skeleton(shell, order=()):
    """A unit-tile problem and plan with ``shell``'s ownership structure."""
    plan = replace(shell, tile=(1, 1, 1), layer_order=order, output='sharded')
    return Problem(*(s*t for s, t in zip(plan.ps, plan.periods))), plan


def head(layers, order):
    """What ``Ownership`` takes from a layer order: its uniform-ring pair, or None."""
    order = order or tuple(range(len(layers)))
    if len(order) < 2:
        return None
    shared, solo = layers[order[0]], layers[order[1]]
    if shared.axis != solo.axis or len(shared.tensors) != 2 or len(solo.tensors) != 1:
        return None
    return tuple(order[:2])


def heads(layers):
    """Every ``head`` some layer order of ``layers`` has."""
    return {head(layers, o) for o in permutations(range(len(layers)))} if layers else {None}


def representative(layers, pair):
    if pair is not None:
        return pair+tuple(i for i in range(len(layers)) if i not in pair)
    return next(o for o in permutations(range(len(layers))) if head(layers, o) is None)


@lru_cache(maxsize=1 << 14)
def rotation(shell, pair):
    """Per gcd layer, for layer orders headed by ``pair``: ``(front, hops)`` with the
    ``frontier`` of the tensors the layer shifts together in one phase and the
    longest route of the compiler's shift (``own.neighbor(core, layer, v, -1)``)."""
    problem, plan = skeleton(shell, representative(shell.layers, pair) if shell.layers else ())
    own = Ownership(problem, plan); side = mesh(plan.cores); out = []
    for i, layer in enumerate(own.layers):
        pairs = {v: [(s, own.neighbor(s, i, v, -1), 1) for s in range(plan.cores)] for v in layer.tensors}
        loads = {v: load(f, side) for v, f in pairs.items()}
        hops = max(len(route(s, d, side)) if side else 1 for f in pairs.values() for s, d, _ in f)
        out.append((frontier(loads), hops))
    return tuple(out)


@lru_cache(maxsize=1 << 14)
def reduction(shell):
    """Residual split-K finalization: its busiest key summed over the steps of one
    wave, per tile and step (>= 1). Owner groups are the cores sharing home C
    blocks, ordered by id like the compiler's; ``sharded``/``replicated`` run the
    bidirectional line reduce-scatter, ``first``/``last`` two inward lines."""
    problem, plan = skeleton(shell)
    own = Ownership(problem, plan); side = mesh(plan.cores); owners = defaultdict(list)
    for core in range(plan.cores):
        owners[tuple(own.tensor_blocks(core, 'C'))].append(core)
    groups = list(owners.values()); r = len(groups[0])
    if r < 2:
        return 1
    steps = defaultdict(list)
    for g in groups:
        if shell.output in ('first', 'last'):
            for s, src, dst in reduce_steps(g, root_index(shell.output, r, 0)):steps[s].append((src, dst, 1))
            continue
        for s in range(r-1):
            for j in range(r):
                if j <= s and j+1 < r:steps[s].append((g[j], g[j+1], 1))
                if j >= r-1-s and j > 0:steps[s].append((g[j], g[j-1], 1))
    return sum(max(load(f, side).values()) for f in steps.values())/len(steps)


@lru_cache(maxsize=1 << 12)
def gather(problem, shell):
    """Elements crossing the busiest port or link of the linear all-gather that
    distributes single-copy inputs (``distribution.distribute``), summed over its
    hops; None when a replica group is not a mesh sub-grid."""
    from ..tasks import gather_lines
    from .distribution import replica_groups
    plan = replace(shell, tile=(1, 1, 1)); side = mesh(plan.cores)
    groups = [g for v in 'AB' for g in replica_groups(problem, plan, v)[1]]
    if not side:
        return None
    try:
        phases = gather_lines(groups, side)
    except ValueError:
        return None
    total = 0
    for lines in phases:
        # Hop t: the slab of member o moves from o+d*(t-1) to o+d*t, both directions.
        for hop in range(1, max((len(m) for m, _ in lines), default=1)):
            flows = [(members[j-d].core, members[j].core, slabs[o][2]*slabs[o][3])
                     for members, slabs in lines for o in range(len(members)) for d in (1, -1)
                     if 0 <= (j := o+d*hop) < len(members)]
            total += max(load(flows, side).values(), default=0)
    return total
