"""Shared two-stage search: a cheap proxy proposes, the simulator decides.

Single GEMMs and whole Transformer graphs follow one contract: a space yields
candidates, ``score`` ranks them analytically, and only the best few (plus
explicitly retained incumbents) are measured by ATLAS. Proposal strategies are
interchangeable: ``shortlist``/``diverse`` scan a finite space exactly,
``beam`` chooses jointly over a sequence of operators, and ``anneal``
explores spaces too large to enumerate.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import heapq
import math
import random


def _bounded(bound, values):
    """``True`` when every component of a lower bound reaches its threshold."""
    bound = bound if isinstance(bound, tuple) else (bound,)
    return all(b-1e-9*abs(b) >= v for b, v in zip(bound, values))


def shortlist(space, score, budget, family=None, bound=None, families=None):
    """Exact top-``budget`` by score, plus the best candidate of each family.

    Families keep structurally different choices (e.g. rotation patterns) from
    being crowded out by one approximate score. ``space(skip)`` yields the
    candidates; with a lower bound ``bound(prefix)`` of ``score`` and the
    ``families(prefix)`` its completions can belong to, prefixes that could
    neither enter the top list nor improve a family are skipped, which leaves
    the result unchanged. Returns ``[(candidate, score)]`` in ascending score
    order and the number of candidates scored.
    """
    heap = []; best = {}; count = 0
    def skip(prefix):
        if len(heap) < budget or not _bounded(bound(prefix), (-heap[0][0],)):
            return False
        return family is None or all(f in best and _bounded(bound(prefix), (best[f][0],)) for f in families(prefix))
    for count, c in enumerate(space(skip if bound else None), 1):
        value = score(c)
        if len(heap) < budget:
            heapq.heappush(heap, (-value, -count, c))
        elif value < -heap[0][0]:
            heapq.heapreplace(heap, (-value, -count, c))
        if family is not None:
            key = family(c)
            if key not in best or value < best[key][0]:
                best[key] = (value, count, c)
    ranked = sorted([(-s, -i, c) for s, i, c in heap]+list(best.values()), key=lambda x: x[:2])
    unique = {}
    for value, _, c in ranked:
        unique.setdefault(c, value)
    return list(unique.items()), count


def diverse(space, score, budget, family, per=1, bound=None):
    """The ``per`` best candidates of each of the ``budget`` best families.

    Used where every measurement is expensive (whole graphs): slots go to
    distinct structural decisions instead of near-duplicate tilings of the
    best one, while ``per > 1`` still lets the simulator correct the proxy's
    choice of tile within a family. ``score`` may return a tuple; each
    component then selects its own families and the lists are merged, so
    several proxies share one scan of the space.

    ``space(skip)`` yields the candidates. With a lower bound ``bound(prefix)``
    of ``score``, a prefix is skipped when, for every component, it can
    neither enter its family's ``per`` best nor lift its family into the
    ``budget`` best seen so far. Exact when every family arrives contiguously
    and its first prefix bounds all of it (``gemm.space.structure``).
    """
    tables = []; finished = []; count = 0; family_bound = {}
    tup = lambda v: v if isinstance(v, tuple) else (v,)

    def skip(prefix):
        key = family(prefix); low = tup(bound(prefix))
        if key not in family_bound:
            for j, table in enumerate(tables):   # the previous family is complete
                for other, heap in table.items():
                    if other not in finished[j][1] and heap:
                        finished[j][1].add(other); finished[j][0].append(min(-v for v, _, _ in heap))
                finished[j][0].sort(); del finished[j][0][budget:]
            family_bound[key] = low
        for j, value in enumerate(low):
            heap = tables[j].get(key, []) if tables else []
            if len(heap) < per or value < -heap[0][0]*(1+1e-9):
                best, done = (finished[j][0], True) if finished else ([], False)
                if not done or len(best) < budget or family_bound[key][j] < best[-1]*(1+1e-9):
                    return False
        return True

    for count, c in enumerate(space(skip if bound else None), 1):
        values = tup(score(c))
        if not tables:
            tables = [{} for _ in values]; finished = [([], set()) for _ in values]
        key = family(c)
        for value, table in zip(values, tables):
            heap = table.setdefault(key, [])
            if len(heap) < per:
                heapq.heappush(heap, (-value, -count, c))
            elif value < -heap[0][0]:
                heapq.heapreplace(heap, (-value, -count, c))
    chosen = {}
    for table in tables:
        families = sorted((sorted(((-v, -i, c) for v, i, c in heap), key=lambda x: x[:2]) for heap in table.values()),
                          key=lambda x: x[0][:2])
        for group in families[:budget]:
            for v, _, c in group:
                chosen.setdefault(c, v)
    return list(chosen.items()), count


def beam(decisions, cost, width=64, keep=8):
    """Beam search over a sequence of discrete decisions.

    ``decisions`` lists ``(name, options)``; ``cost(name, option, partial)``
    is the incremental cost of choosing ``option`` given the earlier choices
    in ``partial`` (so pairwise terms are charged once, at the later one).
    Exact when ``width`` covers every partial assignment. Returns the
    ``keep`` cheapest complete assignments as ``[(cost, {name: option})]``.
    """
    states = [(0.0, {})]
    for name, options in decisions:
        states = heapq.nsmallest(width, ((c+cost(name, o, s), {**s, name: o}) for c, s in states for o in options),
                                 key=lambda x: x[0])
    return states[:keep]


def measure_all(candidates, measure, workers=1, processes=False):
    """Measure each distinct candidate once, preserving order.

    Use ``processes`` when ``measure`` spends real Python time lowering tasks
    before the simulator subprocess runs; it must then be picklable.
    """
    unique = list(dict.fromkeys(candidates))
    pool_type = ProcessPoolExecutor if processes else ThreadPoolExecutor
    with pool_type(max_workers=max(1, workers)) as pool:
        return list(pool.map(measure, unique))


def best(rows):
    return min(rows, key=lambda r: r['result']['e2e_cycles'])
