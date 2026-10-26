"""Standalone GEMM inputs: one DRAM copy per element; replicas are charged.

Any row/column partition of A and B is a free initial placement. A block needed
by r cores starts as r disjoint slabs, one per replica, and an ATLAS-style
linear all-gather (mesh rows, then columns) completes it. Native and extended
plans share this rule and collective.
"""
from collections import Counter, defaultdict

from ..tasks import Graph, Region
from .ownership import Ownership
from .plan import DEPENDENCIES
from ..sim import FIELDS


def replica_groups(problem, plan, tensor):
    own = Ownership(problem, plan); shape = [own.base[d] for d in DEPENDENCIES[tensor]]
    shapes = []; groups = defaultdict(list)
    for core in range(plan.cores):
        blocks = own.tensor_blocks(core, tensor)
        index = [sorted({b[i] for b in blocks}) for i in range(2)]
        shapes.append([len(index[0])*shape[0], len(index[1])*shape[1]])
        for b in blocks:
            groups[b].append(Region(core, tensor, b[0]*shape[0], b[1]*shape[1], *shape,
                                    index[0].index(b[0])*shape[0], index[1].index(b[1])*shape[1]))
    return shapes, [g for g in groups.values() if len(g) > 1]


def received_bytes(problem, plan):
    """Largest per-core inbound volume; divided by the flit size it bounds the cycles."""
    inbound = Counter()
    for v in 'AB':
        for group in replica_groups(problem, plan, v)[1]:
            for r in group:
                inbound[r.core] += r.area*problem.dtype_bytes*(len(group)-1)//len(group)
    return max(inbound.values(), default=0)


def distribute(problem, plan, chip, out):
    graph = Graph(chip); groups = []
    for v in 'AB':
        shapes, replicas = replica_groups(problem, plan, v)
        for core, shape in enumerate(shapes):
            graph.tensor(core, v, shape, dtype=problem.dtype_bytes, column=v == 'B')
        groups += replicas
    if not groups:
        return {k: 0 for k in FIELDS}
    graph.allgather('distribute', groups)
    return graph.simulate(out, timeout=7200)['result']


def charge(kernel, distribution):
    """Distribution precedes the kernel, so their cycles add."""
    result = dict(kernel, kernel_cycles=kernel['e2e_cycles'], distribution_cycles=distribution['e2e_cycles'])
    result['e2e_cycles'] += distribution['e2e_cycles']
    return result
