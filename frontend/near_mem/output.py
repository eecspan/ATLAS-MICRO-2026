"""Shared finite output contracts for additive and softmax reductions.

Contracts describe ownership, not the reduction algebra. A sharded result
keeps the compiler's owner; first/last collect at a group endpoint; replicated
retains one copy at every group member after a linear SRAM broadcast.
"""
from dataclasses import replace
from math import prod

POLICIES = ('sharded', 'first', 'last', 'replicated')


def root_index(policy, size, default):
    if policy not in POLICIES:
        raise ValueError(f'Unknown output contract: {policy}')
    return 0 if policy == 'first' else size-1 if policy == 'last' else default


def destinations(policy, group, default):
    root = root_index(policy, len(group), default)
    return tuple(group) if policy == 'replicated' else (group[root],)


def reduce_steps(group, root):
    """Two inward lines. Each destination merges before its next send."""
    for step in range(max(root, len(group)-1-root)):
        if step < root:
            yield step, group[step], group[step+1]
        if step < len(group)-1-root:
            yield step, group[-1-step], group[-2-step]


def broadcast_steps(group, root):
    """Outward lines: every non-root receives exactly once, no DRAM staging."""
    for step in range(max(root, len(group)-1-root)):
        if root-step > 0:
            yield step, group[root-step], group[root-step-1]
        if root+step < len(group)-1:
            yield step, group[root+step], group[root+step+1]


def variants(plan):
    """Boundary choices for a shortlisted plan, shared by all graph methods.

    Global-layout baselines keep their prescribed layout; without a residual
    reduction group the current contract has no extra choices.
    """
    from .gemm.plan import Plan
    from .model.attention import AttentionPlan
    if isinstance(plan, AttentionPlan):
        size = plan.context_parts
    elif isinstance(plan, Plan) and plan.layout == 'local':
        size = plan.ps[2]//prod(plan.c)
    else:
        size = 1
    return [replace(plan, output=p) for p in POLICIES] if size > 1 else [plan]
