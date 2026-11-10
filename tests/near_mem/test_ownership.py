from dataclasses import replace
from itertools import permutations, product

import numpy as np
import pytest

from frontend.near_mem.gemm.plan import Plan, Problem
from frontend.near_mem.gemm.ownership import Ownership


def pairs(n):
    return [(a,b) for a in range(1,n+1) for b in range(1,n+1) if n%(a*b)==0]


def legal_plans():
    problem=Problem(16,16,16)
    for ps in ((4,4,1),(2,4,2),(1,4,4),(4,1,4),(2,2,4)):
        for a,b,c in product(pairs(ps[1]),pairs(ps[0]),pairs(ps[2])):
            p=Plan(ps,(1,1,1),a,b,c)
            try:
                p.validate(problem)
            except ValueError:
                continue
            yield p


@pytest.mark.parametrize('plan',list(legal_plans()))
def test_all_legal_temporal_partitions_and_layer_orders(plan):
    problem=Problem(16,16,16)
    rng=np.random.default_rng(7)
    a,b=rng.normal(size=(16,16)),rng.normal(size=(16,16))
    for order in permutations(range(len(plan.layers))):
        ownership=Ownership(problem,replace(plan,layer_order=order))
        np.testing.assert_allclose(ownership.replay(a,b),a@b,atol=1e-12)
        for i,layer in enumerate(ownership.layers):
            for tensor in layer.tensors:
                dst=[ownership.neighbor(core,i,tensor,-1) for core in range(plan.cores)]
                assert sorted(dst)==list(range(plan.cores))
                assert all(ownership.neighbor(d,i,tensor,1)==c for c,d in enumerate(dst))


def test_mixed_axes_and_c_rotation_with_nontrivial_layer_order():
    problem=Problem(16,16,16)
    plan=Plan((4,4,4),(1,1,1),(2,2),(2,2),(2,2),layer_order=(2,0,1))
    rng=np.random.default_rng(4)
    a,b=rng.normal(size=(16,16)),rng.normal(size=(16,16))
    np.testing.assert_allclose(Ownership(problem,plan).replay(a,b),a@b,atol=1e-12)


def test_nonbinary_unequal_periods():
    problem=Problem(12,12,12)
    plan=Plan((2,6,1),(1,1,1),a=(1,6),b=(2,1))
    a=np.arange(144).reshape(12,12)/11
    b=np.arange(144).reshape(12,12)/7
    np.testing.assert_allclose(Ownership(problem,plan).replay(a,b),a@b)


def test_rejects_illegal_space_instead_of_approximating_it():
    with pytest.raises(ValueError,match='gcd'):
        Plan((3,2,1),(1,1,1),a=(1,2),b=(3,1)).validate(Problem(12,12,12))
    with pytest.raises(ValueError,match='replica'):
        Plan((2,2,1),(1,1,1),a=(2,2)).validate(Problem(16,16,16))
    with pytest.raises(ValueError,match='saturate'):
        Plan((1,1,1),(2,2,2),group=(1,2,1)).validate(Problem(16,16,16))
    with pytest.raises(ValueError,match='panel'):
        Plan((2,2,1),(1,1,1),a=(1,2),panel=(1000,1,1)).validate(Problem(16,16,16))


def test_unequal_periods_follow_uniform_rings():
    # The paper's 2x4 example: X (A) rotates over the 4-core row, W (B) over
    # the 2-core column; core (r,c) computes k_q at step s with q=(c+2r+s)%4.
    problem=Problem(8,16,16)
    own=Ownership(problem,Plan((2,4,1),(4,4,4),a=(1,4),b=(2,1)))
    steps=[state for event,state in own.events() if event=='compute']
    for s,state in enumerate(steps):
        assert [own.block(core,state)[2] for core in range(8)]==[(c+2*r+s)%4 for r in range(2) for c in range(4)]
    # Every A transfer is between ring neighbours of the row.
    for layer in range(len(own.layers)):
        assert all(abs(own.neighbor(core,layer,'A',-1)%4-core%4) in (1,3) for core in range(8))
    rng=np.random.default_rng(5);a=rng.normal(size=(8,16));b=rng.normal(size=(16,16))
    np.testing.assert_allclose(own.replay(a,b),a@b,atol=1e-12)
