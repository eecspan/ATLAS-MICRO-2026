from dataclasses import replace

from frontend.near_mem.plan import Plan,Problem
from frontend.near_mem.compiler import compile_plan
from frontend.near_mem.search import candidates,groups,resident_bytes,shortlist


def test_enumerated_space_is_legal_and_keeps_major_families():
    problem=Problem(8,8,8)
    plans=list(candidates(problem,4,8192,4,'all','all'))
    assert len(plans)==len(set(plans))==702
    for p in plans:
        p.validate(problem)
        assert compile_plan(problem,p,8192,1024**2).sram_bytes==resident_bytes(p)
    assert {p.loop for p in plans}=={'mnk','mkn','nmk','nkm','kmn','knm'}
    assert any(p.a[1]!=p.b[0] for p in plans)
    assert any(p.c!=(1,1) for p in plans)
    assert any(p.ps[2]>1 for p in plans)


def test_no_shift_group_canonicalization_preserves_exact_ir():
    problem=Problem(8,8,8)
    for loop in ('mnk','mkn','nmk','nkm','kmn','knm'):
        p=Plan((1,1,1),(2,2,2),loop=loop)
        reference=compile_plan(problem,p,8192,1024**2).phases
        for g in groups((4,4,4),loop):
            assert compile_plan(problem,replace(p,group=g),8192,1024**2).phases==reference


def test_shortlist_is_reproducible_and_preserves_distinct_families():
    problem=Problem(8,8,8)
    plans=list(candidates(problem,4,8192,4,'all','all'))
    score=lambda problem,p:sum(p.tile)
    first,count=shortlist(problem,iter(plans),score,8)
    second,_=shortlist(problem,iter(plans),score,8)
    assert first==second and count==702
    assert len(first)>8  # Family retention is an explicit extra budget.
    assert any(p.c!=(1,1) for p,_ in first)
