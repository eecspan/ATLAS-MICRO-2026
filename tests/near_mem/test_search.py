from dataclasses import replace

from frontend.near_mem.gemm.plan import Plan,Problem
from frontend.near_mem.gemm.compiler import compile_plan
from frontend.near_mem.gemm.space import candidates,family,groups,resident_bytes
from frontend.near_mem.search import shortlist


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
    score=lambda p:sum(p.tile)
    first,count=shortlist(lambda skip:iter(plans),score,8,family)
    second,_=shortlist(lambda skip:iter(plans),score,8,family)
    assert first==second and count==702
    assert len(first)>8  # Family retention is an explicit extra budget.
    assert any(p.c!=(1,1) for p,_ in first)


def test_proxy_bound_pruning_is_exact(tmp_path):
    from frontend.near_mem.gemm.space import Proxy,families,structure
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.search import diverse
    from frontend.near_mem.sim import make_hardware
    proxy=Proxy(make_hardware(tmp_path/'hw',4,64));problem=Problem(64,64,128);hw=dict(cores=4,sram=64*1024,dram=2**30,tile_min=8)
    for m in METHODS.values():
        space=lambda skip=None:m.space(problem,skip=skip,**hw)
        for p in space():
            k=proxy.kernel(problem,p)
            assert proxy.lower(problem,p)<=k*(1+1e-9) and proxy.lower(problem,replace(p,tile=None))<=k*(1+1e-9)
        score=lambda p:proxy(problem,p);bound=lambda q:proxy.lower(problem,q)+proxy.distribution(problem,q)
        assert shortlist(space,score,3,family)[0]==shortlist(space,score,3,family,bound,families)[0]
        pair=lambda p:(k:=proxy.kernel(problem,p),k+proxy.distribution(problem,p))
        low=lambda q:(k:=proxy.lower(problem,q),k+proxy.distribution(problem,q))
        assert diverse(space,pair,2,structure,2)[0]==diverse(space,pair,2,structure,2,low)[0]
