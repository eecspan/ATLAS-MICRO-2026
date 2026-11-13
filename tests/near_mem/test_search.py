import pytest
from dataclasses import replace

from frontend.near_mem.gemm.plan import Plan,Problem
from frontend.near_mem.gemm.compiler import compile_plan
from frontend.near_mem.gemm.space import candidates,family,groups,resident_bytes
from frontend.near_mem.search import shortlist


def test_enumerated_space_is_legal_and_keeps_major_families():
    problem=Problem(8,8,8)
    plans=list(candidates(problem,4,8192,4,'all','all',distinct=False))
    assert len(plans)==len(set(plans))==702
    for p in plans:
        p.validate(problem)
        # ``resident_bytes`` is the baseline (waiting C) footprint; ``auto``
        # adds the deferred third C panel only when it fits.
        program=compile_plan(problem,p,8192,1024**2)
        assert program.sram_bytes==p.sram_bytes(2,True,program.defer)
        assert resident_bytes(p)<=program.sram_bytes<=8192
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


def test_distinct_space_keeps_one_plan_per_compiled_program(tmp_path):
    # Loop orders that differ only in where one-iteration loops sit compile to
    # the same program and score; ``distinct`` keeps the first of them only.
    from frontend.near_mem.gemm.space import Proxy,schedule_key
    from frontend.near_mem.sim import make_hardware
    proxy=Proxy(make_hardware(tmp_path/'hw',4,64))
    for problem,receive in ((Problem(8,8,8),'sram_consume'),(Problem(16,32,16),'sram_consume'),
                            (Problem(16,16,16),'dram_land')):
        args=(problem,4,1<<20,4,'all','elk',1<<30,True)
        kept=list(candidates(*args,receive=receive))
        every=list(candidates(*args,receive=receive,distinct=False))
        assert len(set(kept))==len(kept)<len(every) and set(kept)<=set(every)
        counts=lambda p:tuple(n//(s*q*t) for n,s,q,t in zip(problem.shape,p.ps,p.periods,p.tile))
        identity=lambda p:(replace(p,loop='mnk'),schedule_key(p,counts(p)))
        index={identity(p):p for p in kept}
        assert len(index)==len(kept)
        program=lambda p:compile_plan(problem,p,1<<20,1<<30).phases
        for p in every:
            twin=index[identity(p)]
            assert program(p)==program(twin) and proxy.kernel(problem,p)==proxy.kernel(problem,twin),p


def test_shortlist_is_reproducible_and_preserves_distinct_families():
    problem=Problem(8,8,8)
    plans=list(candidates(problem,4,8192,4,'all','all',distinct=False))
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
    cases=[(m,None) for m in METHODS.values() if not m.native]+[(METHODS['tp'],d) for d in (0,1)]
    for m,direction in cases:
        space=lambda skip=None:m.space(problem,skip=skip,direction=direction,**hw)
        for p in space():
            k=proxy.kernel(problem,p)
            assert proxy.lower(problem,p)<=k*(1+1e-9)
            if not m.flash:assert proxy.lower(problem,replace(p,tile=None))<=k*(1+1e-9)
        score=lambda p:proxy(problem,p);bound=lambda q:proxy.lower(problem,q)+proxy.distribution(problem,q)
        fam=lambda q:families(q,m.loops)
        assert shortlist(space,score,3,family)[0]==shortlist(space,score,3,family,bound,fam)[0]
        pair=lambda p:(k:=proxy.kernel(problem,p),k+proxy.distribution(problem,p))
        low=lambda q:(k:=proxy.lower(problem,q),k+proxy.distribution(problem,q))
        assert diverse(space,pair,2,structure,2)[0]==diverse(space,pair,2,structure,2,low)[0]


def test_t10_padding_metadata_is_used_during_proxy_pruning(tmp_path):
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.gemm.space import Proxy,families
    from frontend.near_mem.search import shortlist
    from frontend.near_mem.sim import make_hardware
    problem=Problem(1,30,31);chip=make_hardware(tmp_path/'hw',16,64)
    method=METHODS['t10'];hw=dict(cores=16,sram=64*1024,dram=2**30,tile_min=1)
    space=lambda skip=None:method.space(problem,skip=skip,**hw)
    proxy=Proxy(chip)
    score=lambda p:proxy(problem,p)
    bound=lambda q:proxy.lower(problem,q)+proxy.distribution(problem,q)
    ranked,count=shortlist(space,score,4,family,bound,lambda q:families(q,method.loops))
    assert count and ranked
    assert all(p.logical_shape==problem.shape and p.execution_shape==(1,32,32) for p,_ in ranked)


def test_tp_mesh_padding_is_scored_as_physical_work(tmp_path):
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.gemm.plan import execution_problem
    from frontend.near_mem.gemm.space import Proxy
    from frontend.near_mem.sim import make_hardware
    logical=Problem(1,5,7);chip=make_hardware(tmp_path/'hw',4,64)
    plan=next(METHODS['tp'].space(logical,cores=4,sram=64*1024,dram=1<<30,direction=0))
    physical=execution_problem(logical,plan);proxy=Proxy(chip)
    assert physical.shape==(1,6,8)
    assert proxy.kernel(logical,plan)==proxy.kernel(physical,plan)


def test_m1_tiny_tp_distribution_skips_empty_allgather_slabs(tmp_path):
    from frontend.near_mem.gemm.distribution import distribute
    from frontend.near_mem.gemm.plan import execution_problem
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.sim import ROOT,make_hardware
    if not list((ROOT/'simulator/build/lib').glob('atlasim.*.so')):
        pytest.skip('Build ATLAS first')
    logical=Problem(1,5,7)
    plan=next(METHODS['tp'].space(logical,cores=16,sram=64*1024,dram=1<<30))
    physical=execution_problem(logical,plan)
    result=distribute(physical,plan,make_hardware(tmp_path/'hw',16,64),tmp_path/'distribution')
    assert result['e2e_cycles']>0


def test_measure_all_deadline_drops_aborted_candidates():
    from subprocess import TimeoutExpired
    import time
    from frontend.near_mem.search import measure_all
    stop=time.monotonic()+0.3
    def measure(x):
        if x and time.monotonic()>stop:raise TimeoutExpired('x',0)
        time.sleep(0.2);return x
    rows=measure_all(range(50),measure,4,deadline=stop)
    assert rows[0]==0 and len(rows)<50
    with pytest.raises(TimeoutExpired):
        measure_all(range(3),lambda x:(_ for _ in ()).throw(TimeoutExpired('x',0)),2,deadline=stop)


def test_proxy_transfer_counts_match_the_compiled_program():
    from collections import Counter
    from frontend.near_mem.gemm.space import transfers
    problem=Problem(16,16,16)
    for receive in ('sram_consume','dram_land'):
        for p in candidates(problem,4,1<<30,4,'all','elk',1<<30,True,receive=receive):
            if p.ps[2]//(p.c[0]*p.c[1])>1:continue  # split-K finalization is charged apart
            counted=Counter((a['tensor'],a['op']) for row in compile_plan(problem,p,1<<30,1<<30).phases
                            for a in row[0] if a['op'] in ('load','store'))
            expected=transfers(problem,p)
            assert (counted['A','load'],counted['B','load'])==(expected['A'][0],expected['B'][0]),p


def test_proxy_split_k_finalization_counts_the_busiest_rank():
    from collections import Counter
    from math import ceil, prod
    from frontend.near_mem.gemm.space import transfers
    from frontend.near_mem.output import variants
    problem=Problem(8,8,16)
    for base in candidates(problem,4,1<<30,4,'all','elk',1<<30,True):
        residual=base.ps[2]//prod(base.c)
        if residual<=1:continue
        for p in variants(base):
            per=[Counter() for _ in range(p.cores)]
            for row in compile_plan(problem,p,1<<30,1<<30).phases:
                for core,ops in enumerate(row):
                    per[core].update(a['op'] for a in ops if a.get('tensor')=='C' and a['op'] in ('load','store'))
            loads,stores=transfers(problem,p)['C']
            tiles=(problem.m//p.ps[0]//p.tile[0])*(problem.n//p.ps[1]//p.tile[1])//prod(p.c)
            final=tiles if p.output!='sharded' else ceil(tiles/residual)
            assert (max(c['load'] for c in per),max(c['store'] for c in per))==(loads+tiles,stores+final),p


def test_split_diverse_scan_is_exact(tmp_path):
    # A scan split into pieces of whole families, each pruned by its own
    # bound, merges into exactly the single scan's proposals and order.
    from frontend.near_mem.gemm.space import Proxy,structure
    from frontend.near_mem.methods import METHODS
    from frontend.near_mem.search import diverse,diverse_part,merge_diverse
    from frontend.near_mem.sim import make_hardware
    proxy=Proxy(make_hardware(tmp_path/'hw',4,64));problem=Problem(64,64,128);hw=dict(cores=4,sram=64*1024,dram=2**30,tile_min=8)
    pair=lambda p:(k:=proxy.kernel(problem,p),k+proxy.distribution(problem,p))
    low=lambda q:(k:=proxy.lower(problem,q),k+proxy.distribution(problem,q))
    for m in [m for m in METHODS.values() if not m.native]:
        space=lambda skip=None:m.space(problem,skip=skip,**hw)
        whole,count=diverse(space,pair,2,structure,2,low)
        for parts in (1,3,7):
            pieces=[diverse_part(space,pair,2,structure,2,low,i,parts) for i in range(parts)]
            merged=merge_diverse([r for rows,_ in pieces for r in rows],pair,2,structure,2)
            assert merged==whole,(m.name,parts)
            assert parts>1 or pieces[0][1]==count
