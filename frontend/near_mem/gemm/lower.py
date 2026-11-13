"""Native task lowering and execution for the general multi-axis program."""
from dataclasses import asdict
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import time
import sys

from frontend.auto.gemm_schedule import gemm_work

from .compiler import compile_kernel, compile_plan
from .distribution import charge, distribute
from .layout import GlobalLayout
from .plan import Problem, load, execution_problem
from ..sim import ROOT, KEYS, hardware, fingerprint


def prefetch_windows(phases):
    """Read issue/use positions for the compiler's two local A/B banks.

    A new panel may start while the preceding panel computes, but never
    crosses another load or a send/receive of that tensor: those occupy the
    second bank. Thus B can prefetch across A rotations (and conversely),
    without inventing a third bank for a tensor that is itself in flight.
    """
    uses = {}; sends = {}; ready = {}
    for t in range(len(phases)-1,-1,-1):
        for core,actions in enumerate(phases[t]):
            # Uses in this phase precede loads appended by the pipeline.
            for a in actions:
                if a['op'] == 'load' and a['tensor'] in ('A','B'):
                    v = a['tensor']
                    end = min(uses.get((core,v,a['key']),len(phases)),sends.get((core,v),len(phases)))
                    if end < len(phases):ready[id(a)] = end
            for a in actions:
                if a['op'] == 'gemm':
                    m,n,k = a['index']
                    uses[core,'A',(m,k)] = t; uses[core,'B',(k,n)] = t
                elif a['op'] == 'send' and a['tensor'] in ('A','B'):
                    sends[core,a['tensor']] = t
    first_compute = {}; windows = {}
    for t,row in enumerate(phases):
        touched = set()
        for core,actions in enumerate(row):
            for v in ('A','B'):
                key = (core,v)
                if any(a['op']=='gemm' for a in actions):first_compute.setdefault(key,t)
                for a in actions:
                    if a['op']=='load' and a['tensor']==v:
                        issue = first_compute.get(key,t)
                        if id(a) in ready:windows[id(a)] = (issue,ready[id(a)])
                        touched.add(key)
                    elif a['op']=='send' and a['tensor']==v:
                        touched.update((key,(a['dst'],v)))
        for key in touched:first_compute.pop(key,None)
    return windows


def lower_tasks(program, flit_size, final_ops=None):
    """Return native tasks without imposing a file or graph boundary.

    A program phase t computes at simulator iteration t and loads what phase
    t+1 consumes. A tile read from another core's DRAM (``layout='global'``)
    is read by its home one iteration earlier and sent in iteration t; a
    final output tile owned elsewhere is sent at t and written by its home.
    A DRAM-landing ``forward`` reads at t-1, sends at t and lands at t+1.
    """
    p=program.plan;dtype=program.problem.dtype_bytes
    if dtype!=2:raise ValueError('This backend matches native FP16 SRAM payloads')
    if prod(p.tile)>=2**31:raise ValueError('Native signed matrix count would overflow')
    glob=GlobalLayout(program.problem,p.cores) if p.layout=='global' else None
    # Match ATLAS's explicit compute/NoC SRAM traffic. DRAM staging does not
    # add separate SRAM-port accesses; this is a shared DMA abstraction, not
    # a claim that Ramulator already charges those accesses. Capacity still
    # includes the staging banks, and implicit accumulator clear is omitted.
    phases=[row for row in program.phases if any(a['op'] not in ('clear','zero','drop','swap') for ops in row for a in ops)]
    if glob:phases.insert(0,[[] for _ in range(p.cores)])  # room for the first remote reads
    windows = prefetch_windows(phases) if program.overlap and not glob and not p.land else {}
    length=len(phases)+1
    raw=[[{k:[] for k in KEYS} for _ in range(length)] for _ in range(p.cores)]
    fid=macs=vectors=reads=writes=0
    finalized={}
    def cell(core,slot):
        if slot<0:raise ValueError('Program starts before its operands can arrive')
        return raw[core][slot]
    def buffer(core,i,count,write):
        cell(core,i)['buffer_store' if write else 'buffer_load'].append(dict(name='sram',byte_count=count,is_write=write))
    def dram(core,slot,name,base,extent,write,ready=None):
        nonlocal reads,writes
        rank=len(base);count=prod(extent)*dtype
        cell(core,slot)['dram'].append(dict(name=name,is_write=write,access_base=list(base),access_extent=list(extent),
                                            access_stride_add=[1]*rank,access_offset_add=[0]*rank,
                                            init_iter=0,stride_iter=1,total_iter=1,
                                            **({} if ready is None else dict(ready_iter=ready))))
        reads+=0 if write else count;writes+=count if write else 0
    def send(slot,src,dst,name,count):
        nonlocal fid
        flits=(count+flit_size-1)//flit_size
        task=dict(name=name,src=src,dst=dst,flit_num=flits,init_flit_id=fid)
        cell(src,slot)['noc_tx'].append(dict(task,is_send=True))
        cell(dst,slot)['noc_rx'].append(dict(task,is_send=False));fid+=flits
        buffer(src,slot,count,False);buffer(dst,slot,count,True)
    for i,row in enumerate(phases,-1):
        for core,actions in enumerate(row):
            for a in actions:
                op=a['op']
                if op in ('load','store'):
                    if op=='store' and a.get('final') and final_ops is not None:
                        ready=a['key'] in finalized
                        a,operations,read_bytes,write_bytes = final_ops(a)
                        if operations and not ready:
                            cell(core,i-1)['vector'].append(dict(name='output_epilogue',vec_count=operations))
                            buffer(core,i-1,read_bytes,False);buffer(core,i-1,write_bytes,True)
                            vectors+=operations
                    write=op=='store'
                    if glob and (a['tensor']!='C' or a.get('final')) and a['tensor']!='partial':
                        # Plan-independent homes: split the tile over the blocks it touches.
                        origin=[k*n for k,n in zip(a['key'],a['shape'])]
                        for home,base,extent in glob.pieces(a['tensor'],*origin,*a['shape']):
                            name=a['tensor']+'g' if write else a['tensor']
                            if home==core:dram(core,i-1 if write else i+1,name,base,extent,write)
                            elif write:
                                send(i,core,home,'C',prod(extent)*dtype);dram(home,i,name,base,extent,True)
                            else:
                                dram(home,i,name,base,extent,False);send(i,home,core,a['tensor'],prod(extent)*dtype)
                    else:
                        issue,ready = windows.get(id(a),(i+1,None))
                        dram(core,i-1 if write else issue,a.get('name',a['tensor']),a['base'],a['extent'],write,ready)
                elif op=='forward':
                    dram(core,i,a['name'],a['base'],a['extent'],False)
                    send(i,core,a['dst'],a['tensor'],a['byte_count'])
                    name,base,extent=a['dst_name'],a['base'],a['extent']
                    if a.get('final') and final_ops is not None:
                        # The copy passes the receiver's SRAM: apply the epilogue there.
                        out,operations,read_bytes,write_bytes=final_ops(dict(op='store',tensor='C',key=a['key'],
                                                    shape=extent,base=base,extent=extent,final=True))
                        cell(a['dst'],i)['vector'].append(dict(name='output_epilogue',vec_count=operations))
                        buffer(a['dst'],i,read_bytes,False);buffer(a['dst'],i,write_bytes,True);vectors+=operations
                        name,base,extent=('C' if out['tensor']=='C' else out['tensor']),out['base'],out['extent']
                    dram(a['dst'],i,name,base,extent,True)
                elif op=='gemm':
                    m,n,k=p.tile;count=m*n*k
                    for key,tasks in gemm_work(p.tile,dtype).items():cell(core,i)[key].extend(tasks)
                    macs+=count;vectors+=m*n
                elif op=='add':
                    count=a['byte_count']//dtype;vectors+=count
                    cell(core,i)['vector'].append(dict(name='split_k_add',vec_count=count))
                    buffer(core,i,a['byte_count']*2,False);buffer(core,i,a['byte_count'],True)
                elif op=='finalize':
                    out,operations,read_bytes,write_bytes=final_ops(a) if final_ops else (a,0,0,0)
                    finalized[a['key']]=out
                    if operations:
                        cell(core,i)['vector'].append(dict(name='output_epilogue',vec_count=operations))
                        buffer(core,i,read_bytes,False);buffer(core,i,write_bytes,True);vectors+=operations
                elif op in ('send','copy'):
                    count=a['byte_count']
                    if op=='copy' and a['key'] in finalized:
                        count=prod(finalized[a['key']]['extent'])*dtype
                    send(i,core,a['dst'],a['tensor'],count)
                elif op not in ('clear','zero','drop','nop','swap'):raise ValueError(f'Unknown IR operation: {op}')
    while not any(any(x[-1].values()) for x in raw):
        for x in raw:x.pop()
    assert macs==prod(program.problem.shape)
    if fid>=2**31:raise ValueError('Native flit identifiers would overflow')
    manifest = dict(problem=asdict(program.problem),plan=asdict(p),sram_bytes=program.sram_bytes,
                macs=macs,vector_ops=vectors,logical_dram_read_bytes=reads,logical_dram_write_bytes=writes,
                noc_bytes=fid*flit_size,phases=len(program.phases),
                accounting='native ATLAS compute/NoC SRAM accesses; extra DRAM staging SRAM accesses omitted')
    return raw, manifest


def lower(program, out, flit_size):
    from math import isqrt
    from ..resources import count
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    raw, manifest=program.lower(flit_size)
    tensors=[{t['name']:t for t in placement['tensor']} for placement in program.placements]
    manifest['resources']=count(raw,False,tensors,flit_size,isqrt(len(raw)))
    length=len(raw[0])
    for core,execution in enumerate(raw):
        (out/f'core_{core}.yaml').write_text(json.dumps(dict(iteration=length,execution=execution)))
    (out/'operators.yaml').write_text(json.dumps(dict(operator=[dict(name='joint_gemm',type='mpmd-general',iteration=length,file_prefix=str(out))])))
    (out/'placement.yaml').write_text(json.dumps(dict(core_tensor=program.placements)))
    return manifest


def simulate(problem,plan,chip,out,timeout=7200,overlap=True,provenance=None,deadline=None):
    chip=Path(chip).resolve();out=Path(out).resolve()
    if deadline is not None:
        timeout=min(timeout,deadline-time.monotonic())
        if timeout<=0:raise subprocess.TimeoutExpired('frontend.near_mem.sim',0)
    logical_shape=plan.logical_shape or problem.shape
    problem=execution_problem(problem,plan)
    arch,dram,_=hardware(chip)
    if arch['core_num']!=plan.cores:raise ValueError('Core count mismatch')
    program=(compile_plan(problem,plan,arch['core']['buffer']['buffer_size']*1024,dram,overlap) if not overlap else
             compile_kernel(problem,plan,arch['core']['buffer']['buffer_size']*1024,dram))
    manifest=lower(program,out,arch['noc']['flit_size'])
    manifest.update(logical_problem_shape=list(logical_shape),execution_problem_shape=list(problem.shape))
    manifest.update(overlap=overlap,sha256=provenance or fingerprint(chip))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'result.json').unlink(missing_ok=True)
    with (out/'simulator.log').open('w') as log:
        subprocess.run([sys.executable,'-m','frontend.near_mem.sim','--worker','--chip',str(chip),'--output',str(out)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    result=json.loads((out/'result.json').read_text())
    assert result['flop_count']==2*manifest['macs']+manifest['vector_ops']
    return dict(manifest=manifest,result=result)


def measure(problem,plan,chip,folder,provenance,deadline=None):
    """ATLAS cycles of one plan, including the single-copy input distribution.
    ``deadline`` (``time.monotonic()``) aborts a run that cannot finish in time."""
    problem=execution_problem(problem,plan)
    key=hashlib.sha256(json.dumps(asdict(plan),sort_keys=True).encode()).hexdigest()[:24]
    out=Path(folder)/key
    run=simulate(problem,plan,chip,out,provenance=provenance,deadline=deadline)
    result=charge(dict(run['result'],resources=run['manifest']['resources']),
                  distribute(problem,plan,chip,out/'distribution',deadline))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    return dict(key=key,plan=asdict(plan),result=result,path=str(out))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',required=True,type=Path,help='JSON with problem and plan objects')
    ap.add_argument('--chip',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();cfg=json.loads(args.plan.read_text())
    print(json.dumps(simulate(Problem(**cfg['problem']),load(cfg['plan']),args.chip,args.output),indent=2))
