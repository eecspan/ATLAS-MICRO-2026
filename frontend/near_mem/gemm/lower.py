"""Native task lowering and execution for the general multi-axis program."""
from dataclasses import asdict
import hashlib
import json
from math import prod
from pathlib import Path
import subprocess
import sys

from frontend.auto.gemm_schedule import gemm_work

from .compiler import compile_plan
from .distribution import charge, distribute
from .plan import Problem, Plan
from ..sim import ROOT, KEYS, hardware, fingerprint


def lower_tasks(program, flit_size, final_ops=None):
    """Return native tasks without imposing a file or graph boundary."""
    p=program.plan;dtype=program.problem.dtype_bytes
    if dtype!=2:raise ValueError('This backend matches native FP16 SRAM payloads')
    if prod(p.tile)>=2**31:raise ValueError('Native signed matrix count would overflow')
    # ATLAS accounts SRAM accesses at compute/NoC consumers; its native GEMM
    # does not additionally charge DRAM staging or implicit accumulator clear.
    phases=[row for row in program.phases if any(a['op'] not in ('clear','zero','drop') for ops in row for a in ops)]
    def slot(t,op):return t if op=='load' else t-2 if op=='store' else t-1
    slots=[slot(t,a['op']) for t,row in enumerate(phases) for ops in row for a in ops if a['op'] not in ('clear','zero','drop')]
    if min(slots)<0:raise ValueError('Program starts before its operands can arrive')
    length=max(slots)+1
    raw=[[{k:[] for k in KEYS} for _ in range(length)] for _ in range(p.cores)]
    fid=macs=vectors=reads=writes=credits=0
    def buffer(core,i,count,write):
        raw[core][i]['buffer_store' if write else 'buffer_load'].append(dict(name='sram',byte_count=count,is_write=write))
    for i,row in enumerate(phases,-1):
        for core,actions in enumerate(row):
            for a in actions:
                op=a['op']
                if op in ('load','store'):
                    if op=='store' and a.get('final') and final_ops is not None:
                        a,operations,read_bytes,write_bytes = final_ops(a)
                        if operations:
                            raw[core][i-1]['vector'].append(dict(name='output_epilogue',vec_count=operations))
                            buffer(core,i-1,read_bytes,False);buffer(core,i-1,write_bytes,True)
                            vectors+=operations
                    write=op=='store';count=prod(a['shape'])*dtype;rank=len(a['base'])
                    task=dict(name=a['tensor'],is_write=write,access_base=a['base'],access_extent=a['extent'],
                              access_stride_add=[1]*rank,access_offset_add=[0]*rank,init_iter=0,stride_iter=1,total_iter=1)
                    raw[core][i-1 if write else i+1]['dram'].append(task)
                    reads+=0 if write else count;writes+=count if write else 0
                elif op=='gemm':
                    m,n,k=p.tile;count=m*n*k;c=m*n*dtype
                    for key,tasks in gemm_work(p.tile,dtype).items():raw[core][i][key].extend(tasks)
                    macs+=count;vectors+=m*n
                elif op=='add':
                    count=a['byte_count']//dtype;vectors+=count
                    raw[core][i]['vector'].append(dict(name='split_k_add',vec_count=count))
                    buffer(core,i,a['byte_count']*2,False);buffer(core,i,a['byte_count'],True)
                elif op in ('send','credit'):
                    count=1 if op=='credit' else a['byte_count'];flits=(count+flit_size-1)//flit_size
                    task=dict(name=a['tensor'],src=core,dst=a['dst'],flit_num=flits,init_flit_id=fid)
                    raw[core][i]['noc_tx'].append(dict(task,is_send=True))
                    raw[a['dst']][i]['noc_rx'].append(dict(task,is_send=False));fid+=flits
                    if op=='send':buffer(core,i,count,False);buffer(a['dst'],i,count,True)
                    else:credits+=flits*flit_size
                elif op not in ('clear','zero','drop'):raise ValueError(f'Unknown IR operation: {op}')
    assert macs==prod(program.problem.shape)
    if fid>=2**31:raise ValueError('Native flit identifiers would overflow')
    manifest = dict(problem=asdict(program.problem),plan=asdict(p),sram_bytes=program.sram_bytes,
                macs=macs,vector_ops=vectors,logical_dram_read_bytes=reads,logical_dram_write_bytes=writes,
                noc_bytes=fid*flit_size,noc_credit_bytes=credits,phases=len(program.phases),
                accounting='native ATLAS compute/NoC SRAM accesses; DRAM staging not charged twice',
                scope='single GEMM, explicit full-resident panels and final reduction; preplaced shards')

    return raw, manifest


def lower(program, out, flit_size):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    raw, manifest=lower_tasks(program,flit_size)
    length=len(raw[0])
    for core,execution in enumerate(raw):
        (out/f'core_{core}.yaml').write_text(json.dumps(dict(iteration=length,execution=execution)))
    (out/'operators.yaml').write_text(json.dumps(dict(operator=[dict(name='joint_gemm',type='mpmd-general',iteration=length,file_prefix=str(out))])))
    (out/'placement.yaml').write_text(json.dumps(dict(core_tensor=program.placements)))
    return manifest


def simulate(problem,plan,chip,out,timeout=300,overlap=True,provenance=None):
    chip=Path(chip).resolve();out=Path(out).resolve()
    arch,dram,_=hardware(chip)
    if arch['core_num']!=plan.cores:raise ValueError('Core count mismatch')
    program=compile_plan(problem,plan,arch['core']['buffer']['buffer_size']*1024,dram,overlap)
    manifest=lower(program,out,arch['noc']['flit_size'])
    manifest.update(overlap=overlap,sha256=provenance or fingerprint(chip))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'result.json').unlink(missing_ok=True)
    with (out/'simulator.log').open('w') as log:
        subprocess.run([sys.executable,'-m','frontend.near_mem.sim','--worker','--chip',str(chip),'--output',str(out)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    result=json.loads((out/'result.json').read_text())
    assert result['flop_count']==2*manifest['macs']+manifest['vector_ops']
    return dict(manifest=manifest,result=result)


def measure(problem,plan,chip,cache,provenance):
    """ATLAS cycles of one plan, including the single-copy input distribution."""
    signature=dict(problem=asdict(problem),plan=asdict(plan),backend=provenance,overlap=True,inputs='single-copy')
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:24]
    out=Path(cache)/key
    if (out/'result.json').exists() and (out/'signature.json').exists():
        if json.loads((out/'signature.json').read_text())==json.loads(json.dumps(signature)):
            return dict(key=key,plan=asdict(plan),result=json.loads((out/'result.json').read_text()),path=str(out))
    run=simulate(problem,plan,chip,out,provenance=provenance)
    result=charge(run['result'],distribute(problem,plan,chip,out/'distribution'))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'signature.json').write_text(json.dumps(signature,sort_keys=True))
    return dict(key=key,plan=asdict(plan),result=result,path=str(out))


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',required=True,type=Path,help='JSON with problem and plan objects')
    ap.add_argument('--chip',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();cfg=json.loads(args.plan.read_text())
    p={k:tuple(v) if isinstance(v,list) else v for k,v in cfg['plan'].items()}
    print(json.dumps(simulate(Problem(**cfg['problem']),Plan(**p),args.chip,args.output),indent=2))
