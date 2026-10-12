"""Native task lowering and execution for the general multi-axis program."""
from dataclasses import asdict
import hashlib
import json
from math import prod
from pathlib import Path
import re
import subprocess
import sys

import yaml
from frontend.auto.gemm_schedule import gemm_work

from .compiler import compile_plan
from .plan import Problem, Plan
from .run import ROOT, revision


KEYS=('matrix','vector','buffer_load','buffer_store','dram','noc_tx','noc_rx')


def hardware(chip):
    chip=Path(chip).resolve()
    arch=yaml.safe_load(chip.read_text())['architecture']
    paths=[chip]
    for name in ('dram','noc'):
        path=Path(arch[name]['config_path'])
        paths.append(path if path.is_absolute() else ROOT/path)
    org=yaml.safe_load(paths[1].read_text())['MemorySystem']['DRAM']['org']
    density=re.match(r'HBDRAM_(\d+)(Gb|Mb)_',org['preset'])
    if density is None:raise ValueError('An explicit HBDRAM density preset is required')
    capacity=int(density[1])*(1024**3 if density[2]=='Gb' else 1024**2)//8*org['channel']
    return arch,capacity,paths


def fingerprint(chip):
    _,_,paths=hardware(chip)
    paths += sorted((ROOT/'frontend/near_mem').glob('*.py'))
    paths += [ROOT/'frontend/auto'/name for name in ('tiling_explorer_cloud.py', 'generalize.py', 'gemm_schedule.py', 'collectives.py')]
    # Content keyed: unrelated output directory paths in chip configs may reduce
    # reuse but can never produce an unsafe cross-hardware cache hit.
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes['atlas']=revision(ROOT)
    for folder in ('dram/ramulator2','noc/booksim2'):
        hashes[folder]=revision(ROOT/'simulator/src'/folder)
    for library in sorted((ROOT/'simulator/build/lib').glob('*.so')):
        digest=hashlib.sha256(library.read_bytes()).hexdigest()
        hashes['binary:'+library.name]=digest
        installed=ROOT/'simulator/atlasim'/library.name
        if installed.exists() and hashlib.sha256(installed.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Package/build backend libraries differ: {library.name}; synchronize the build first')
    return hashes


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
        subprocess.run([sys.executable,'-m','frontend.near_mem.run','--worker','--chip',str(chip),'--output',str(out)],
                       cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    result=json.loads((out/'result.json').read_text())
    assert result['flop_count']==2*manifest['macs']+manifest['vector_ops']
    return dict(manifest=manifest,result=result)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--plan',required=True,type=Path,help='JSON with problem and plan objects')
    ap.add_argument('--chip',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();cfg=json.loads(args.plan.read_text())
    p={k:tuple(v) if isinstance(v,list) else v for k,v in cfg['plan'].items()}
    print(json.dumps(simulate(Problem(**cfg['problem']),Plan(**p),args.chip,args.output),indent=2))
