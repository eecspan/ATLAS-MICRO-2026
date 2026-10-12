"""Fixed-workload Transformer evaluation: measurement cache and search.

This module contains only experiment orchestration — measuring
``llm.build`` graphs, proposing GEMM candidates, and running the
coordinate-descent comparison against the ATLAS native baseline. Core model
and graph semantics live in ``llm.py``.
"""
import argparse
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from itertools import permutations
from pathlib import Path
import time

from .attention import KVLayout, plans as attention_plans
from .backend import fingerprint, hardware
from .execution import ExecutionCache, configuration_key
from .llm import (Workload, build, configuration, decode_configuration,
                  tune_native)
from .run import dump
from .search import Proxy, candidates


class Experiment:
    def __init__(self,workload,chip,out):
        self.workload = workload; self.chip = Path(chip).resolve(); self.out = Path(out).resolve()
        self.out.mkdir(parents=True,exist_ok=True)
        self.provenance = fingerprint(chip); self.rows = []
        self.cache = ExecutionCache(chip,self.out/'runs',self.provenance)

    def measure(self,config,label,record=True):
        signature = dict(workload=asdict(self.workload),configuration=config,sha256=self.provenance)
        key = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:20]
        folder = self.out/'runs'/key
        sigfile = folder/'signature.json'
        if (sigfile.exists() and (folder/'result.json').exists() and (folder/'graph.json').exists()
                and json.loads(sigfile.read_text()) == json.loads(json.dumps(signature))):
            result = json.loads((folder/'result.json').read_text())
            manifest = json.loads((folder/'graph.json').read_text())
        else:
            graph = build(self.workload,self.chip,config)
            run = graph.simulate(folder,self.cache,configuration_key(signature)); result,manifest = run['result'],run['graph']
            dump(sigfile,signature)
        row = dict(label=label,key=key,configuration=config,result=result,
                   peak_dram_bytes=max(manifest['peak_dram_bytes']),path=str(folder),
                   layout_noc_bytes=sum(e['noc_data_bytes'] for e in manifest['edges']),
                   layout_write_bytes=sum(e['dram_write_bytes'] for e in manifest['edges']))
        if record:
            self.record(row)
        return row

    def record(self,row):
        self.rows.append(row); dump(self.out/'measurements.json',self.rows)
        print(f'{row["label"]}: {row["result"]["e2e_cycles"]} cycles, peak {row["peak_dram_bytes"]} B/core',flush=True)

    def measure_many(self,configs,label,workers):
        # Only independent neighbors of the same incumbent run together. Each
        # worker uses a separate native simulator process and output directory.
        unique = {json.dumps(cfg,sort_keys=True):cfg for cfg in configs}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda cfg:self.measure(cfg,label,record=False),unique.values()))
        for row in rows:
            self.record(row)
        return rows


def propose(workload,chip,budget,tile_min):
    arch,dram,_ = hardware(chip); proxy = Proxy(chip)
    selected = {}; report = {}
    for name,problem in workload.problems.items():
        domains = {}; counts = {}
        for rotation,label in ((False,'local'),(True,'joint')):
            families = {}; count = 0
            domain = candidates(problem,arch['core_num'],arch['core']['buffer']['buffer_size']*1024,
                                tile_min,'powers','elk',dram,rotation)
            for plan in domain:
                count += 1
                if rotation and not plan.layers:
                    continue
                # Spend reranking on distinct spatial/residency decisions;
                # all six loops/tiles still compete within each proxy family.
                family = (plan.ps,plan.a,plan.b,plan.c)
                score = proxy(problem,plan)
                if family not in families or score < families[family][0]:
                    families[family] = (score,plan)
            chosen = [p for score,p in sorted(families.values(),key=lambda x:x[0])[:budget]]
            domains[label] = chosen; counts[label] = count
        selected[name] = domains
        report[name] = dict(enumerated=counts,selected={k:[asdict(p) for p in v] for k,v in domains.items()})
    return selected,report


def run(workload,chip,out,budget=4,workers=4,tile_min=16,passes=2):
    exp = Experiment(workload,chip,out); started = time.monotonic()
    # Upstream's tile cache omits hardware/source hashes: isolate it by our full fingerprint.
    provenance_key = hashlib.sha256(json.dumps(exp.provenance,sort_keys=True).encode()).hexdigest()[:16]
    kernels = tune_native(workload,chip,exp.out/'native_tuning'/provenance_key,workers)
    arch,_,_ = hardware(chip); layout = KVLayout(workload.attention,arch['core_num'])
    native_configs = [configuration(kernels,plan) for plan in
                      attention_plans(layout,arch['core']['buffer']['buffer_size']*1024,False)]
    baseline = min(exp.measure_many(native_configs,'atlas_attention_tuning',workers),
                   key=lambda row:row['result']['e2e_cycles']) if native_configs else None
    if baseline is None:
        raise ValueError('No SRAM-feasible attention plan')
    proposals,report = propose(workload,chip,budget,tile_min)
    dump(exp.out/'search.json',dict(gemm=report,budget_per_family=budget,passes=passes,
         policy='whole-graph coordinate descent; bounded proxy proposals; exact native incumbent retained'))
    best = baseline; local = baseline
    for family in ('local','joint'):
        for sweep in range(passes):
            before = best['key']
            for name in workload.problems:
                incumbent = best
                domain = proposals[name]['local'] if family == 'local' else proposals[name]['joint']+proposals[name]['local']
                choices = [dict(plan=asdict(plan),native=False) for plan in domain]
                choices.append(baseline['configuration']['kernels'][name])
                configs = []
                for choice in choices:
                    cfg = deepcopy(incumbent['configuration'])
                    cfg['kernels'][name].update(choice); configs.append(cfg)
                rows = exp.measure_many(configs,f'{family}_{sweep}_{name}',workers)
                best = min([best,*rows],key=lambda row:row['result']['e2e_cycles'])
            if family == 'joint':
                configs = []
                for plan in attention_plans(layout,arch['core']['buffer']['buffer_size']*1024,True):
                    if plan.query_parts == 1:
                        continue
                    cfg = deepcopy(best['configuration']); cfg['attention'] = asdict(plan); configs.append(cfg)
                rows = exp.measure_many(configs,f'{family}_{sweep}_attention',workers)
                best = min([best,*rows],key=lambda row:row['result']['e2e_cycles'])
            if before == best['key']:
                break
        if family == 'local':
            local = best
    # A matched no-rotation control receives the winning spatial/tile choices.
    # Retune its six local loop orders as a whole graph, rather than attribute
    # an omitted local tile to temporal rotation. Native fused kernels remain.
    control_config = deepcopy(best['configuration']); rotated = []
    for name,value in control_config['kernels'].items():
        if value['native']:
            continue
        kernel,_ = decode_configuration(workload,dict(kernels={name:value},attention=control_config['attention']))
        plan = kernel[name].plan
        if plan.layers:
            rotated.append(name)
            value['plan'] = asdict(replace(plan,a=(1,1),b=(1,1),c=(1,1),group=(1,1,1),layer_order=(),panel=None))
    control_config['attention']['query_parts'] = 1
    matched = None
    try:
        matched = exp.measure(control_config,'matched_local')
        for name in rotated:
            current = matched
            configs = []
            for order in permutations('mnk'):
                cfg = deepcopy(current['configuration']); cfg['kernels'][name]['plan']['loop'] = ''.join(order)
                configs.append(cfg)
            rows = exp.measure_many(configs,'matched_local_'+name,workers)
            matched = min([matched,*rows],key=lambda row:row['result']['e2e_cycles'])
        if matched['result']['e2e_cycles'] < local['result']['e2e_cycles']:
            local = matched
        if local['result']['e2e_cycles'] < best['result']['e2e_cycles']:
            best = local
    except ValueError as error:
        # Capacity may increase when the rotating resident-output representation
        # is replaced by four native C banks; no impossible control is timed.
        if 'capacity' not in str(error) and 'SRAM' not in str(error):
            raise
        dump(exp.out/'matched_control_infeasible.json',dict(reason=str(error),configuration=control_config))
    summary = dict(workload=asdict(workload),baseline=baseline,local=local,joint=best,matched_local=matched,
                   speedup=baseline['result']['e2e_cycles']/best['result']['e2e_cycles'],
                   joint_over_local=local['result']['e2e_cycles']/best['result']['e2e_cycles'],
                   seconds=time.monotonic()-started,sha256=exp.provenance,
                   scope='fixed-workload dense Transformer stack; includes prefill/decode attention, norms, RoPE, gated FFN, residual and layout edges; excludes embedding/logit head and inter-device communication',
                   baseline_definition='ATLAS native fixed K/N GEMMs, exhaustive native tiles and original Q/KV/FFN collectives, with common static sharded KV and generalized fused attention',
                   input_output='same unique row shards; boundary redistributions charged; weights/history preplaced; all configured layers capacity-resident',
                   search='heuristic whole-graph coordinate descent; native and local incumbents preserved; no global-optimality claim')
    dump(exp.out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ('speedup','joint_over_local','seconds')},indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chip',required=True,type=Path); parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=('prefill','decode'),default='prefill')
    for name,value in [('batch',1),('context',128),('query-heads',8),('kv-heads',2),('head-dim',64),
                       ('ffn',1536),('layers',1),('budget',4),('workers',4),('tile-min',16),('passes',2)]:
        parser.add_argument('--'+name,type=int,default=value)
    a = parser.parse_args()
    if min(a.batch,a.context,a.query_heads,a.kv_heads,a.head_dim,a.ffn,a.layers,a.budget,a.workers,a.tile_min,a.passes) < 1:
        parser.error('Counts must be positive')
    workload = Workload(a.batch,a.context if a.mode=='prefill' else 1,a.context,a.query_heads,a.kv_heads,
                        a.head_dim,a.ffn,a.layers)
    run(workload,a.chip,a.output,a.budget,a.workers,a.tile_min,a.passes)


if __name__ == '__main__':
    main()
