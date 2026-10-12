"""Run the complete upstream automatic cloud compiler, without kernel substitution.

This is the native decode path, not a prefill emulation. Only worker count is
bound here; shapes, tile enumeration, placement, collectives and KV generation
come from upstream. A fresh output namespace prevents its shape-only cache from
mixing hardware or compiler versions.
"""
import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import time

import yaml

from .run import ROOT, revision
from .backend import fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chip', type=Path, default=ROOT/'configs/architecture/chip/cloud/stratum/atlas.yaml')
    parser.add_argument('--model', type=Path, default=ROOT/'configs/models/llama3_8b.json')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch,args.context,args.block_size,args.tp,args.workers) <= 0:
        parser.error('All counts must be positive')
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Use an empty output directory: upstream tiling cache is not hardware-keyed')
    out.mkdir(parents=True,exist_ok=True)
    from frontend.hardware_parser import CloudSystemConfig
    from frontend.model_parser import get_model_config_from_hf
    from frontend.auto import system
    from frontend.auto.tiling_explorer_cloud import explore_cloud_tiling
    from frontend.util import generate_context_slot_mapping

    random.seed(42)
    cfg=yaml.safe_load((ROOT/'configs/architecture/system/test_cloud_system.yaml').read_text())
    cfg['system']['chip']['config_path']=str(args.chip.resolve())
    cfg['system']['parallelism'].update(tp_size=args.tp,ep_size=1)
    cfg['system']['kv_cache'].update(max_context_length=args.batch*(args.context+args.block_size),block_size=args.block_size)
    (out/'system.yaml').write_text(yaml.safe_dump(cfg))
    hardware=CloudSystemConfig.from_yaml(cfg)
    model=get_model_config_from_hf(args.model.stem,str(args.model.resolve()))
    contexts=[args.context]*args.batch
    slots=generate_context_slot_mapping(hardware,contexts)
    system.explore_cloud_tiling=partial(explore_cloud_tiling,num_workers=args.workers)
    files=[args.chip.resolve(),args.model.resolve()]
    arch=yaml.safe_load(args.chip.read_text())['architecture']
    files += [Path(arch[c]['config_path']).resolve() for c in ('dram','noc')]
    files += [ROOT/p for p in ('frontend/auto/system.py','frontend/auto/tiling_explorer_cloud.py',
                               'frontend/auto/gemm_schedule.py','frontend/auto/generalize.py','frontend/auto/collectives.py',
                               'frontend/model_parser.py','frontend/hardware_parser.py','frontend/util.py',
                               'frontend/near_mem/cloud.py')]
    manifest=dict(atlas_revision=revision(ROOT), model=asdict(model),batch=args.batch,context=args.context,
                  block_size=args.block_size,tp=args.tp,workers=args.workers,
                  entry='frontend.auto.system.CloudSystem.inference',
                  scope='complete upstream cloud decode compiler path; not complete physical-model validation or prefill',
                  changes='worker count bounded; native tile ranges retained; shared GEMM/DRAM, gated FFN and decode address/count fixes',
                  backend=fingerprint(args.chip),
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.monotonic()
    result=system.CloudSystem(hardware,model,2).inference(contexts,slots,str(out/'native'))
    stats=result['intra_chip_computation_performance'].e2e_stats
    summary={k:v for k,v in result.items() if k!='intra_chip_computation_performance'}
    summary.update(layer_cycles=stats.e2e_cycles,wall_seconds=time.monotonic()-start,
                   flop_count=stats.flop_count,memory_access_bytes=stats.memory_access_bytes)
    (out/'result.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
