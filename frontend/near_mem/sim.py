"""The ATLAS boundary: hardware parsing, provenance, and isolated simulator runs."""
import argparse
import ctypes
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
KEYS = ('matrix', 'vector', 'buffer_load', 'buffer_store', 'dram', 'noc_tx', 'noc_rx')


def strides(shape, dtype=2, column=False):
    """Row- (or column-) major strides of a 2-D tensor, shared by every method.

    A leading dimension that is a multiple of 4 KiB is padded by one 128-byte
    burst. ATLAS interleaves channels at 4 KiB, so such pitches would make
    every row (column) start in the same subset of channels.
    """
    rows, cols = shape; lead = rows if column else cols
    if lead*dtype % 4096 == 0:
        lead += 128//dtype
    return [1, lead] if column else [lead, 1]


def span(shape, strides, dtype=2):
    """Bytes a strided tensor occupies."""
    return (sum((n-1)*s for n, s in zip(shape, strides))+1)*dtype


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


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
    paths += sorted((ROOT/'frontend/near_mem').rglob('*.py'))
    paths += [ROOT/'frontend/auto'/name for name in ('tiling_explorer_cloud.py', 'generalize.py', 'gemm_schedule.py', 'collectives.py')]
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


def make_hardware(directory, cores, sram_kib=128, macs=1024, channels=1, flit_size=64):
    import math
    import re
    import yaml
    if math.isqrt(cores)**2 != cores:
        raise ValueError("This hardware helper supports square 2D meshes only")
    directory.mkdir(parents=True, exist_ok=True)
    chip = yaml.safe_load((ROOT / "configs/architecture/chip/test_chip_16ch.yaml").read_text())
    arch = chip["architecture"]
    arch["core_num"] = cores
    arch["core"]["matrix"]["mac_num"] = macs
    arch["core"]["buffer"].update(buffer_size=sram_kib, read_bw=1024, write_bw=1024)
    dram = yaml.safe_load((ROOT / "configs/architecture/dram/hb_dram_16ch.yaml").read_text())
    dram["MemorySystem"]["DRAM"]["org"]["channel"] = channels
    (directory / "dram.yaml").write_text(yaml.safe_dump(dram))
    noc = (ROOT / "configs/architecture/noc/4x4_mesh").read_text()
    noc = re.sub(r"(?m)^k\s*=\s*4;", f"k = {math.isqrt(cores)};", noc)
    (directory / "mesh").write_text(noc)
    arch["dram"]["config_path"] = str((directory / "dram.yaml").resolve())
    arch["noc"].update(config_path=str((directory / "mesh").resolve()), flit_size=flit_size)
    (directory / "chip.yaml").write_text(yaml.safe_dump(chip))
    return directory / "chip.yaml"


FIELDS = ('e2e_cycles', 'matrix_cycles', 'vector_cycles', 'buffer_cycles',
          'dram_cycles', 'noc_cycles', 'flop_count', 'memory_access_bytes')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def execute(chip, out, operators, placement, timeout=600):
    out.mkdir(parents=True, exist_ok=True)
    dump(out/'operators.yaml', operators)
    dump(out/'placement.yaml', placement)
    with (out/'simulator.log').open('w') as log:
        subprocess.run([sys.executable, '-m', 'frontend.near_mem.sim',
                        '--worker', '--chip', str(chip), '--output', str(out)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    return json.loads((out/'result.json').read_text())


def worker(chip, out):
    sys.path.insert(0, str(ROOT/'simulator/build/lib'))
    from atlasim import Chip
    # MCRouter uses libc rand(), separately from BookSim's seeded RNG. Native
    # compiler tuning can consume this process-global stream before final run.
    ctypes.CDLL(None).srand(1)
    instance = Chip(str(chip), str(out/'operators.yaml'), str(out/'placement.yaml'))
    perf = instance.simulate()
    result = {k: getattr(perf.e2e_stats, k) for k in FIELDS}
    result['operators'] = {name: {k: getattr(stats, k) for k in FIELDS}
                           for name, stats in perf.operator_stats}
    # ATLAS times an operator per core from that core's own start, so its
    # per-operator e2e includes waiting for slower cores of the previous one.
    # ``span`` is the operator's share of the critical path: latest finish
    # minus latest start. Spans sum exactly to the whole-graph cycles.
    finish = None
    for name, stats in perf.operator_stats_per_core:
        start = finish or [0]*len(stats)
        finish = [s+x.e2e_cycles for s, x in zip(start, stats)]
        result['operators'][name]['span'] = max(finish)-max(start)
    dump(out/'result.json', result)



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--chip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();worker(a.chip,a.output)


if __name__=='__main__':main()
