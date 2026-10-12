"""Shared isolated ATLAS execution and hardware configuration helpers."""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


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
        subprocess.run([sys.executable, '-m', 'frontend.near_mem.run',
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
    dump(out/'result.json', result)



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--chip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();worker(a.chip,a.output)


if __name__=='__main__':main()
