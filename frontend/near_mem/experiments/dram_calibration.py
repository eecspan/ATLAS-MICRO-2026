"""Calibrate the proxy's DRAM channel model (``Proxy.CALIBRATION``) against ATLAS.

Simulates single DRAM batches on core 0 of each given chip -- strided runs of
several lengths and pitches, and mixed multi-tensor read/write batches --
and fits ``cycles = bursts*burst/channel_bw + ROW*rows - OVERLAP`` on the
busiest channel, the burst cost being fixed by the configuration. Run once
per DRAM timing preset; chips that differ only in channel count share it.
The proxy only ranks candidates; reported cycles always come from ATLAS.
"""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from ..gemm.space import Proxy
from ..tasks import Graph


def cases(seed=0):
    rng = random.Random(seed)
    for run in (64, 128, 256, 512, 896, 2048, 8192):
        for pitch in (run, 4096, 8192, 16384, 28672, 65536):
            if pitch >= run:
                for lines in (8, 64, 448):
                    yield [(run, pitch, lines, False)]
    for _ in range(40):
        yield [(rng.choice((64, 128, 256, 512, 1024, 4096)), rng.choice((4096, 8192, 28800, 28672, 57344)),
                rng.choice((16, 64, 256)), rng.random() < 0.3) for _ in range(rng.randint(2, 4))]


def features(proxy, batch, bases, row=37.0):
    """Bursts and opened rows of the busiest channel (the proxy's own address map)."""
    bits = proxy.channels.bit_length()-1; low = proxy.interleave; bursts = {}; rows = {}
    for (run, pitch, lines, _), base in zip(batch, bases):
        for line in range(lines):
            start = base+line*pitch
            for t in range(start//proxy.burst, (start+run+proxy.burst-1)//proxy.burst):
                c = (t >> low) % proxy.channels
                bursts.setdefault(c, set()).add(t)
                rows.setdefault(c, set()).add((((t >> (low+bits)) << low) | (t & ((1 << low)-1)))//proxy.columns)
    c = max(bursts, key=lambda c: len(bursts[c])*proxy.burst/proxy.channel_bw+row*len(rows[c]))
    return len(bursts[c])*proxy.burst/proxy.channel_bw, len(rows[c])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('chips', type=Path, nargs='+'); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); rows = []
    for chip in a.chips:
        proxy = Proxy(chip)
        for i, batch in enumerate(cases()):
            g = Graph(chip); tl = g.timeline(); bases = []
            for j, (run, pitch, lines, write) in enumerate(batch):
                tensor = g.tensor(0, f'x{j}', (lines+1, pitch//2), strides=[pitch//2, 1])
                bases.append(tensor['base_addr'])
                for core in range(1, g.cores):
                    g.tensor(core, f'x{j}', (8, 8))
                tl.access(0, 2 if write else 0, f'x{j}', [0, 0], [lines, run//2], write=write)
            g.add_timeline('batch', tl)
            cycles = g.simulate(a.output/chip.parent.name/str(i))['result']['operators']['batch']['dram_cycles']
            rows.append((*features(proxy, batch, bases), cycles))
    burst, opened, y = (np.array(x, float) for x in zip(*rows))
    (row, overlap), *_ = np.linalg.lstsq(np.stack([opened, -np.ones_like(opened)], 1)/y[:, None], (y-burst)/y,
                                         rcond=None)
    error = abs((burst+row*opened-overlap)/y-1)
    summary = dict(ROW=round(row, 1), OVERLAP=round(overlap, 1), cases=len(y),
                   relative_error=dict(median=float(np.median(error)), p90=float(np.percentile(error, 90))))
    (a.output/'calibration.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
