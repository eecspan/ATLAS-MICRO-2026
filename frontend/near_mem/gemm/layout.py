"""A fixed global DRAM layout, independent of any mapping plan.

T10 treats DRAM as one unified backing store: every tensor is cut into
``side x side`` blocks over the square mesh, block ``(r, c)`` on core
``r*side+c``, whatever plan later reads it. A tile then lives on one or more
home cores and a non-home reader pays a remote access (home DRAM + NoC).
"""
from math import isqrt

from ..tasks import Region


class GlobalLayout:
    def __init__(self, problem, cores):
        self.side = isqrt(cores)
        if self.side*self.side != cores:
            raise ValueError('The global layout needs a square mesh')
        self.shapes = dict(A=(problem.m, problem.k), B=(problem.k, problem.n), C=(problem.m, problem.n))
        if any(n % self.side for shape in self.shapes.values() for n in shape):
            raise ValueError('The global layout needs mesh-divisible dimensions')
        if problem.n % (2*self.side) == 0:
            self.shapes['G'] = (problem.m, problem.n//2)  # fused gated-FFN output

    def block(self, tensor):
        rows, cols = self.shapes[tensor]
        return rows//self.side, cols//self.side

    def pieces(self, tensor, row, col, rows, cols):
        """``(home, local_base, extent)`` of each home block a window touches."""
        h, w = self.block(tensor)
        for r in range(row//h, (row+rows-1)//h+1):
            for c in range(col//w, (col+cols-1)//w+1):
                r0, r1 = max(row, r*h), min(row+rows, (r+1)*h)
                c0, c1 = max(col, c*w), min(col+cols, (c+1)*w)
                yield r*self.side+c, [r0-r*h, c0-c*w], [r1-r0, c1-c0]

    def regions(self, tensor):
        h, w = self.block(tensor)
        return [Region(core, tensor, core//self.side*h, core % self.side*w, h, w) for core in range(self.side**2)]
