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
        # Decode GEMMs with M=1 have no row parallelism. Keep one DRAM copy,
        # stripe A along K and B/C along N across all banks. Readers then pay
        # the real remote-access cost for A; this does not silently replicate it.
        if problem.m == 1:
            self.grids = {name: (1, cores) for name in self.shapes}
        else:
            self.grids = {name: (self.side, self.side) for name in self.shapes}
        if any(n % part for name, shape in self.shapes.items()
               for n, part in zip(shape, self.grids[name])):
            raise ValueError('The global layout needs dimensions divisible by its per-tensor grid')
        if problem.n % (2*self.grids['C'][1]) == 0:
            self.shapes['G'] = (problem.m, problem.n//2)  # fused gated-FFN output
            self.grids['G'] = self.grids['C']

    def block(self, tensor):
        rows, cols = self.shapes[tensor]
        gr, gc = self.grids[tensor]
        return rows//gr, cols//gc

    def pieces(self, tensor, row, col, rows, cols):
        """``(home, local_base, extent)`` of each home block a window touches."""
        h, w = self.block(tensor)
        gr, gc = self.grids[tensor]
        for r in range(row//h, (row+rows-1)//h+1):
            for c in range(col//w, (col+cols-1)//w+1):
                r0, r1 = max(row, r*h), min(row+rows, (r+1)*h)
                c0, c1 = max(col, c*w), min(col+cols, (c+1)*w)
                yield r*gc+c, [r0-r*h, c0-c*w], [r1-r0, c1-c0]

    def regions(self, tensor):
        h, w = self.block(tensor)
        gr, gc = self.grids[tensor]
        return [Region(core, tensor, core//gc*h, core % gc*w, h, w) for core in range(gr*gc)]
