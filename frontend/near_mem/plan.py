"""Cost-free description of the original multi-axis mapping space.

Keep algorithm choices here, independently of enumeration/ranking and native
execution. The first backend still accepts the older restricted Mapping;
accepting a Plan here does not imply it can already be lowered by that backend.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import gcd, lcm, prod


@dataclass(frozen=True)
class Layer:
    axis: int  # m,n,k = 0,1,2
    radix: int
    tensors: tuple[str, ...]


DEPENDENCIES = {'A': (0, 2), 'B': (2, 1), 'C': (0, 1)}
REPLICA_AXIS = {'A': 1, 'B': 0, 'C': 2}


@dataclass(frozen=True)
class Problem:
    m: int
    n: int
    k: int
    dtype_bytes: int = 2

    @property
    def shape(self):
        return self.m, self.n, self.k


@dataclass(frozen=True)
class Plan:
    ps: tuple[int, int, int]
    tile: tuple[int, int, int]
    # Per-tensor periods are in its dependency-axis order: A(m,k),B(k,n),C(m,n).
    a: tuple[int, int] = (1, 1)
    b: tuple[int, int] = (1, 1)
    c: tuple[int, int] = (1, 1)
    loop: str = 'mnk'
    group: tuple[int, int, int] = (1, 1, 1)
    # Empty order means canonical m,n,k gcd layers; search chooses permutations.
    layer_order: tuple[int, ...] = ()
    # None selects full shifted panels. Explicit counts preserve old endpoints.
    panel: tuple[int, int, int] | None = None
    receive: str = 'sram_consume'

    @property
    def temporal(self):
        return {'A': self.a, 'B': self.b, 'C': self.c}

    @property
    def cores(self):
        return prod(self.ps)

    @property
    def periods(self):
        return lcm(self.a[0],self.c[0]), lcm(self.b[1],self.c[1]), lcm(self.a[1],self.b[0])

    @property
    def layers(self):
        return self._layers(self.a,self.b,self.c)

    @staticmethod
    @lru_cache(None)
    def _layers(a,b,c):
        temporal={'A':a,'B':b,'C':c}
        layers=[]
        for axis in range(3):
            active={v:temporal[v][axes.index(axis)] for v,axes in DEPENDENCIES.items() if axis in axes}
            while any(n>1 for n in active.values()):
                names=tuple(v for v,n in active.items() if n>1)
                radix=gcd(*(active[v] for v in names))
                if radix<=1:
                    raise ValueError('Temporal periods require a valid gcd-layer decomposition')
                layers.append(Layer(axis,radix,names))
                active={v:n//radix if n>1 else n for v,n in active.items()}
        return tuple(layers)

    @property
    def payload_tiles(self):
        return tuple(prod(self.periods[d]//parts[i]*self.group[d] for i,d in enumerate(DEPENDENCIES[v]))
                     for v,parts in self.temporal.items())

    def validate(self, problem):
        if any(len(x)!=3 for x in (self.ps,self.tile,self.group)) or any(len(x)!=2 for x in (self.a,self.b,self.c)):
            raise ValueError('Spatial/tile/group triples and tensor period pairs required')
        numbers=(*problem.shape,problem.dtype_bytes,*self.ps,*self.tile,*self.a,*self.b,*self.c,*self.group)
        if any(type(n) is not int or n<=0 for n in numbers):
            raise ValueError('All dimensions and factors must be positive integers')
        if sorted(self.loop)!=list('kmn'):
            raise ValueError('Loop must permute m,n,k')
        for v,parts in self.temporal.items():
            if self.ps[REPLICA_AXIS[v]] % prod(parts):
                raise ValueError(f'{v} temporal partition must divide its replica group')
        base=[]
        for d,(n,sp,tp,t,g) in enumerate(zip(problem.shape,self.ps,self.periods,self.tile,self.group)):
            if n % (sp*tp*t*g):
                raise ValueError('Exact spatial/temporal/tile/group divisibility required')
            base.append(n//(sp*tp*t))
        # Match the old saturate-inner grouping domain, not all rectangular groups.
        outer_nonunit=False
        for axis in self.loop:
            d='mnk'.index(axis)
            if outer_nonunit and self.group[d]!=base[d]:
                raise ValueError('Grouping must saturate inner loops first')
            outer_nonunit |= self.group[d]>1
        layers=self.layers
        if self.layer_order and sorted(self.layer_order)!=list(range(len(layers))):
            raise ValueError('Layer order must be a permutation of all gcd layers')
        if self.receive not in ('sram_consume','dram_land'):
            raise ValueError('Unknown receive endpoint')
        if self.panel is not None:
            if len(self.panel)!=3:
                raise ValueError('Three panel capacities required')
            for (v,parts),n,payload in zip(self.temporal.items(),self.panel,self.payload_tiles):
                if type(n) is not int or n<1 or n>(payload if prod(parts)>1 else 1):
                    raise ValueError(f'Invalid {v} panel capacity')

    def orders(self):
        """Lazy exact layer orders; a search policy may rank, but must not rename, them."""
        return permutations(range(len(self.layers)))

    @classmethod
    def from_legacy(cls, old):
        if old.dram_residency!='co_designed' or old.dram_layout!='co_located':
            raise ValueError('Decoupled baseline residency is a separate algorithm family')
        if old.layer_order!='elk':
            raise ValueError('Resolve legacy min layer-order policy to a concrete order before conversion')
        plan=cls((old.ps_m,old.ps_n,old.ps_k),(old.tile_m,old.tile_n,old.tile_k),
                 (old.pt_m_a,old.pt_k_a),(old.pt_k_b,old.pt_n_b),(old.pt_m_c,old.pt_n_c),
                 old.loop_order,(old.g_m,old.g_n,old.g_k),panel=(old.panel_a,old.panel_b,old.panel_c),
                 receive=old.recv_mode)
        # ELK order depends on the actual tensor payload sizes, not radix alone.
        sizes={v:count*prod(plan.tile[d] for d in DEPENDENCIES[v])
               for v,count in zip(DEPENDENCIES,plan.payload_tiles)}
        from dataclasses import replace
        return replace(plan,layer_order=tuple(sorted(range(len(plan.layers)),
                       key=lambda i:-sum(sizes[v] for v in plan.layers[i].tensors))))
