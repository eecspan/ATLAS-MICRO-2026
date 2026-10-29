"""Cost-free description of the original multi-axis mapping space.

Keep algorithm choices here, independently of enumeration/ranking and native
execution. The first backend still accepts the older restricted Mapping;
accepting a Plan here does not imply it can already be lowered by that backend.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from math import gcd, isqrt, lcm, prod


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
    # Rotation endpoint: 'sram_consume' keeps shifted panels in SRAM (ours);
    # 'dram_land' lands every shifted A/B block in the receiver's DRAM.
    receive: str = 'sram_consume'
    # DRAM home of A/B/C: 'local' follows the plan (ours); 'global' is one
    # fixed 2-D block layout over the mesh, independent of the plan (T10).
    layout: str = 'local'
    # Number cores across the square mesh column-major instead of row-major
    # (a TP partition's mesh direction, see ``Ownership``).
    transpose: bool = False

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
    def land(self):
        return self.receive=='dram_land'

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
        if self.layout not in ('local','global'):
            raise ValueError('Unknown DRAM layout')
        if self.receive=='dram_land':
            # Whole blocks shift between whole-block computations: one epoch,
            # i.e. the compiler saturates the group (canonically (1,1,1)).
            if self.layout!='local' or self.panel is not None or self.group!=(1,1,1):
                raise ValueError('DRAM landing rotates whole blocks of a local layout')
        if self.transpose and isqrt(self.cores)**2!=self.cores:
            raise ValueError('A transposed core numbering needs a square mesh')
        if self.panel is not None:
            if len(self.panel)!=3:
                raise ValueError('Three panel capacities required')
            for (v,parts),n,payload in zip(self.temporal.items(),self.panel,self.payload_tiles):
                if type(n) is not int or n<1 or n>(payload if prod(parts)>1 else 1):
                    raise ValueError(f'Invalid {v} panel capacity')

    def sram_bytes(self, dtype=2, overlap=True):
        """Live SRAM: resident panels, double buffers and transit buffers.

        C's partial-result scratch is reused as its receive bank. A/B two-bank
        allocation admits prefetch or computation/shift overlap, never both
        future payloads in one bank.
        """
        tile={v:prod(self.tile[d] for d in axes)*dtype for v,axes in DEPENDENCIES.items()}
        total=0
        for (v,parts),payload in zip(self.temporal.items(),self.payload_tiles):
            shifted=prod(parts)>1;landed=shifted and self.land
            resident=shifted and not landed
            total+=(payload if resident else 1)*tile[v]*(2 if overlap or resident or v=='C' else 1)
            total+=3*tile[v]*landed  # a forwarded tile being read, sent and landed
        if overlap and prod(self.c)==1:
            total+=2*tile['C']  # Native double-buffered partial and running outputs.
        if self.layout=='global':
            total+=2*(tile['A']+tile['B'])  # tiles a home core serves to remote readers
        if self.ps[2]//prod(self.c)>1:
            # The final line reduce-scatter runs after compute; per direction a
            # rank holds the tile it sends, the one it accumulates and the next.
            total=max(total,6*tile['C'])
        return total

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


@dataclass(frozen=True)
class FlashPlan:
    """FlashGEMM on a square mesh (see ``gemm.flash``): per (M panel, K tile)
    step operands are all-gathered along mesh lines while the previous step
    computes. ``b_resident`` (weight-stationary only) keeps the local B in
    SRAM across M panels. ``n_tile=None`` holds the whole local N, as in the
    paper; a smaller N panel is only our enabling extension when no such
    plan fits the SRAM."""
    dataflow: str
    m_tile: int
    k_tile: int
    b_resident: bool = False
    cores: int = 16
    kind: str = 'flash'
    n_tile: int | None = None


def load(data):
    """A plan from its ``asdict`` form."""
    data={k:tuple(v) if isinstance(v,list) else v for k,v in data.items()}
    return FlashPlan(**data) if data.get('kind')=='flash' else Plan(**data)
