import math
import heapq
import ctypes
import signal
from typing import TYPE_CHECKING, Any, List, Dict, Tuple
if TYPE_CHECKING:
    from frontend.hardware_parser import CloudSystemConfig


_KB = 1024
_MB = 1024 * _KB
_GB = 1024 * _MB

_MHz = 1e6
_GHz = 1e9


def make_tensor_placement(
    *,
    name: str,
    base_addr: int,
    element_size: int,
    shape: List[int],
    strides: List[int],
) -> Dict[str, Any]:
    if not shape or len(shape) != len(strides):
        raise ValueError("Tensor placement shape/strides must be non-empty and have the same rank")
    return {
        "name": name,
        "base_addr": base_addr,
        "shape": [int(v) for v in shape],
        "strides": [int(v) for v in strides],
        "element_size": element_size,
    }


def make_contiguous_strides(shape: List[int], *, last_dim_contiguous: bool = True) -> List[int]:
    if not shape:
        raise ValueError("Shape must be non-empty")

    strides = [1] * len(shape)
    if last_dim_contiguous:
        for dim in range(len(shape) - 2, -1, -1):
            strides[dim] = strides[dim + 1] * shape[dim + 1]
    else:
        for dim in range(1, len(shape)):
            strides[dim] = strides[dim - 1] * shape[dim - 1]
    return strides


def tensor_dim(placement: Dict[str, Any], dim: int, *, expected_rank: int | None = None) -> int:
    shape = [int(v) for v in placement["shape"]]
    if expected_rank is not None and len(shape) != expected_rank:
        raise ValueError(f"Expected rank-{expected_rank} tensor placement, got rank {len(shape)}")
    if dim < 0 or dim >= len(shape):
        raise ValueError(f"Tensor placement rank {len(shape)} has no dimension {dim}")
    return shape[dim]


def make_dram_task(
    *,
    name: str,
    is_write: bool,
    access_base: List[int],
    access_extent: List[int],
    access_stride_add: List[int],
    access_offset_add: List[int],
    init_iter: int,
    stride_iter: int,
    total_iter: int,
) -> Dict[str, Any]:
    rank = len(access_base)
    if rank == 0 or rank != len(access_extent) or rank != len(access_stride_add) or rank != len(access_offset_add):
        raise ValueError("DRAM task access fields must be non-empty and have matching rank")
    return {
        "name": name,
        "is_write": is_write,
        "access_base": [int(v) for v in access_base],
        "access_extent": [int(v) for v in access_extent],
        "access_stride_add": [int(v) for v in access_stride_add],
        "access_offset_add": [int(v) for v in access_offset_add],
        "init_iter": init_iter,
        "stride_iter": stride_iter,
        "total_iter": total_iter,
    }


def set_pdeathsig():
    """Linux: receive SIGTERM when parent process exits (via PR_SET_PDEATHSIG)."""
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


def get_factors(n, min_factor=1) -> List[int]:
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            if i >= min_factor:
                factors.append(i)
            if i != n // i and n // i >= min_factor:
                factors.append(n // i)
    return sorted(factors)


def partition_list(numbers, n) -> List[List[int]]:
    # Sort numbers in descending order
    sorted_numbers = sorted(numbers, reverse=True)
    heap = [(0, i, []) for i in range(n)]
    heapq.heapify(heap)

    # Greedy allocation
    for num in sorted_numbers:
        current_sum, idx, sublist = heapq.heappop(heap)
        sublist.append(num)
        if type(num) in [int, float]:
            current_sum += num
        elif type(num) in [list, tuple]:
            products = 1
            for x in num:
                assert type(x) in [int, float]
                products *= x
            current_sum += products
        else:
            raise ValueError(f"Invalid number type: {type(num)}")
        heapq.heappush(heap, (current_sum, idx, sublist))

    # Extract results
    result = [item[2] for item in sorted(heap, key=lambda x: x[1])]
    return result


def generate_1d_mesh_mst_trees(core_array_size) -> Tuple[int, List[Dict[int, List[int]]]]:
    assert core_array_size % 2 == 0, \
        f"Currently we assume core array size is even, but got {core_array_size} % 2 != 0"
    trees = [{}, {}]
    depth = core_array_size // 2
    for i in range(depth+1):
        if i == 0:
            trees[0][i] = [depth-1]
            trees[1][i] = [depth]
        else:
            trees[0][i] = [depth-1-i, depth-1+i]
            trees[1][i] = [depth-i, depth+i]
    return depth, trees


def generate_mesh_ring_topology(core_array_size: int, return_rank_to_core: bool = True) -> Dict[int, int]:
    if core_array_size % 2 != 0:
        raise ValueError(f"Array size must be even, got {core_array_size}")

    path: List[Tuple[int, int]] = []
    
    # 1. First row (left to right)
    for j in range(core_array_size):
        path.append((0, j))
        
    # 2. Last column (downward)
    for i in range(1, core_array_size):
        path.append((i, core_array_size - 1))
        
    # 3. Zig-zag through remaining columns (core_array_size-2 to 0)
    # Rows involved are 1 to core_array_size-1
    for j in range(core_array_size - 2, -1, -1):
        # Determine direction
        # If (core_array_size - 2 - j) is even: UP (N-1 -> 1)
        # If (core_array_size - 2 - j) is odd: DOWN (1 -> N-1)
        if (core_array_size - 2 - j) % 2 == 0:
             rows = range(core_array_size - 1, 0, -1)
        else:
             rows = range(1, core_array_size)
             
        for i in rows:
            path.append((i, j))
            
    rank_to_core: Dict[int, int] = {}
    core_to_rank: Dict[int, int] = {}
    for rank, (r, c) in enumerate(path):
        rank_to_core[rank] = r * core_array_size + c
        core_to_rank[r * core_array_size + c] = rank
        
    if return_rank_to_core:
        return rank_to_core
    else:
        return core_to_rank


def sort_mesh_nodes_by_l1_distance(n: int) -> List[Tuple[int, int]]:
    centers: List[Tuple[int, int]] = []
    if n % 2 != 0:
        centers.append((n // 2, n // 2))
    else:
        # For even n, e.g., 4, centers are (1,1), (1,2), (2,1), (2,2)
        mid1 = n // 2 - 1
        mid2 = n // 2
        centers.append((mid1, mid1))
        centers.append((mid1, mid2))
        centers.append((mid2, mid1))
        centers.append((mid2, mid2))

    nodes: List[Tuple[float, int, int]] = []
    for r in range(n):
        for c in range(n):
            # Calculate minimum Manhattan distance to any center node
            min_dist = float('inf')
            for cr, cc in centers:
                dist = abs(r - cr) + abs(c - cc)
                if dist < min_dist:
                    min_dist = dist
            nodes.append((min_dist, r, c))
    nodes.sort(key=lambda x: x[0])

    return [(r, c) for _, r, c in nodes]


def compute_num_cores_per_req(
    context_length_list: List[int],
    core_num: int,
    block_size: int,
) -> List[int]:
    batch_size = len(context_length_list)
    if batch_size == 0:
        return []

    max_cores_per_req = [min(core_num, max(1, c // block_size)) for c in context_length_list]

    # --- Homogeneous case (common path) ---
    if len(set(context_length_list)) == 1:
        c = context_length_list[0]
        max_n = max_cores_per_req[0]
        # Smallest n such that batch_size * n is divisible by core_num,
        # guaranteeing every core receives the same number of request chunks.
        step = core_num // math.gcd(batch_size, core_num)
        n_val = step if step <= max_n else max_n
        return [n_val] * batch_size

    # --- Heterogeneous case ---
    num_cores_per_req = []
    for i in range(batch_size):
        c = context_length_list[i]
        if batch_size >= core_num:
            num_cores_per_req.append(1)
        else:
            per_core = math.ceil(c / core_num)
            if per_core >= block_size:
                num_cores_per_req.append(core_num)
            else:
                num_cores_per_req.append(max(1, c // block_size))
    return num_cores_per_req


def generate_context_slot_mapping(
    cloud_config: 'CloudSystemConfig',
    context_length_list: List[int],
):
    core_num = cloud_config.chip_config.core_num
    batch_size = len(context_length_list)
    block_size = cloud_config.block_size

    num_cores_per_req = compute_num_cores_per_req(context_length_list, core_num, block_size)
    per_req_valid_slots_per_core = [math.ceil(context_length_list[i] / num_cores_per_req[i]) for i in range(batch_size)]

    # Assign requests to cores with load balancing.
    # Use LPT (Longest Processing Time first) ordering: assign the largest
    # per-core workloads first, each time to the least-loaded cores.
    core_slot_counts = [0] * core_num
    req_core_assignments: List[List[int]] = [[] for _ in range(batch_size)]
    req_core_slot_counts: List[Dict[int, int]] = [{} for _ in range(batch_size)]

    order = sorted(range(batch_size),
                   key=lambda i: per_req_valid_slots_per_core[i], reverse=True)
    for i in order:
        n = num_cores_per_req[i]
        sorted_cores = sorted(range(core_num), key=lambda core_idx: core_slot_counts[core_idx])
        assigned = sorted_cores[:n]
        req_core_assignments[i] = assigned
        quotient, remainder = divmod(context_length_list[i], n)
        for rank, core_idx in enumerate(assigned):
            # ceil(c/n) on every core silently creates extra valid KV tokens.
            count = quotient + int(rank < remainder)
            req_core_slot_counts[i][core_idx] = count
            core_slot_counts[core_idx] += count

    # Per-core slot assignment: each core's slots start from 0 and increment
    # in request-ID order.
    context_slot_mapping = [{} for _ in range(core_num)]
    for core_id in range(core_num):
        reqs_on_core = sorted(
            i for i in range(batch_size) if core_id in req_core_assignments[i]
        )
        cur_slot = 0
        for req_id in reqs_on_core:
            num_slots = req_core_slot_counts[req_id][core_id]
            context_slot_mapping[core_id][req_id] = list(range(cur_slot, cur_slot + num_slots))
            cur_slot += num_slots

    return context_slot_mapping


def generate_last_slot_mapping(
    batch_size: int,
    core_array_size: int,
    core_num: int,
) -> List[List[List[int]]]:
    assert core_array_size**2 == core_num, f"Core array size {core_array_size} mismatch core num {core_num}"
    core_list_by_l1 = sort_mesh_nodes_by_l1_distance(core_array_size)

    last_slot_mapping = [[[] for _ in range(core_array_size)] for _ in range(core_array_size)]
    core_idx = 0
    for i in range(batch_size):
        r, c = core_list_by_l1[core_idx]
        last_slot_mapping[r][c].append(i)
        core_idx += 1
        if core_idx >= core_num:
            core_idx = 0

    return last_slot_mapping


def prepare_kv_comm_workload_size(
    core_array_size: int,
    last_slot_mapping: List[List[List[int]]],
):
    x_reduce_first_half_req_num = 0
    for i in range((core_array_size//2)):
        for j in range(core_array_size):
            x_reduce_first_half_req_num += len(last_slot_mapping[i][j])

    x_reduce_second_half_req_num = 0
    for i in range((core_array_size//2), core_array_size):
        for j in range(core_array_size):
            x_reduce_second_half_req_num += len(last_slot_mapping[i][j])
    x_scatter_first_half_per_hop_req_num = []

    tmp_req_num = 0
    # 1st MST is traversed reversely, from the second PE (the first PE does not need to transfer)
    for i in range((core_array_size//2)-1):
        tmp_row_req_num = 0
        for j in range(core_array_size):
            tmp_row_req_num += len(last_slot_mapping[i][j])
        tmp_req_num += tmp_row_req_num
        x_scatter_first_half_per_hop_req_num.append(tmp_req_num)
    x_scatter_first_half_per_hop_req_num.reverse()
    x_scatter_second_half_per_hop_req_num = []

    tmp_req_num = 0
    # 2nd MST is traversed normally, from the second PE (the first PE does not need to transfer)
    for i in range(core_array_size-1, (core_array_size//2), -1):
        tmp_row_req_num = 0
        for j in range(core_array_size):
            tmp_row_req_num += len(last_slot_mapping[i][j])
        tmp_req_num += tmp_row_req_num
        x_scatter_second_half_per_hop_req_num.append(tmp_req_num)
    x_scatter_second_half_per_hop_req_num.reverse()
    
    y_gather_first_half_req_num = []
    y_gather_second_half_req_num = []
    for i in range(core_array_size):
        first_half_req_num = 0
        for j in range((core_array_size//2)):
            first_half_req_num += len(last_slot_mapping[i][j])
        y_gather_first_half_req_num.append(first_half_req_num)

        # Get 2nd MST's req num
        second_half_req_num = 0
        for j in range((core_array_size//2), core_array_size):
            second_half_req_num += len(last_slot_mapping[i][j])
        y_gather_second_half_req_num.append(second_half_req_num)

    y_scatter_first_half_per_hop_req_num = []
    y_scatter_second_half_per_hop_req_num = []
    for i in range(core_array_size):
        # Similar to X scatter, we scatter from center to edge, the request number gradually reduces
        first_half_per_hop_req_num = []
        tmp_req_num = 0
        # 1st MST is traversed reversely, from the second PE (the first PE does not need to transfer)
        for j in range((core_array_size//2)-1):
            tmp_req_num += len(last_slot_mapping[i][j])
            first_half_per_hop_req_num.append(tmp_req_num)
        first_half_per_hop_req_num.reverse()
        y_scatter_first_half_per_hop_req_num.append(first_half_per_hop_req_num)

        second_half_per_hop_req_num = []
        tmp_req_num = 0
        # 2nd MST is traversed normally, from the second PE (the first PE does not need to transfer)
        for j in range(core_array_size-1, (core_array_size//2), -1):
            tmp_req_num += len(last_slot_mapping[i][j])
            second_half_per_hop_req_num.append(tmp_req_num)
        second_half_per_hop_req_num.reverse()
        y_scatter_second_half_per_hop_req_num.append(second_half_per_hop_req_num)

    return (
        x_reduce_first_half_req_num,
        x_reduce_second_half_req_num,
        x_scatter_first_half_per_hop_req_num,
        x_scatter_second_half_per_hop_req_num,
        y_gather_first_half_req_num,
        y_gather_second_half_req_num,
        y_scatter_first_half_per_hop_req_num,
        y_scatter_second_half_per_hop_req_num,
    )
