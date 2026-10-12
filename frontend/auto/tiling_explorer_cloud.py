import yaml
import math
import os
import contextlib
import multiprocessing
from typing import List, Dict, Any, Tuple
from itertools import product
import copy

from atlasim import Chip, SimulatorOperatorType, ChipConfig

from frontend.model_parser import Operator, OperatorType, ParallelConfig, ModelConfig
from frontend.hardware_parser import CloudSystemConfig, extract_noc_topology
from frontend.util import (
    _KB,
    generate_1d_mesh_mst_trees,
    generate_mesh_ring_topology,
    get_factors,
    make_contiguous_strides,
    make_dram_task,
    make_tensor_placement,
    partition_list,
    set_pdeathsig,
    sort_mesh_nodes_by_l1_distance,
    tensor_dim,
)


def set_shape(
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    context_length_list: List[int],
    parallel_config: ParallelConfig,
    core_num: int,
    core_array_size: int,
    block_size: int = 1,
):
    batch_size = len(context_length_list)
    for operator in attention_block:
        if operator.op_type == OperatorType.GEMM:
            operator.M = batch_size
            operator.N = math.ceil(operator.N/core_array_size)
            operator.K = math.ceil(operator.K/core_array_size)
        elif operator.op_type == OperatorType.ATTENTION:
            operator.input_length = [1 for _ in range(len(context_length_list))]
            # Homogeneous case: find minimum n giving perfect load balance
            if len(set(context_length_list)) == 1 and batch_size > 0:
                c = context_length_list[0]
                max_n = min(core_num, max(1, c // block_size))
                step = core_num // math.gcd(batch_size, core_num)
                n_val = step if step <= max_n else max_n
                operator.context_length = [math.ceil(c / n_val)] * batch_size
            else:
                per_core_context = []
                for c in context_length_list:
                    if batch_size >= core_num:
                        per_core_context.append(c)
                    else:
                        per_core = math.ceil(c / core_num)
                        if per_core >= block_size:
                            per_core_context.append(per_core)
                        else:
                            n = max(1, c // block_size)
                            per_core_context.append(math.ceil(c / n))
                operator.context_length = per_core_context
        elif operator.op_type == OperatorType.ALLREDUCE:
            operator.M = batch_size

    if model_config.is_moe:
        # assume a uniform distribution of expert routing
        per_ffn_token_num = math.ceil(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts) / model_config.moe_num_experts)
        per_device_ffn_num = math.ceil(min(batch_size * (model_config.moe_top_k + model_config.moe_shared_experts), model_config.moe_num_experts) / parallel_config.ep_size)
    else:
        per_device_ffn_num = 1
        per_ffn_token_num = batch_size
    for operator in ffn_moe_block:
        if operator.op_type == OperatorType.GEMM:
            operator.B = per_device_ffn_num
            operator.M = per_ffn_token_num
            operator.N = math.ceil(operator.N/core_array_size)
            operator.K = math.ceil(operator.K/core_array_size)
        elif operator.op_type == OperatorType.ALLREDUCE:
            assert per_device_ffn_num == 1 and not model_config.is_moe
            operator.M = per_ffn_token_num
        elif operator.op_type == OperatorType.ALL2ALL:
            operator.M = per_ffn_token_num * per_device_ffn_num


def make_gemm_description(operator, element_size, tiling_factor_combination,
                          input_name, output_name, weight_name):
    """Build the native MNK kernel, shared by tuning and compiler extensions.

    Buffer accounting, prefetch and writeback iterations follow upstream.
    Address strides count actual accesses, not compute iterations; this also
    fixes upstream's C stride for split-K tiles and A stride when nK=1.
    """
    tM, tK, tN = tiling_factor_combination
    nM = math.ceil(operator.M / tM)
    nK = math.ceil(operator.K / tK)
    nN = math.ceil(operator.N / tN)
    from frontend.auto.gemm_schedule import gemm_work

    comp_description = {
        "operator": [{
            "name": operator.name,
            "type": str(SimulatorOperatorType.GEMM),
            "iteration": nM*nK*nN,
            "execution": {
                **gemm_work((tM, tN, tK), element_size),
                "dram": [
                    make_dram_task(
                        name=input_name,
                        is_write=False,
                        access_base=[0, 0],
                        access_extent=[tM, tK],
                        access_stride_add=[nK * nN if nK > 1 else 1, 1],
                        access_offset_add=[tM, tK],
                        init_iter=0,
                        # In output stationary, when nK=1, all nN tiles in the same nM rows
                        # can share the same input tile.
                        stride_iter=1 if nK > 1 else nN,
                        total_iter=nM * nK * nN if nK > 1 else nM,
                    ),
                    make_dram_task(
                        name=weight_name,
                        is_write=False,
                        access_base=[0, 0],
                        access_extent=[tK, tN],
                        access_stride_add=[1, nK],
                        access_offset_add=[tK, tN],
                        init_iter=0,
                        # In output stationary, when nK*nN=1, we only need to load the full weight for one time
                        stride_iter=1,
                        total_iter=nM * nK * nN if nK * nN > 1 else 1,
                    ),
                    make_dram_task(
                        name=output_name,
                        is_write=True,
                        access_base=[0, 0],
                        access_extent=[tM, tN],
                        access_stride_add=[nN, 1],
                        access_offset_add=[tM, tN],
                        init_iter=nK + 1,
                        stride_iter=nK,
                        total_iter=nM * nN,
                    ),
                ]
            }
        }]
    }
    return comp_description["operator"][0]


def explore_gemm_comp_worker(
    # Tensor description
    input_placement: Dict[str, Any],
    output_placement: Dict[str, Any],
    weight_placement_list: List[Dict[str, Any]],
    # Operator shape description
    operator: Operator,
    element_size: int,
    # Chip config
    chip_config_path: str,
    # Tiling factors to evaluate
    tiling_factor_combination_list: List[Tuple[int, int, int]],
    # Hyper parameters
    worker_id: int,
    cur_op_dir: str,
) -> Dict[str, Any]:
    cur_worker_dir = os.path.join(cur_op_dir, f"worker_{worker_id}")
    os.makedirs(cur_worker_dir, exist_ok=True)

    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
            *weight_placement_list,
        ]
    }
    data_placement_config_path = os.path.join(cur_worker_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)
    # Only use the first weight placement for tiling exploration
    input_name = input_placement["name"]
    output_name = output_placement["name"]
    weight_name = weight_placement_list[0]["name"]

    opt_comp_description = [None, (-1, -1, -1)]
    opt_latency = float('inf')
    comp_description_path = os.path.join(cur_worker_dir, "comp_description.yaml")
    for tiling_factor_combination in tiling_factor_combination_list:
        comp_description = {"operator": [make_gemm_description(
            operator, element_size, tiling_factor_combination,
            input_name, output_name, weight_name)]}
        with open(comp_description_path, "w") as f:
            yaml.dump(comp_description, f)
        
        # Redirect C++ output to a log file
        import atlasim
        log_file_path = os.path.join(cur_worker_dir, "worker_output.log")
        with open(log_file_path, "w") as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                with atlasim.ostream_redirect(stdout=True, stderr=True):
                    chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
                    performance = chip.simulate()
                    del chip

        latency = performance.e2e_stats.e2e_cycles
        if latency < opt_latency:
            opt_latency = latency
            opt_comp_description = (comp_description, tuple(tiling_factor_combination))

    all_bacth_comp_description = []
    if opt_comp_description[0] is not None:
        for b in range(operator.B):
            cur_weight_name = weight_placement_list[b]["name"]
            cur_comp_description = opt_comp_description[0]["operator"][0].copy()
            cur_comp_description["execution"]["dram"][1]["name"] = cur_weight_name
            all_bacth_comp_description.append(cur_comp_description)

    return {
        "latency": opt_latency,
        "comp_description_list": all_bacth_comp_description,
        "tiling_factors": opt_comp_description[1],
    }


def explore_gemm_comp(
    # Tensor description
    input_placement: Dict[str, Any],
    output_placement: Dict[str, Any],
    weight_placement_list: List[Dict[str, Any]],
    # Operator shape description
    operator: Operator,
    element_size: int,
    # Chip config
    chip_config: ChipConfig,
    chip_config_path: str,
    # Hyper parameters
    min_tM: int, min_tK: int, min_tN: int,
    num_workers: int,
    # Intermediate result directory
    intermediate_result_dir: str,
    # Shared GEMM tiling cache directory (reusable across context lengths)
    gemm_tiling_cache_dir: str = "",
) -> Dict[str, Any]:
    assert operator.op_type == OperatorType.GEMM
    cache_base_dir = gemm_tiling_cache_dir if gemm_tiling_cache_dir else intermediate_result_dir
    cur_op_dir = os.path.join(cache_base_dir, operator.name)
    os.makedirs(cur_op_dir, exist_ok=True)

    cache_file_path = os.path.join(cur_op_dir, "gemm_opt_general_layout.yaml")
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cached = yaml.safe_load(f)
        if (cached is not None
            and cached.get("M") == operator.M
            and cached.get("K") == operator.K
            and cached.get("N") == operator.N
            and cached.get("B") == operator.B
            and cached.get("element_size") == element_size):
            print(f"[Cache hit] {operator.name}: tiling factors {cached['tiling_factors']}")
            return {
                "latency": cached["latency"],
                "comp_description_list": cached["comp_description_list"],
                "tiling_factors": tuple(cached["tiling_factors"]),
            }

    # Generate all possible tiling factor combinations
    tM_list = get_factors(operator.M, min_tM)
    tK_list = get_factors(operator.K, min_tK)
    tN_list = get_factors(operator.N, min_tN)
    while len(tM_list) == 0:
        min_tM = min_tM // 2
        tM_list = get_factors(operator.M, min_tM)
    while len(tK_list) == 0:
        min_tK = min_tK // 2
        tK_list = get_factors(operator.K, min_tK)
    while len(tN_list) == 0:
        min_tN = min_tN // 2
        tN_list = get_factors(operator.N, min_tN)
    if len(tK_list) == 0:
        tK_list = get_factors(operator.K)
    if len(tN_list) == 0:
        tN_list = get_factors(operator.N)
    tiling_factor_combination_list = list(product(tM_list, tK_list, tN_list))

    # Get each core's SRAM buffer size
    buffer_size = chip_config.buffer_config.buffer_size * _KB
    # Prune OOM tiling factor combinations
    legal_tiling_factor_combination_list = []
    for tiling_factor_combination in tiling_factor_combination_list:
        tM, tK, tN = tiling_factor_combination
        # 2*tM*tN: one for gemm partial sum, one for final output
        if element_size * (tM*tK + tK*tN + 2*tM*tN) > buffer_size / 2: # Double buffer
            continue
        legal_tiling_factor_combination_list.append(tiling_factor_combination)

    # Generate tensor placement yaml file
    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
            *weight_placement_list,
        ]
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)

    # Partition tiling factor combinations into multiple parts for parallel exploration
    tiling_factor_partitions = partition_list(legal_tiling_factor_combination_list, num_workers)
    opt_latency = float('inf')
    opt_comp_description = None
    # Use 'spawn' context to avoid nested-fork deadlocks when this function is
    # called from inside a forked child process (e.g. multiprocessing.Process).
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(num_workers, initializer=set_pdeathsig) as pool:
        results_list: List[Dict[str, Any]] = pool.starmap(
            explore_gemm_comp_worker,
            [
                (
                    input_placement,
                    output_placement,
                    weight_placement_list,
                    operator,
                    element_size,
                    chip_config_path,
                    tiling_factor_partition,
                    worker_id,
                    cur_op_dir,
                )
                for worker_id, tiling_factor_partition in enumerate(tiling_factor_partitions)
            ]
        )
        
        for result in results_list:
            if result["latency"] < opt_latency:
                opt_latency = result["latency"]
                opt_comp_description = result

    if opt_comp_description is not None:
        cache_data = {
            "M": operator.M,
            "K": operator.K,
            "N": operator.N,
            "B": operator.B,
            "element_size": element_size,
            "latency": float(opt_comp_description["latency"]),
            "comp_description_list": opt_comp_description["comp_description_list"],
            "tiling_factors": list(opt_comp_description["tiling_factors"]),
        }
        with open(cache_file_path, "w") as f:
            yaml.dump(cache_data, f)

    return opt_comp_description


def generate_q_proj_comm(
    # Tensor description
    input_placement: Dict[str, Any], # Current operator's output tensor
    output_placement: Dict[str, Any], # Next operator's input tensor
    # Operator shape description
    name: str,
    row_num: int, input_column_offset: int, q_length: int,
    element_size: int, op_num_per_element: int,
    reduce_direction: int, # 0: X-axis, 1: Y-axis
    # Chip config
    chip_config_path: str,
    core_num: int, core_array_size: int,
    flit_size: int, core_flit_index_list: List[int],
    # Intermediate result directory
    intermediate_result_dir: str,
    *, build_only: bool = False,
):
    cur_op_dir = os.path.join(intermediate_result_dir, name+"_comm_q")
    if not build_only:
        os.makedirs(cur_op_dir, exist_ok=True)

    ############################################
    #         Part 1: X-axis All Reduce        #
    ############################################

    all_reduce_iteration = 2 * (core_array_size - 1)
    all_reduce_description = {
        "name": name+"_comm_q_all_reduce",
        "type": str(SimulatorOperatorType.Communication),
        "iteration": all_reduce_iteration,
        "core_num": core_num,
        "file_prefix": name+"_comm_q_all_reduce/", # Need to join root directory storing full computation description
    }
    per_core_all_reduce_description = [{
        "communication": {
            "on_chip": [{
                "tx": {
                    "buffer_load": [],
                    "noc": [],
                },
                "rx": {
                    "buffer_load": [],
                    "buffer_store": [],
                    "vector": [],
                    "noc": [],
                }
            } for _ in range(all_reduce_iteration)],
            "dram": [],
        }
    } for _ in range(core_num)]

    per_iter_chunk_length = math.ceil(q_length / core_array_size)
    per_iter_chunk_byte = row_num * per_iter_chunk_length * element_size
    per_iter_flit_num = math.ceil(per_iter_chunk_byte / flit_size)
    per_iter_vec_op_count = row_num * per_iter_chunk_length * op_num_per_element
    
    # Reduce scatter in all reduce
    init_iter = 1
    chunk_loaded = [[False for _ in range(core_array_size)] for _ in range(core_num)]
    # On-chip communication description
    # print("------q comm rs in ar-----")
    for t in range(core_array_size-1):
        # Parallel core columns
        for i in range(core_array_size):
            # Each column conducting reduce scatter
            for j in range(core_array_size):
                src_core_id = j*core_array_size + i if reduce_direction == 0 else i*core_array_size + j
                dst_info_list = []
                if j <= t:
                    dst_core_id = (j+1)*core_array_size + i if reduce_direction == 0 else i*core_array_size + j+1
                    chunk_id = abs(j-t)
                    dst_info_list.append((j+1, dst_core_id, chunk_id))
                if j >= core_array_size-1-t:
                    dst_core_id = (j-1)*core_array_size + i if reduce_direction == 0 else i*core_array_size + j-1
                    chunk_id = core_array_size-1 - abs(j+t-(core_array_size-1))
                    dst_info_list.append((j-1, dst_core_id, chunk_id))
                # print(f"{t} {i} {j} {src_core_id} {dst_info_list}")
                
                for dst_idx, dst_core_id, chunk_id in dst_info_list:
                    if dst_idx>=0 and dst_idx<=core_array_size-1:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_all_reduce_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_all_reduce_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_load"].append({
                            "name": f"reduce{src_core_id}_load",
                            "byte_count": per_iter_chunk_byte * 2,
                            "is_write": False,
                        })
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"reduce{src_core_id}_store",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["vector"].append({
                            "name": f"reduce{src_core_id}",
                            "vec_count": per_iter_vec_op_count,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += per_iter_flit_num

                    # Src DRAM access description
                    if not chunk_loaded[src_core_id][chunk_id]:
                        chunk_loaded[src_core_id][chunk_id] = True
                        per_core_all_reduce_description[src_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=input_placement["name"],
                                is_write=False,
                                access_base=[0, input_column_offset + chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(input_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t - 1,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
                    
                    # Dst DRAM access description
                    if dst_idx>=0 and dst_idx<=core_array_size-1 and not chunk_loaded[dst_core_id][chunk_id]:
                        chunk_loaded[dst_core_id][chunk_id] = True
                        per_core_all_reduce_description[dst_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=input_placement["name"],
                                is_write=False,
                                access_base=[0, input_column_offset + chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(input_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t - 1,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
    # print("------q comm rs in ar-----")
    
    # print("------q comm ag in ar-----")
    # All gather in all reduce
    init_iter = core_array_size
    chunk_stored = [[False for _ in range(core_array_size)] for _ in range(core_num)]
    # On-chip communication description
    for t in range(core_array_size-1):
        # Parallel core columns
        for i in range(core_array_size):
            # Each column conducting all gather
            for j in range(core_array_size):
                src_core_id = j*core_array_size + i if reduce_direction == 0 else i*core_array_size + j
                dst_info_list = []
                if j >= t:
                    dst_core_id = (j+1)*core_array_size + i if reduce_direction == 0 else i*core_array_size + j+1
                    chunk_id = core_array_size-1-abs(j-t)
                    dst_info_list.append((j+1, dst_core_id, chunk_id))
                if j <= core_array_size-1-t:
                    dst_core_id = (j-1)*core_array_size + i if reduce_direction == 0 else i*core_array_size + j-1
                    chunk_id = abs(j+t-(core_array_size-1))
                    dst_info_list.append((j-1, dst_core_id, chunk_id))
                # print(f"{t} {i} {j} {src_core_id} {dst_info_list}")
                
                for dst_idx, dst_core_id, chunk_id in dst_info_list:
                    if dst_idx>=0 and dst_idx<=core_array_size-1:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_all_reduce_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_all_reduce_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += per_iter_flit_num

                    # Src DRAM access description
                    if not chunk_stored[src_core_id][chunk_id]:
                        chunk_stored[src_core_id][chunk_id] = True
                        per_core_all_reduce_description[src_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, i * q_length + chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
                    
                    # Dst DRAM access description
                    if dst_idx>=0 and dst_idx<=core_array_size-1 and not chunk_stored[dst_core_id][chunk_id]:
                        chunk_stored[dst_core_id][chunk_id] = True
                        per_core_all_reduce_description[dst_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, i * q_length + chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
    # print("------q comm ag in ar-----")
    
    ############################################
    #         Part 2: Y-axis All Gather        #
    ############################################

    all_gather_iteration = core_array_size - 1
    all_gather_description = {
        "name": name+"_comm_q_all_gather",
        "type": str(SimulatorOperatorType.Communication),
        "iteration": all_gather_iteration,
        "core_num": core_num,
        "file_prefix": name+"_comm_q_all_gather/", # Need to join root directory storing full computation description
    }
    per_core_all_gather_description = [{
        "communication": {
            "on_chip": [{
                "tx": {
                    "buffer_load": [],
                    "noc": [],
                },
                "rx": {
                    "buffer_load": [],
                    "buffer_store": [],
                    "vector": [],
                    "noc": [],
                }
            } for _ in range(all_gather_iteration)],
            "dram": [],
        }
    } for _ in range(core_num)]

    per_iter_chunk_length = q_length
    per_iter_chunk_byte = row_num * per_iter_chunk_length * element_size
    per_iter_flit_num = math.ceil(per_iter_chunk_byte / flit_size)
    per_iter_vec_op_count = row_num * per_iter_chunk_length * op_num_per_element
    
    # print("------q comm ag-----")
    # All gather
    init_iter = 1
    chunk_stored = [[False for _ in range(core_array_size)] for _ in range(core_num)]
    for t in range(core_array_size-1):
        # Parallel core rows
        for i in range(core_array_size):
            # Each row conducting all gather
            for j in range(core_array_size):
                src_core_id = i*core_array_size + j if reduce_direction == 0 else i*core_array_size + j
                dst_info_list = []
                if j >= t:
                    dst_core_id = i*core_array_size + (j+1) if reduce_direction == 0 else i*core_array_size + j+1
                    chunk_id = core_array_size-1-abs(j-t)
                    # NOTE: After X-axis AR, in each row, rank i has ith chunk
                    # Since in tidalmesh, the allgather is conducted when rank i has (N-1-i)th chunk,
                    # we need to convert the chunk id to operate
                    chunk_id = core_array_size-1-chunk_id
                    dst_info_list.append((j+1, dst_core_id, chunk_id))
                if j <= core_array_size-1-t:
                    dst_core_id = i*core_array_size + (j-1) if reduce_direction == 0 else i*core_array_size + j-1
                    chunk_id = abs(j+t-(core_array_size-1))
                    # NOTE: After X-axis AR, in each row, rank i has ith chunk
                    # Since in tidalmesh, the allgather is conducted when rank i has (N-1-i)th chunk,
                    # we need to convert the chunk id to operate
                    chunk_id = core_array_size-1-chunk_id
                    dst_info_list.append((j-1, dst_core_id, chunk_id))
                # print(f"{t} {i} {j} {src_core_id} {dst_info_list}")
                
                # Since the data can be stored on SRAM in previous communication
                # We do not need to really load data
                if t == 0:
                    # For chunk load, we only need the jth core to load the jth chunk in iteration 0
                    chunk_stored[src_core_id][j] = True
                
                for dst_idx, dst_core_id, chunk_id in dst_info_list:
                    if dst_idx>=0 and dst_idx<=core_array_size-1:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_all_gather_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_all_gather_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += per_iter_flit_num

                    # Src DRAM access description
                    if not chunk_stored[src_core_id][chunk_id]:
                        chunk_stored[src_core_id][chunk_id] = True
                        per_core_all_gather_description[src_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
                    
                    # Dst DRAM access description
                    if dst_idx>=0 and dst_idx<=core_array_size-1 and not chunk_stored[dst_core_id][chunk_id]:
                        chunk_stored[dst_core_id][chunk_id] = True
                        per_core_all_gather_description[dst_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
    # print("------q comm ag-----")

    ############################################
    #         Get Communication Latency        #
    ############################################

    if build_only:
        return ('communication', [(all_reduce_description, per_core_all_reduce_description),
                                   (all_gather_description, per_core_all_gather_description)], None)

    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
        ]
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)
    
    comp_description_path = os.path.join(cur_op_dir, "comp_description.yaml")
    comp_description = {
        "operator" : [
            copy.deepcopy(all_reduce_description),
            copy.deepcopy(all_gather_description),
        ]
    }
    for operator in comp_description["operator"]:
        operator["file_prefix"] = os.path.join(cur_op_dir, operator["file_prefix"])
        os.makedirs(operator["file_prefix"], exist_ok=True)
    with open(comp_description_path, "w") as f:
        yaml.dump(comp_description, f)

    per_core_description_lists = [per_core_all_reduce_description, per_core_all_gather_description]
    for i, per_core_description_list in enumerate(per_core_description_lists):
        for j, per_core_description in enumerate(per_core_description_list):
            output_dir = comp_description["operator"][i]["file_prefix"]
            with open(os.path.join(output_dir, f"core_{j}.yaml"), "w") as f:
                yaml.dump(per_core_description, f)

    import atlasim
    log_file_path = os.path.join(cur_op_dir, "operator_output.log")
    with open(log_file_path, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with atlasim.ostream_redirect(stdout=True, stderr=True):
                chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
                performance = chip.simulate()
                del chip

    latency = performance.e2e_stats.e2e_cycles

    ############################################
    #        Return Generated Description      #
    ############################################

    comm_description = (
        "communication",
        [
            (all_reduce_description, per_core_all_reduce_description),
            (all_gather_description, per_core_all_gather_description),
        ],
        latency,
    )
    return comm_description


def generate_kv_proj_comm(
    # Tensor description
    input_placement: Dict[str, Any], # Current operator's output tensor
    kv_cache_placement: Dict[str, Any], # Next operator's input tensor
    # Operator shape description
    name: str,
    row_num: int, q_length: int, kv_length: int, # lengths in each core's qkv_proj shard
    total_slot_num: int, num_kv_heads: int, 
    element_size: int, op_num_per_element: int,
    # Chip config
    chip_config_path: str,
    core_num: int, core_array_size: int,
    flit_size: int, core_flit_index_list: List[int],
    # Online serving parameters
    last_block_mapping: List[List[List[int]]], # List[List[]] indicates 2D core array, List[int] indicates slot id to write
    # Intermediate results storing configs
    intermediate_result_dir: str = "",
    *, build_only: bool = False,
):
    assert total_slot_num * num_kv_heads == tensor_dim(kv_cache_placement, 0, expected_rank=2)
    assert kv_length * core_array_size == tensor_dim(kv_cache_placement, 1, expected_rank=2) * num_kv_heads
    initial_flit_index_list = copy.deepcopy(core_flit_index_list)
    cur_op_dir = os.path.join(intermediate_result_dir, name+"_comm_kv")
    if not build_only:
        os.makedirs(cur_op_dir, exist_ok=True)

    iter_count_per_stage = math.ceil(core_array_size / 2)
    kv_comm_description = {
        "name": name+"_comm_kv",
        "type": str(SimulatorOperatorType.Communication),
        "iteration": iter_count_per_stage*4 - 2,
        "core_num": core_num,
        "file_prefix": name+"_comm_kv/", # Need to join root directory storing full computation description
    }
    per_core_kv_comm_description = [{
        "communication": {
            "on_chip": [{
                "tx": {
                    "buffer_load": [],
                    "noc": [],
                },
                "rx": {
                    "buffer_load": [],
                    "buffer_store": [],
                    "vector": [],
                    "noc": [],
                }
            } for _ in range(iter_count_per_stage*4 - 2)],
            "dram": [],
        }
    } for _ in range(core_num)]

    tree_depth, mst_trees = generate_1d_mesh_mst_trees(core_array_size)

    ############################################
    #           Part 1: X-axis Reduce          #
    ############################################

    first_half_req_num = 0
    for i in range((core_array_size//2)):
        for j in range(core_array_size):
            first_half_req_num += len(last_block_mapping[i][j])
    
    second_half_req_num = 0
    for i in range((core_array_size//2), core_array_size):
        for j in range(core_array_size):
            second_half_req_num += len(last_block_mapping[i][j])
    
    send_req_num_list = [first_half_req_num, second_half_req_num]
    init_iter = 1
    # print("------kv comm x-axis reduce-----")
    for t in range(iter_count_per_stage):
        # Parallel core columns
        for i in range(core_array_size):
            # Load full kv vector partial sum from DRAM
            if t == 0:
                 # Each column conducting reduce scatter
                for j in range(core_array_size):
                    src_core_id = j*core_array_size + i
                    per_core_kv_comm_description[src_core_id]["communication"]["dram"].append(
                        make_dram_task(
                            name=input_placement["name"],
                            is_write=False,
                            access_base=[0, q_length],
                            access_extent=[row_num, kv_length],
                            access_stride_add=[1, 1],
                            access_offset_add=[tensor_dim(input_placement, 0, expected_rank=2), kv_length],
                            init_iter=init_iter + t - 1,
                            stride_iter=1,
                            total_iter=1,
                        )
                    )

            # Send partial sum according to tree depth
            cur_send_depth = tree_depth - t
            for tree, send_req_num in zip(mst_trees, send_req_num_list):
                root = tree[0][0]
                cur_depth_nodes = tree[cur_send_depth]
                for node in cur_depth_nodes:
                    if node == root or node < 0 or node > core_array_size-1:
                        continue
                    src_core_id = node*core_array_size + i
                    dst_core_id = (node+1)*core_array_size + i if node < root else (node-1)*core_array_size + i
                    dst_idx = (node+1) if node < root else (node-1)
                    # print(f"{t} {i} {src_core_id} {dst_core_id} {node} {dst_idx} {send_req_num}")

                    if dst_idx >= 0 and dst_idx <= core_array_size-1 and send_req_num > 0:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_kv_comm_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": send_req_num * kv_length * element_size,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })
                        
                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_kv_comm_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_load"].append({
                            "name": f"reduce{src_core_id}_load",
                            "byte_count": 2 * send_req_num * kv_length * element_size,
                            "is_write": False,
                        })
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": send_req_num * kv_length * element_size,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"reduce{src_core_id}_store",
                            "byte_count": send_req_num * kv_length * element_size,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["vector"].append({
                            "name": f"reduce{src_core_id}",
                            "vec_count": send_req_num * kv_length * op_num_per_element,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += math.ceil(send_req_num * kv_length * element_size / flit_size)
    # print("------kv comm x-axis reduce-----")
    
    ############################################
    #           Part 2: X-axis Scatter         #
    ############################################

    # We scatter from center to edge, the request number gradually reduces
    first_half_per_hop_req_num = []
    tmp_req_num = 0
    # 1st MST is traversed reversely, from the second PE (the first PE does not need to transfer)
    for i in range((core_array_size//2)-1):
        tmp_row_req_num = 0
        for j in range(core_array_size):
            tmp_row_req_num += len(last_block_mapping[i][j])
        tmp_req_num += tmp_row_req_num
        first_half_per_hop_req_num.append(tmp_req_num)
    first_half_per_hop_req_num.reverse()

    second_half_per_hop_req_num = []
    tmp_req_num = 0
    # 2nd MST is traversed normally, from the second PE (the first PE does not need to transfer)
    for i in range(core_array_size-1, (core_array_size//2), -1):
        tmp_row_req_num = 0
        for j in range(core_array_size):
            tmp_row_req_num += len(last_block_mapping[i][j])
        tmp_req_num += tmp_row_req_num
        second_half_per_hop_req_num.append(tmp_req_num)
    second_half_per_hop_req_num.reverse()

    send_req_num_lists = [first_half_per_hop_req_num, second_half_per_hop_req_num]
    dst_offset_list = [-1, 1]
    init_iter = iter_count_per_stage + 1
    # print("------kv comm x-axis scatter-----")
    for t in range(iter_count_per_stage-1):
        # Parallel core columns
        for i in range(core_array_size):
            # Send fully reduced sub-vector according to tree depth
            cur_send_depth = t
            for tree, send_req_num_list, dst_offset in zip(mst_trees, send_req_num_lists, dst_offset_list):
                root = tree[0][0]
                cur_depth_nodes = tree[cur_send_depth]
                send_req_num = send_req_num_list[t]
                for node in cur_depth_nodes:
                    if node < 0 or node > core_array_size-1:
                        continue
                    if (node-root)*dst_offset < 0:
                        continue
                    src_core_id = node*core_array_size + i
                    dst_core_id = (node+dst_offset)*core_array_size + i
                    dst_idx = node + dst_offset
                    # print(f"{t} {i} {src_core_id} {dst_core_id} {node} {dst_idx} {send_req_num}")

                    if dst_idx >= 0 and dst_idx <= core_array_size-1 and send_req_num > 0:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_kv_comm_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": send_req_num * kv_length * element_size,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_kv_comm_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": send_req_num * kv_length * element_size,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += math.ceil(send_req_num * kv_length * element_size / flit_size)
    # print("------kv comm x-axis scatter-----")

    ############################################
    #           Part 3: Y-axis Gather          #
    ############################################

    init_iter = iter_count_per_stage * 2
    # print("------kv comm y-axis gather-----")
    for i in range(core_array_size):
        # Get 1st MST's req num
        first_half_req_num = 0
        for j in range((core_array_size//2)):
            first_half_req_num += len(last_block_mapping[i][j])
        
        # Get 2nd MST's req num
        second_half_req_num = 0
        for j in range((core_array_size//2), core_array_size):
            second_half_req_num += len(last_block_mapping[i][j])

        send_req_num_list = [first_half_req_num, second_half_req_num]
        longer_path_list = [1, -1] # -1 means the part of (root-node < 0) has longer path
        for t in range(iter_count_per_stage):
            # Send sub-vector collection according to tree depth
            cur_send_depth = tree_depth - t
            for tree, send_req_num, longer_path in zip(mst_trees, send_req_num_list, longer_path_list):
                root = tree[0][0]
                cur_depth_nodes = tree[cur_send_depth]
                for node in cur_depth_nodes:
                    if node == root or node < 0 or node > core_array_size-1:
                        continue
                    src_core_id = i*core_array_size + node
                    dst_core_id = i*core_array_size + (node+1) if node < root else i*core_array_size + (node-1)
                    dst_idx = node+1 if node < root else node-1

                    # NOTE: vector length is gradually increasing during the gather process
                    if (node-root)*longer_path > 0:
                        chunk_num = t+1
                    else:
                        chunk_num = t
                    
                    # print(f"{t} {i} {src_core_id} {dst_core_id} {node} {dst_idx} {chunk_num} {send_req_num}")
                    if dst_idx >= 0 and dst_idx <= core_array_size-1 and send_req_num*chunk_num > 0:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_kv_comm_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": chunk_num * send_req_num * kv_length * element_size,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(chunk_num * send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_kv_comm_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": chunk_num * send_req_num * kv_length * element_size,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(chunk_num * send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += math.ceil(chunk_num * send_req_num * kv_length * element_size / flit_size)
    # print("------kv comm y-axis gather-----")

    ############################################
    #           Part 4: Y-axis Scatter         #
    ############################################

    init_iter = iter_count_per_stage * 3
    # print("------kv comm y-axis scatter-----")
    for i in range(core_array_size):
        # Similar to X scatter, we scatter from center to edge, the request number gradually reduces
        first_half_per_hop_req_num = []
        tmp_req_num = 0
        # 1st MST is traversed reversely, from the second PE (the first PE does not need to transfer)
        for j in range((core_array_size//2)-1):
            tmp_req_num += len(last_block_mapping[i][j])
            first_half_per_hop_req_num.append(tmp_req_num)
        first_half_per_hop_req_num.reverse()

        second_half_per_hop_req_num = []
        tmp_req_num = 0
        # 2nd MST is traversed normally, from the second PE (the first PE does not need to transfer)
        for j in range(core_array_size-1, (core_array_size//2), -1):
            tmp_req_num += len(last_block_mapping[i][j])
            second_half_per_hop_req_num.append(tmp_req_num)
        second_half_per_hop_req_num.reverse()

        send_req_num_lists = [first_half_per_hop_req_num, second_half_per_hop_req_num]
        dst_offset_list = [-1, 1]
        for t in range(iter_count_per_stage-1):
            # Send sub-vector collection according to tree depth
            cur_send_depth = t
            for tree, send_req_num_list, dst_offset in zip(mst_trees, send_req_num_lists, dst_offset_list):
                root = tree[0][0]
                cur_depth_nodes = tree[cur_send_depth]
                send_req_num = send_req_num_list[t]
                for node in cur_depth_nodes:
                    if node < 0 or node > core_array_size-1:
                        continue
                    if (node-root)*dst_offset < 0:
                        continue
                    src_core_id = i*core_array_size + node
                    dst_core_id = i*core_array_size + (node+dst_offset)
                    dst_idx = node + dst_offset
                    # print(f"{t} {i} {src_core_id} {dst_core_id} {node} {dst_idx} {send_req_num}")

                    if dst_idx >= 0 and dst_idx <= core_array_size-1 and send_req_num > 0:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_kv_comm_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": core_array_size * send_req_num * kv_length * element_size,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(core_array_size * send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })
                        
                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_kv_comm_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": core_array_size * send_req_num * kv_length * element_size,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": math.ceil(core_array_size * send_req_num * kv_length * element_size / flit_size),
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })
                        
                        # Update flit count
                        core_flit_index_list[src_core_id] += math.ceil(core_array_size * send_req_num * kv_length * element_size / flit_size)

    # Always commit the completed vector, including a 2x2 mesh whose final
    # scatter stage has zero hops. Slot IDs are local to the destination core.
    for i in range(core_array_size):
        for j in range(core_array_size):
            for h in range(num_kv_heads):
                for slot_id in last_block_mapping[i][j]:
                    if not 0 <= slot_id < total_slot_num:
                        raise ValueError("KV destination slot is outside the cache")
                    per_core_kv_comm_description[i*core_array_size+j]["communication"]["dram"].append(
                        make_dram_task(
                            name=kv_cache_placement["name"], is_write=True,
                            access_base=[h*total_slot_num+slot_id, 0],
                            access_extent=[1, tensor_dim(kv_cache_placement, 1, expected_rank=2)],
                            access_stride_add=[1, 1],
                            access_offset_add=[1, tensor_dim(kv_cache_placement, 1, expected_rank=2)],
                            init_iter=kv_comm_description["iteration"] + 1,
                            stride_iter=1, total_iter=1,
                        )
                    )
    # print("------kv comm y-axis scatter-----")

    ############################################
    #        Return Generated Description      #
    ############################################

    if build_only:
        return ('communication', [(kv_comm_description, per_core_kv_comm_description)], None)

    data_placement_config = {
        "tensor": [
            input_placement,
            kv_cache_placement,
        ]
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)

    comp_description_path = os.path.join(cur_op_dir, "comp_description.yaml")
    comp_description = {
        "operator": [
            copy.deepcopy(kv_comm_description),
        ]
    }
    for operator in comp_description["operator"]:
        operator["file_prefix"] = os.path.join(cur_op_dir, operator["file_prefix"])
        os.makedirs(operator["file_prefix"], exist_ok=True)
    with open(comp_description_path, "w") as f:
        yaml.dump(comp_description, f)

    output_dir = comp_description["operator"][0]["file_prefix"]
    for core_id, per_core_description in enumerate(per_core_kv_comm_description):
        with open(os.path.join(output_dir, f"core_{core_id}.yaml"), "w") as f:
            yaml.dump(per_core_description, f)

    import atlasim
    log_file_path = os.path.join(cur_op_dir, "operator_output.log")
    with open(log_file_path, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with atlasim.ostream_redirect(stdout=True, stderr=True):
                chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
                performance = chip.simulate()
                del chip

    latency = performance.e2e_stats.e2e_cycles

    comm_description = (
        "communication",
        [
            (
                kv_comm_description,
                per_core_kv_comm_description, 
                (
                    input_placement,
                    kv_cache_placement,
                    name, 
                    q_length, kv_length, 
                    total_slot_num, num_kv_heads, 
                    initial_flit_index_list
                )
            ),
        ],
        latency,
    )
    return comm_description


def generate_1d_allreduce_comm(
    # Tensor description
    input_placement: Dict[str, Any], # Current operator's output tensor
    output_placement: Dict[str, Any], # Next operator's input tensor
    # Operator shape description
    name: str,
    row_num: int, input_column_offset: int, input_column_length: int,
    element_size: int, op_num_per_element: int,
    reduce_direction: int, # 0: X-axis, 1: Y-axis
    # Chip config
    chip_config_path: str,
    core_num: int, core_array_size: int,
    flit_size: int, core_flit_index_list: List[int],
    # Intermediate result directory
    intermediate_result_dir: str,
    gated: bool = False,
):
    cur_op_dir = os.path.join(intermediate_result_dir, name+"_comm_1d_all_reduce")
    os.makedirs(cur_op_dir, exist_ok=True)

    from frontend.auto.collectives import allreduce_tasks
    if core_num != core_array_size**2:
        raise ValueError("Collective core count does not match the mesh")
    all_reduce_description, per_core_all_reduce_description = allreduce_tasks(
        input_placement, output_placement, name, row_num, input_column_offset,
        input_column_length, element_size, op_num_per_element, reduce_direction,
        core_array_size, flit_size, core_flit_index_list, gated=gated,
    )

    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
        ]
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)
    
    comp_description_path = os.path.join(cur_op_dir, "comp_description.yaml")
    comp_description = {
        "operator" : [
            copy.deepcopy(all_reduce_description),
        ]
    }
    for operator in comp_description["operator"]:
        operator["file_prefix"] = os.path.join(cur_op_dir, operator["file_prefix"])
        os.makedirs(operator["file_prefix"], exist_ok=True)
    with open(comp_description_path, "w") as f:
        yaml.dump(comp_description, f)

    per_core_description_lists = [per_core_all_reduce_description]
    for i, per_core_description_list in enumerate(per_core_description_lists):
        for j, per_core_description in enumerate(per_core_description_list):
            output_dir = comp_description["operator"][i]["file_prefix"]
            with open(os.path.join(output_dir, f"core_{j}.yaml"), "w") as f:
                yaml.dump(per_core_description, f)

    import atlasim
    log_file_path = os.path.join(cur_op_dir, "operator_output.log")
    with open(log_file_path, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with atlasim.ostream_redirect(stdout=True, stderr=True):
                chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
                performance = chip.simulate()
                del chip

    latency = performance.e2e_stats.e2e_cycles

    ############################################
    #        Return Generated Description      #
    ############################################

    comm_description = (
        "communication",
        [
            (all_reduce_description, per_core_all_reduce_description),
        ],
        latency,
    )
    return comm_description


def generate_attention_comp(
    # Tensor description
    input_placement: Dict[str, Any],
    output_placement: Dict[str, Any],
    weight_placement_list: List[Dict[str, Any]],
    # Operator shape description
    operator: Operator,
    element_size: int,
    context_length_list: List[List[List[int]]],
    # System config
    cloud_config: CloudSystemConfig,
):
    assert operator.op_type == OperatorType.ATTENTION
    if operator.is_mla:
        kv_vector_length = operator.head_dim
    else:
        kv_vector_length = operator.v_head_dim + operator.head_dim
    kv_vector_size = kv_vector_length * element_size

    sram_buffer_size = cloud_config.chip_config.buffer_config.buffer_size * _KB
    assert sram_buffer_size / 2 >= kv_vector_size
    token_tile_size = math.floor(sram_buffer_size/2 / kv_vector_size)
    for i in range(len(context_length_list)):
        for j in range(len(context_length_list[i])):
            for context_length in context_length_list[i][j]:
                token_tile_size = min(token_tile_size, context_length)
    token_tile_size = max(token_tile_size, 1)

    # reduce_max, max, sub, exp, sub, exp, mul, reduce_sum, add
    softmax_vector_count = 9
    # mul, add, div
    accumulation_vector_count = 3

    assert len(weight_placement_list) == 1
    kv_cache_placement = weight_placement_list[0]
    assert kv_cache_placement["name"] == "kv_cache"
    total_slot_num = tensor_dim(kv_cache_placement, 0, expected_rank=2) // operator.kv_head_num

    attention_comp_description = {
        "name": operator.name,
        "type": str(SimulatorOperatorType.DecodeAttention),
        "iteration": -1,
        "execution": {
            "matrix": [
                {
                    "name": operator.name + "_qk",
                    "mac_count": operator.kv_group_num * operator.head_dim * token_tile_size,
                },
                {
                    "name": operator.name + "_sv",
                    "mac_count": operator.kv_group_num * token_tile_size * operator.v_head_dim,
                }
            ],
            "vector": [
                {
                    "name": operator.name + "_softmax",
                    "vec_count": operator.kv_group_num * token_tile_size * softmax_vector_count,
                },
                {
                    "name": operator.name + "_accumulation",
                    "vec_count": operator.kv_group_num * operator.v_head_dim * accumulation_vector_count,
                }
            ],
            "buffer_load": [
                {
                    "name": operator.name + "_qk_load",
                    "byte_count": (operator.kv_group_num*operator.head_dim + operator.head_dim*token_tile_size) * element_size,
                    "is_write": False,
                },
                {
                    "name": operator.name + "_softmax_load",
                    "byte_count": operator.kv_group_num * token_tile_size * softmax_vector_count * element_size,
                    "is_write": False,
                },
                {
                    "name": operator.name + "_sv_load",
                    "byte_count": (operator.kv_group_num*token_tile_size + token_tile_size*operator.v_head_dim) * element_size,
                    "is_write": False,
                },
                {
                    "name": operator.name + "_accumulation_load",
                    "byte_count": operator.kv_group_num * operator.v_head_dim * element_size,
                    "is_write": False,
                }
            ],
            "buffer_store": [
                {
                    "name": operator.name + "_qk_store",
                    "byte_count": operator.kv_group_num * token_tile_size * element_size,
                    "is_write": True,
                },
                {
                    "name": operator.name + "_softmax_store",
                    "byte_count": operator.kv_group_num * token_tile_size * element_size,
                    "is_write": True,
                },
                {
                    "name": operator.name + "_sv_store",
                    "byte_count": operator.kv_group_num * operator.v_head_dim * element_size,
                    "is_write": True,
                },
                {
                    "name": operator.name + "_accumulation_store",
                    "byte_count": operator.kv_group_num * operator.v_head_dim * element_size,
                    "is_write": True,
                }
            ],
            "dram": []
        }
    }
    attention_input_template = {
        # Attention shape information
        "q_head_num": operator.kv_group_num * operator.kv_head_num,
        "kv_head_num": operator.kv_head_num,
        "q_head_num_per_kv": operator.kv_group_num,
        "qk_head_dim": operator.head_dim,
        "v_head_dim": operator.v_head_dim,
        # Tensor information
        "input_tensor_name": input_placement["name"],
        "output_tensor_name": output_placement["name"],
        "kv_cache_tensor_name": weight_placement_list[0]["name"],
        "total_slot_num": total_slot_num,
        "kv_head_addr_offset": total_slot_num * kv_vector_length * element_size,
        # Token number information for decoding attention
        "token_tile_size": token_tile_size,
        "block_size": cloud_config.block_size,
        "total_block_num": math.floor(total_slot_num / cloud_config.block_size),
        "random": False,
        "random_seed": 0,
        "core_input_list": [] # Depend on runtime batch size
    }
    return {
        "attention_comp_description": attention_comp_description,
        "attention_input_template": attention_input_template,
    }


def generate_attention_comm(
    # Tensor description
    input_placement: Dict[str, Any], # Current operator's output tensor
    output_placement: Dict[str, Any], # Next operator's input tensor
    # Operator shape description
    row_num: int,
    element_size: int, op_num_per_element: int,
    # Chip config
    chip_config_path: str,
    core_num: int, core_array_size: int,
    flit_size: int, core_flit_index_list: List[int],
    # Intermediate result directory
    intermediate_result_dir: str,
):
    cur_op_dir = os.path.join(intermediate_result_dir, "attention_comm_attn_reduce_scatter")
    os.makedirs(cur_op_dir, exist_ok=True)

    ############################################
    #         Part 1: 2D Reduce Scatter        #
    ############################################

    reduce_scatter_iteration = core_num-1
    reduce_scatter_description = {
        "name": "attention_comm_attn_reduce_scatter",
        "type": str(SimulatorOperatorType.Communication),
        "iteration": reduce_scatter_iteration,
        "core_num": core_num,
        "file_prefix": "attention_comm_attn_reduce_scatter/", # Need to join root directory storing full computation description
    }
    per_core_reduce_scatter_description = [{
        "communication": {
            "on_chip": [{
                "tx": {
                    "buffer_load": [],
                    "noc": [],
                },
                "rx": {
                    "buffer_load": [],
                    "buffer_store": [],
                    "vector": [],
                    "noc": [],
                }
            } for _ in range(reduce_scatter_iteration)],
            "dram": [],
        }
    } for _ in range(core_num)]

    per_iter_chunk_length = math.ceil(tensor_dim(input_placement, 1, expected_rank=2) / core_num)
    per_iter_chunk_byte = row_num * per_iter_chunk_length * element_size
    per_iter_flit_num = math.ceil(per_iter_chunk_byte / flit_size)
    per_iter_vec_op_count = row_num * per_iter_chunk_length * op_num_per_element

    rank_core_map = generate_mesh_ring_topology(core_array_size)
    
    # Reduce scatter in all reduce
    init_iter = 1
    chunk_loaded = [[False for _ in range(core_num)] for _ in range(core_num)]
    # print("------rs in attn comm-----")
    for t in range(reduce_scatter_iteration):
        # print(f"iter {t}")
        for r in range(core_num):
            src_core_id = rank_core_map[r]
            dst_core_id = rank_core_map[(r+1)%core_num]
            chunk_id = rank_core_map[(core_num-1+r-t)%core_num]
            # print(f"{src_core_id} {dst_core_id} {chunk_id}")

            # Src noc tx description
            cur_iter_cur_src_tx = per_core_reduce_scatter_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
            cur_iter_cur_src_tx["buffer_load"].append({
                "name": f"noc_tx{dst_core_id}_load",
                "byte_count": per_iter_chunk_byte,
                "is_write": False,
            })
            cur_iter_cur_src_tx["noc"].append({
                "name": f"noc_tx{dst_core_id}",
                "is_send": True,
                "src": src_core_id,
                "dst": dst_core_id,
                "flit_num": per_iter_flit_num,
                "init_flit_id": core_flit_index_list[src_core_id],
            })

            # Dst noc rx description
            cur_iter_cur_dst_rx = per_core_reduce_scatter_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
            cur_iter_cur_dst_rx["buffer_load"].append({
                "name": f"reduce{src_core_id}_load",
                "byte_count": per_iter_chunk_byte * 2,
                "is_write": False,
            })
            cur_iter_cur_dst_rx["buffer_store"].append({
                "name": f"noc_rx{src_core_id}_store",
                "byte_count": per_iter_chunk_byte,
                "is_write": True,
            })
            cur_iter_cur_dst_rx["buffer_store"].append({
                "name": f"reduce{src_core_id}_store",
                "byte_count": per_iter_chunk_byte,
                "is_write": True,
            })
            cur_iter_cur_dst_rx["vector"].append({
                "name": f"reduce{src_core_id}",
                "vec_count": per_iter_vec_op_count,
            })
            cur_iter_cur_dst_rx["noc"].append({
                "name": f"noc_rx{src_core_id}",
                "is_send": False,
                "src": src_core_id,
                "dst": dst_core_id,
                "flit_num": per_iter_flit_num,
                "init_flit_id": core_flit_index_list[src_core_id],
            })

            # Update flit count
            core_flit_index_list[src_core_id] += per_iter_flit_num

            # Src DRAM access description
            if not chunk_loaded[src_core_id][chunk_id]:
                chunk_loaded[src_core_id][chunk_id] = True
                per_core_reduce_scatter_description[src_core_id]["communication"]["dram"].append(
                    make_dram_task(
                        name=input_placement["name"],
                        is_write=False,
                        access_base=[0, chunk_id * per_iter_chunk_length],
                        access_extent=[row_num, per_iter_chunk_length],
                        access_stride_add=[1, 1],
                        access_offset_add=[tensor_dim(input_placement, 0, expected_rank=2), per_iter_chunk_length],
                        init_iter=init_iter + t - 1,
                        stride_iter=1,
                        total_iter=1,
                    )
                )
            
            # Dst DRAM access description
            if not chunk_loaded[dst_core_id][chunk_id]:
                chunk_loaded[dst_core_id][chunk_id] = True
                per_core_reduce_scatter_description[dst_core_id]["communication"]["dram"].append(
                    make_dram_task(
                        name=input_placement["name"],
                        is_write=False,
                        access_base=[0, chunk_id * per_iter_chunk_length],
                        access_extent=[row_num, per_iter_chunk_length],
                        access_stride_add=[1, 1],
                        access_offset_add=[tensor_dim(input_placement, 0, expected_rank=2), per_iter_chunk_length],
                        init_iter=init_iter + t - 1,
                        stride_iter=1,
                        total_iter=1,
                    )
                )
    # print("------rs in attn comm-----")
    
    ############################################
    #         Part 2: Y-axis All Gather        #
    ############################################

    all_gather_iteration = core_array_size - 1
    all_gather_description = {
        "name": "attention_comm_attn_all_gather",
        "type": str(SimulatorOperatorType.Communication),
        "iteration": all_gather_iteration,
        "core_num": core_num,
        "file_prefix": "attention_comm_attn_all_gather/", # Need to join root directory storing full computation description
    }
    per_core_all_gather_description = [{
        "communication": {
            "on_chip": [{
                "tx": {
                    "buffer_load": [],
                    "noc": [],
                },
                "rx": {
                    "buffer_load": [],
                    "buffer_store": [],
                    "vector": [],
                    "noc": [],
                }
            } for _ in range(all_gather_iteration)],
            "dram": [],
        }
    } for _ in range(core_num)]

    per_iter_chunk_length = math.ceil(tensor_dim(output_placement, 1, expected_rank=2) / core_array_size)
    per_iter_chunk_byte = row_num * per_iter_chunk_length * element_size
    per_iter_flit_num = math.ceil(per_iter_chunk_byte / flit_size)
    per_iter_vec_op_count = row_num * per_iter_chunk_length * op_num_per_element

    # All gather
    init_iter = 1
    chunk_stored = [[False for _ in range(core_array_size)] for _ in range(core_num)]
    # print("------ag in attn comm-----")
    for t in range(core_array_size-1):
        # Parallel core rows
        for i in range(core_array_size):
            # Each row conducting all gather
            for j in range(core_array_size):
                src_core_id = i*core_array_size + j
                dst_info_list = []
                if j >= t:
                    dst_core_id = i*core_array_size + (j+1)
                    chunk_id = core_array_size-1-abs(j-t)
                    # NOTE: After 2D RS, in each row, rank i has ith chunk
                    # Since in tidalmesh, the allgather is conducted when rank i has (N-1-i)th chunk,
                    # we need to convert the chunk id to operate
                    chunk_id = core_array_size-1-chunk_id
                    dst_info_list.append((j+1, dst_core_id, chunk_id))
                if j <= core_array_size-1-t:
                    dst_core_id = i*core_array_size + (j-1)
                    chunk_id = abs(j+t-(core_array_size-1))
                    # NOTE: After 2D RS, in each row, rank i has ith chunk
                    # Since in tidalmesh, the allgather is conducted when rank i has (N-1-i)th chunk,
                    # we need to convert the chunk id to operate
                    chunk_id = core_array_size-1-chunk_id
                    dst_info_list.append((j-1, dst_core_id, chunk_id))
                # print(f"{t} {i} {j} {src_core_id} {dst_info_list}")

                # Since the fully reduced data are only buffered in SRAM,
                # we need to conduct DRAM store here
                if t == 0:
                    # For chunk load, we only need the jth core to load the jth chunk in iteration 0
                    chunk_stored[src_core_id][j] = True
                    per_core_all_gather_description[src_core_id]["communication"]["dram"].append(
                        make_dram_task(
                            name=output_placement["name"],
                            is_write=True,
                            access_base=[0, j * per_iter_chunk_length],
                            access_extent=[row_num, per_iter_chunk_length],
                            access_stride_add=[1, 1],
                            access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                            init_iter=init_iter + t - 1,
                            stride_iter=1,
                            total_iter=1,
                        )
                    )
                
                for dst_idx, dst_core_id, chunk_id in dst_info_list:
                    if dst_idx>=0 and dst_idx<=core_array_size-1:
                        # Src noc tx description
                        cur_iter_cur_src_tx = per_core_all_gather_description[src_core_id]["communication"]["on_chip"][init_iter+t-1]["tx"]
                        cur_iter_cur_src_tx["buffer_load"].append({
                            "name": f"noc_tx{dst_core_id}_load",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": False,
                        })
                        cur_iter_cur_src_tx["noc"].append({
                            "name": f"noc_tx{dst_core_id}",
                            "is_send": True,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Dst noc rx description
                        cur_iter_cur_dst_rx = per_core_all_gather_description[dst_core_id]["communication"]["on_chip"][init_iter+t-1]["rx"]
                        cur_iter_cur_dst_rx["buffer_store"].append({
                            "name": f"noc_rx{src_core_id}_store",
                            "byte_count": per_iter_chunk_byte,
                            "is_write": True,
                        })
                        cur_iter_cur_dst_rx["noc"].append({
                            "name": f"noc_rx{src_core_id}",
                            "is_send": False,
                            "src": src_core_id,
                            "dst": dst_core_id,
                            "flit_num": per_iter_flit_num,
                            "init_flit_id": core_flit_index_list[src_core_id],
                        })

                        # Update flit count
                        core_flit_index_list[src_core_id] += per_iter_flit_num

                    # Src DRAM access description
                    if not chunk_stored[src_core_id][chunk_id]:
                        chunk_stored[src_core_id][chunk_id] = True
                        per_core_all_gather_description[src_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
                    
                    # Dst DRAM access description
                    if dst_idx>=0 and dst_idx<=core_array_size-1 and not chunk_stored[dst_core_id][chunk_id]:
                        chunk_stored[dst_core_id][chunk_id] = True
                        per_core_all_gather_description[dst_core_id]["communication"]["dram"].append(
                            make_dram_task(
                                name=output_placement["name"],
                                is_write=True,
                                access_base=[0, chunk_id * per_iter_chunk_length],
                                access_extent=[row_num, per_iter_chunk_length],
                                access_stride_add=[1, 1],
                                access_offset_add=[tensor_dim(output_placement, 0, expected_rank=2), per_iter_chunk_length],
                                init_iter=init_iter + t,
                                stride_iter=1,
                                total_iter=1,
                            )
                        )
    # print("------ag in attn comm-----")

    ############################################
    #         Get Communication Latency        #
    ############################################

    data_placement_config = {
        "tensor": [
            input_placement,
            output_placement,
        ]
    }
    data_placement_config_path = os.path.join(cur_op_dir, "data_placement.yaml")
    with open(data_placement_config_path, "w") as f:
        yaml.dump(data_placement_config, f)
    
    comp_description_path = os.path.join(cur_op_dir, "comp_description.yaml")
    comp_description = {
        "operator" : [
            copy.deepcopy(reduce_scatter_description),
            copy.deepcopy(all_gather_description),
        ]
    }
    for operator in comp_description["operator"]:
        operator["file_prefix"] = os.path.join(cur_op_dir, operator["file_prefix"])
        os.makedirs(operator["file_prefix"], exist_ok=True)
    with open(comp_description_path, "w") as f:
        yaml.dump(comp_description, f)

    per_core_description_lists = [per_core_reduce_scatter_description, per_core_all_gather_description]
    for i, per_core_description_list in enumerate(per_core_description_lists):
        for j, per_core_description in enumerate(per_core_description_list):
            output_dir = comp_description["operator"][i]["file_prefix"]
            with open(os.path.join(output_dir, f"core_{j}.yaml"), "w") as f:
                yaml.dump(per_core_description, f)

    import atlasim
    log_file_path = os.path.join(cur_op_dir, "operator_output.log")
    with open(log_file_path, "w") as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            with atlasim.ostream_redirect(stdout=True, stderr=True):
                chip = Chip(chip_config_path, comp_description_path, data_placement_config_path)
                performance = chip.simulate()
                del chip

    latency = performance.e2e_stats.e2e_cycles

    ############################################
    #        Return Generated Description      #
    ############################################

    comm_description = (
        "communication",
        [
            (reduce_scatter_description, per_core_reduce_scatter_description),
            (all_gather_description, per_core_all_gather_description),
        ],
        latency,
    )
    return comm_description


def generate_data_placement_list(
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    # Hardware-related configs
    cloud_config: CloudSystemConfig,
    # Some hyper parameters
    element_size: int = 2,
    dram_row_size: int = 128*1024,
) -> Dict[str, Any]:
    data_placement_list: List[Dict[str, Any]] = []
    data_placement_dict: Dict[str, Dict[str, Any]] = {}

    base_addr = 0
    for operator in attention_block + ffn_moe_block:
        weight_placement_list = []
        if operator.op_type == OperatorType.GEMM:
            input_shape = [operator.M, operator.K]
            input_placement = make_tensor_placement(
                name="input_" + operator.name,
                base_addr=base_addr,
                element_size=element_size,
                shape=input_shape,
                strides=make_contiguous_strides(input_shape, last_dim_contiguous=True),
            )
            input_data_volume = math.prod(input_shape) * element_size
            base_addr += math.ceil(input_data_volume / dram_row_size) * dram_row_size

            output_shape = [operator.M, operator.N]
            output_placement = make_tensor_placement(
                name="output_" + operator.name,
                base_addr=base_addr,
                element_size=element_size,
                shape=output_shape,
                strides=make_contiguous_strides(output_shape, last_dim_contiguous=True),
            )
            output_data_volume = math.prod(output_shape) * element_size
            base_addr += math.ceil(output_data_volume / dram_row_size) * dram_row_size

            for i in range(operator.B):
                weight_shape = [operator.K, operator.N]
                weight_placement = make_tensor_placement(
                    name="weight_" + operator.name + f"_b{i}",
                    base_addr=base_addr,
                    element_size=element_size,
                    shape=weight_shape,
                    strides=make_contiguous_strides(weight_shape, last_dim_contiguous=False),
                )
                weight_data_volume = math.prod(weight_shape) * element_size
                base_addr += math.ceil(weight_data_volume / dram_row_size) * dram_row_size
                weight_placement_list.append(weight_placement)
        elif operator.op_type == OperatorType.ATTENTION:
            input_shape = [sum(operator.input_length), operator.kv_head_num * operator.kv_group_num * operator.head_dim]
            input_placement = make_tensor_placement(
                name="input_" + operator.name,
                base_addr=base_addr,
                element_size=element_size,
                shape=input_shape,
                strides=make_contiguous_strides(input_shape, last_dim_contiguous=True),
            )
            input_data_volume = math.prod(input_shape) * element_size
            base_addr += math.ceil(input_data_volume / dram_row_size) * dram_row_size

            output_shape = [sum(operator.input_length), operator.kv_head_num * operator.kv_group_num * operator.v_head_dim]
            output_placement = make_tensor_placement(
                name="output_" + operator.name,
                base_addr=base_addr,
                element_size=element_size,
                shape=output_shape,
                strides=make_contiguous_strides(output_shape, last_dim_contiguous=True),
            )
            output_data_volume = math.prod(output_shape) * element_size
            base_addr += math.ceil(output_data_volume / dram_row_size) * dram_row_size

            max_context_length = cloud_config.max_context_length
            per_core_max_context_length = math.ceil(max_context_length / cloud_config.chip_config.core_num)
            kv_length = operator.head_dim if operator.is_mla else operator.v_head_dim + operator.head_dim
            weight_shape = [operator.kv_head_num * per_core_max_context_length, kv_length]
            weight_placement = make_tensor_placement(
                name="kv_cache",
                base_addr=base_addr,
                element_size=element_size,
                shape=weight_shape,
                strides=make_contiguous_strides(weight_shape, last_dim_contiguous=True),
            )
            weight_data_volume = math.prod(weight_shape) * element_size
            base_addr += math.ceil(weight_data_volume / dram_row_size) * dram_row_size
            weight_placement_list.append(weight_placement)
        else:
            continue
        
        data_placement_list.extend([input_placement, output_placement, *weight_placement_list])
        data_placement_dict[operator.name] = {
            "input": input_placement,
            "output": output_placement,
            "weight": weight_placement_list,
        }

    return {
        "data_placement_list": data_placement_list,
        "data_placement_dict": data_placement_dict,
    }


def explore_cloud_tiling(
    # Operator-related configs
    model_config: ModelConfig,
    attention_block: List[Operator],
    ffn_moe_block: List[Operator],
    # Input-related configs
    context_length_list: List[int],
    core_flit_index_list: List[int],
    # Hardware-related configs
    cloud_config: CloudSystemConfig,
    # Some hyper parameters
    element_size: int = 2,
    dram_row_size: int = 128*1024,
    min_tM: int = 8,
    min_tK: int = 512,
    min_tN: int = 8,
    num_workers: int = int(0.8 * multiprocessing.cpu_count()),
    # Intermediate results storing configs
    intermediate_result_dir: str = "",
    # Shared GEMM tiling cache directory (reusable across context lengths)
    gemm_tiling_cache_dir: str = "",
):
    if intermediate_result_dir == "":
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        core_num = cloud_config.chip_config.core_num
        intermediate_result_dir = os.path.join(
            project_root,
            f"decaparated/cloud_tiling_exploration/{model_config.name}_{core_num}cores"
        )
    os.makedirs(intermediate_result_dir, exist_ok=True)

    topology, k, n = extract_noc_topology(cloud_config.chip_config.noc_config.config_path)
    if(topology != "mesh"):
        raise ValueError("Currently we only support mesh topology for cloud system")
    core_num = cloud_config.chip_config.core_num
    core_array_size = int(math.sqrt(core_num))
    assert core_array_size**2 == core_num,\
        f"Currently we assume core array is square, but got {core_array_size**2} != {core_num}"
    assert n == 2 and k == core_array_size, \
        f"Currently we assume core array is square, but got {n} != 2 or {k} != {core_array_size}"
    assert core_array_size % 2 == 0, \
        f"Currently we assume core array size is even, but got {core_array_size} % 2 != 0"

    # For computation, get operator shape on each core for detailed intra-core tiling exploration.
    # For inter-chip communication, set data transfer size according to input configs.
    parallel_config = cloud_config.parallel_config
    set_shape(
        model_config=model_config,
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
        context_length_list=context_length_list,
        parallel_config=parallel_config,
        core_num=core_num,
        core_array_size=core_array_size,
        block_size=cloud_config.block_size,
    )
    # Generate last block mapping
    core_list_by_l1 = sort_mesh_nodes_by_l1_distance(core_array_size)
    batch_size = len(context_length_list)
    last_block_mapping = [[[] for _ in range(core_array_size)] for _ in range(core_array_size)]
    core_idx = 0
    for i in range(batch_size):
        r, c = core_list_by_l1[core_idx]
        last_block_mapping[r][c].append(i)
        core_idx += 1
        if core_idx >= core_num:
            core_idx = 0

    # Generate all tensors' data placement descritpion required for all operators
    data_placement_result = generate_data_placement_list(
        attention_block=attention_block,
        ffn_moe_block=ffn_moe_block,
        cloud_config=cloud_config,
        element_size=element_size,
        dram_row_size=dram_row_size,
    )
    data_placement_list = data_placement_result["data_placement_list"]
    data_placement_dict = data_placement_result["data_placement_dict"]

    full_operator_list = attention_block + ffn_moe_block
    kv_cache_placement = None
    for placement in data_placement_list:
        if placement["name"] == "kv_cache":
            kv_cache_placement = placement
            break
    assert kv_cache_placement is not None, "KV cache placement not found"

    task_description_list = []
    inter_chip_communication_list = []
    attention_input_template = None
    # Process attention block
    for i, operator in enumerate(full_operator_list):
        if operator.op_type in [OperatorType.ALLREDUCE, OperatorType.ALL2ALL]:
            inter_chip_communication_list.append(operator)
            continue
        print(f"Processing operator {operator.name}")

        input_placement = data_placement_dict[operator.name]["input"]
        output_placement = data_placement_dict[operator.name]["output"]
        weight_placement_list = data_placement_dict[operator.name]["weight"]

        offset = 1
        next_operator = full_operator_list[(i+offset) % len(full_operator_list)]
        while next_operator.op_type in [OperatorType.ALL2ALL, OperatorType.ALLREDUCE]:
            offset += 1
            next_operator = full_operator_list[(i+offset) % len(full_operator_list)]
        next_input_placement = data_placement_dict[next_operator.name]["input"]

        if operator.op_type == OperatorType.GEMM:
            assert input_placement["name"] == "input_" + operator.name
            assert output_placement["name"] == "output_" + operator.name
            assert all(weight_placement["name"] == "weight_" + operator.name + f"_b{i}" for i, weight_placement in enumerate(weight_placement_list))

            # First explore computation tiling
            # comp_description_list, tiling_factors = [None], [None]
            comp_result = explore_gemm_comp(
                # Tensor description
                input_placement=input_placement,
                output_placement=output_placement,
                weight_placement_list=weight_placement_list,
                # Operator shape description
                operator=operator,
                element_size=element_size,
                # Chip config
                chip_config=cloud_config.chip_config,
                chip_config_path=cloud_config.chip_config_path,
                # Hyper parameters
                min_tM=min_tM,
                min_tK=min_tK,
                min_tN=min_tN,
                num_workers=num_workers,
                intermediate_result_dir=intermediate_result_dir,
                gemm_tiling_cache_dir=gemm_tiling_cache_dir,
            )
            latency = comp_result["latency"]
            comp_description_list = comp_result["comp_description_list"]
            tiling_factors = comp_result["tiling_factors"]

            # Then generate inter-core communication description for each computation description
            for comp_description in comp_description_list:
                task_description_list.append(("computation", comp_description, tiling_factors, latency))

                comm_description_list = []
                # Non-deepseek models
                if operator.name == "qkv_proj":
                    q_head_num = math.ceil(model_config.n_head / parallel_config.tp_size) 
                    kv_head_num = math.ceil(model_config.n_kv_head / parallel_config.tp_size)
                    q_length = int(operator.N * q_head_num / (q_head_num + kv_head_num*2))
                    kv_length = operator.N - q_length
                    
                    comm_description_list.append(generate_q_proj_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        q_length=q_length,
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                    comm_description_list.append(generate_kv_proj_comm(
                        # Tensor description
                        input_placement=output_placement,
                        kv_cache_placement=kv_cache_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        q_length=q_length,
                        kv_length=kv_length,
                        total_slot_num=math.ceil(tensor_dim(kv_cache_placement, 0, expected_rank=2)/kv_head_num),
                        num_kv_heads=kv_head_num,
                        element_size=element_size,
                        op_num_per_element=1,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Online serving parameters
                        last_block_mapping=last_block_mapping,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                # Deepseek models
                elif operator.name == "dq_kr_dkv_proj":
                    kv_head_num = 1
                    q_length = model_config.q_lora_rank // parallel_config.tp_size
                    kv_length = (model_config.kv_lora_rank + model_config.qk_rope_head_dim) // core_array_size

                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=q_length,
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                    comm_description_list.append(generate_kv_proj_comm(
                        # Tensor description
                        input_placement=output_placement,
                        kv_cache_placement=kv_cache_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        q_length=q_length,
                        kv_length=kv_length,
                        total_slot_num=math.ceil(tensor_dim(kv_cache_placement, 0, expected_rank=2)/kv_head_num),
                        num_kv_heads=kv_head_num,
                        element_size=element_size,
                        op_num_per_element=1,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Online serving parameters
                        last_block_mapping=last_block_mapping,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                elif operator.name == "uq_qr_proj":
                    comm_column_length = tensor_dim(output_placement, 1, expected_rank=2)
                    # When TP size is 1, non-rope part conducts 1d-all-reduce,
                    # while rope part conducts q-proj-comm
                    if parallel_config.tp_size == 1:
                        comm_column_length = model_config.n_head * model_config.qk_nope_head_dim
                    assert comm_column_length <= tensor_dim(output_placement, 1, expected_rank=2), \
                        f"Communication column length {comm_column_length} is larger than output column size {tensor_dim(output_placement, 1, expected_rank=2)}"

                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        # There is a inter-chip reduce scatter between uq_ur_proj and uk_proj
                        output_placement=output_placement, 
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=comm_column_length,
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=1,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                elif operator.name == "uk_proj":
                    comm_description_list.append(generate_q_proj_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        q_length=tensor_dim(output_placement, 1, expected_rank=2),
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))

                    # When TP size > 1, this can be merged by inter-chip all reduce after uq_qr_proj
                    if parallel_config.tp_size == 1:
                        uq_ur_output = data_placement_dict["uq_ur_proj"]["output"]
                        uq_ur_q_head_num = math.ceil(model_config.n_head / parallel_config.tp_size) 
                        uq_ur_column_offset = model_config.n_head * model_config.qk_nope_head_dim
                        uq_ur_q_length = uq_ur_q_head_num * model_config.qk_rope_head_dim
                        
                        comm_description_list.append(generate_q_proj_comm(
                            # Tensor description
                            input_placement=uq_ur_output,
                            output_placement=next_input_placement,
                            # Operator shape description
                            name=operator.name,
                            row_num=operator.M,
                            input_column_offset=uq_ur_column_offset,
                            q_length=uq_ur_q_length,
                            element_size=element_size,
                            op_num_per_element=1,
                            reduce_direction=1,
                            # Chip config
                            chip_config_path=cloud_config.chip_config_path,
                            core_num=core_num,
                            core_array_size=core_array_size,
                            flit_size=cloud_config.chip_config.noc_config.flit_size,
                            core_flit_index_list=core_flit_index_list,
                            # Intermediate results storing configs
                            intermediate_result_dir=intermediate_result_dir,
                        ))
                elif operator.name == "uv_proj":
                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=tensor_dim(output_placement, 1, expected_rank=2),
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                # Common blocks, but need to specify reduce directions
                elif operator.name == "o_proj":
                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=tensor_dim(output_placement, 1, expected_rank=2),
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0 if not model_config.is_deepseek else 1,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                elif operator.name == "down_proj":
                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=tensor_dim(output_placement, 1, expected_rank=2),
                        element_size=element_size,
                        # Gated nonlinearity executes after reduce-scatter, once
                        # per output element, before gathering the smaller result.
                        op_num_per_element=1 if model_config.mlp_gated_linear_units else 2,
                        gated=model_config.mlp_gated_linear_units,
                        reduce_direction=1 if not model_config.is_deepseek else 0,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                elif operator.name == "up_proj":
                    comm_description_list.append(generate_1d_allreduce_comm(
                        # Tensor description
                        input_placement=output_placement,
                        output_placement=next_input_placement,
                        # Operator shape description
                        name=operator.name,
                        row_num=operator.M,
                        input_column_offset=0,
                        input_column_length=tensor_dim(output_placement, 1, expected_rank=2),
                        element_size=element_size,
                        op_num_per_element=1,
                        reduce_direction=0 if not model_config.is_deepseek else 1,
                        # Chip config
                        chip_config_path=cloud_config.chip_config_path,
                        core_num=core_num,
                        core_array_size=core_array_size,
                        flit_size=cloud_config.chip_config.noc_config.flit_size,
                        core_flit_index_list=core_flit_index_list,
                        # Intermediate results storing configs
                        intermediate_result_dir=intermediate_result_dir,
                    ))
                task_description_list.extend(comm_description_list)
        elif operator.op_type == OperatorType.ATTENTION:
            assert input_placement["name"] == "input_" + operator.name
            assert output_placement["name"] == "output_" + operator.name
            assert len(weight_placement_list) == 1
            assert weight_placement_list[0]["name"] == "kv_cache"

            attention_result = generate_attention_comp(
                input_placement=input_placement,
                output_placement=output_placement,
                weight_placement_list=weight_placement_list,
                operator=operator,
                element_size=element_size,
                context_length_list=[[operator.context_length for _ in range(core_array_size)] for _ in range(core_array_size)],
                cloud_config=cloud_config,
            )
            attention_comp_description = attention_result["attention_comp_description"]
            attention_input_template = attention_result["attention_input_template"]
            task_description_list.append(("computation", attention_comp_description))

            # Then generate inter-core communication description for each computation description
            attention_comm_description = generate_attention_comm(
                input_placement=output_placement,
                output_placement=next_input_placement,
                # Operator shape description
                row_num=batch_size,
                element_size=element_size,
                op_num_per_element=3,
                # Chip config
                chip_config_path=cloud_config.chip_config_path,
                core_num=core_num,
                core_array_size=core_array_size,
                flit_size=cloud_config.chip_config.noc_config.flit_size,
                core_flit_index_list=core_flit_index_list,
                # Intermediate results storing configs
                intermediate_result_dir=intermediate_result_dir,
            )
            task_description_list.append(attention_comm_description)

        if i == len(attention_block) - 1 or i == len(attention_block) + len(ffn_moe_block) - 1:
            pass

    return {
        "data_placement_list": data_placement_list,
        "task_description_list": task_description_list,
        "attention_input_template": attention_input_template,
        "inter_chip_communication_list": inter_chip_communication_list,
    }
