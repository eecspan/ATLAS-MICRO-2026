#include <algorithm>
#include <numeric>
#include <stdexcept>
// atlasim include
#include "core/core.h"
#include "common/input.h"
#include "common/task.h"
#include "matrix_unit.h"
#include "vector_unit.h"
#include "buffer.h"
#include "dram/dram_wrapper.h"
#include "noc_interface.h"


namespace atlasim {

namespace {

ComponentInputItem make_general_dram_item(
    const TensorPlacement& placement,
    const std::vector<int64_t>& access_base,
    bool is_write,
    const std::vector<int64_t>& access_extent
) {
    if(access_base.size() != access_extent.size()) {
        std::cerr << "General DRAM access base/extents rank mismatch." << std::endl;
        exit(-1);
    }
    ComponentInputItem item = make_component_input_item({
        {"base_addr", placement.base_addr},
        {"element_size", placement.element_size},
        {"is_write", is_write},
        {"layout_rank", placement.rank()},
    });
    for(int dim = 0; dim < placement.rank(); ++dim) {
        item["shape_" + std::to_string(dim)] = placement.shape[dim];
        item["stride_" + std::to_string(dim)] = placement.strides[dim];
    }
    for(size_t dim = 0; dim < access_base.size(); ++dim) {
        item["access_base_" + std::to_string(dim)] = access_base[dim];
        item["access_extent_" + std::to_string(dim)] = access_extent[dim];
    }
    return item;
}

void append_dram_access_item(
    ComponentInput input,
    const TensorPlacement& placement,
    bool is_write,
    const std::vector<int64_t>& access_base,
    const std::vector<int64_t>& access_extent
) {
    if(access_base.empty() || access_base.size() != access_extent.size()) {
        std::cerr << "General DRAM path requires non-empty access base/extents with matching rank." << std::endl;
        exit(-1);
    }
    if(access_base.size() != static_cast<size_t>(placement.rank())) {
        std::cerr << "General DRAM access rank " << access_base.size()
                  << " does not match tensor placement rank " << placement.rank() << "." << std::endl;
        exit(-1);
    }
    for(size_t dim = 0; dim < access_base.size(); ++dim) {
        if(access_base[dim] < 0 || access_extent[dim] < 0) {
            std::cerr << "General DRAM access base/extents must be non-negative." << std::endl;
            exit(-1);
        }
    }
    input->push_back(make_general_dram_item(placement, access_base, is_write, access_extent));
}

void require_decode_attention_rank2_dim1_contiguous(
    const TensorPlacement& placement,
    const std::string& tensor_role
) {
    bool valid = placement.rank() == 2 &&
                 placement.shape.size() == 2 &&
                 placement.strides.size() == 2 &&
                 placement.strides[1] == 1 &&
                 placement.strides[0] == placement.shape[1];
    if(!valid) {
        std::cerr << "DecodeAttention " << tensor_role
                  << " tensor must use the legacy rank-2 dim1-contiguous layout, got shape=[";
        for(size_t i = 0; i < placement.shape.size(); ++i) {
            if(i > 0) std::cerr << ", ";
            std::cerr << placement.shape[i];
        }
        std::cerr << "], strides=[";
        for(size_t i = 0; i < placement.strides.size(); ++i) {
            if(i > 0) std::cerr << ", ";
            std::cerr << placement.strides[i];
        }
        std::cerr << "]." << std::endl;
        exit(-1);
    }
}

Packet make_unicast_packet(const NoCTask& task) {
    Packet p;
    p.fid = task.init_flit_id;
    p.size = task.flit_num;
    p.type = Packet::TransferType::_UNICAST;
    p.path = std::make_shared<MCTree>(task.src);
    p.path->add_segment(task.src, task.dst, nullptr, true);
    return p;
}

}


Core::Core(
    int _core_id, ArchConfig _arch_config, OperatorList _operator_list, PlacementMap _placement_map,
    int _threshold,
    CNInterface _send_queue, CNInterface _received_queue, std::shared_ptr<std::vector<bool>> _pipe_open
)
    : core_id(_core_id)
    , core_name(std::string("Core_") + std::to_string(_core_id))
    , arch_config(_arch_config)
    , operator_list(_operator_list)
    , placement_map(_placement_map) {

    // Init clocks
    clk = 0;

    // Init task simulation status
    is_all_task_finished = false;
    cur_task_id = 0;
    cur_task_iteration_count = 0;
    cur_iteration = 0;
    cur_iteration_cycles = 0;
    cur_iteration_elapsed_cycles = 0;

    // Init hardware components and pre-simulation records
    send_queue = _send_queue;
    received_queue = _received_queue;
    send_buffer = std::make_shared<std::vector<Packet>>();
    received_buffer = std::make_shared<std::vector<Packet>>();
    for(auto p: ctype2str) {
        ComponentType type = p.first;
        std::string name = p.second;
        switch (type) {
            case ComponentType::MATRIX:
                components[name] = std::make_shared<MatrixUnit>(core_name, arch_config);
                pre_estimated_cycles_per_iter[name] = {};
                pre_estimated_op_count_per_iter[name] = {};
                break;
            case ComponentType::VECTOR:
                components[name] = std::make_shared<VectorUnit>(core_name, arch_config);
                pre_estimated_cycles_per_iter[name] = {};
                pre_estimated_op_count_per_iter[name] = {};
                break;
            case ComponentType::BUFFER:
                components[name] = std::make_shared<Buffer>(core_name, arch_config);
                pre_estimated_cycles_per_iter[name] = {};
                pre_estimated_op_count_per_iter[name] = {};
                break;
            case ComponentType::DRAM:
                components[name] = std::make_shared<DRAMWrapper>(core_name, arch_config);
                max_dram_bw = std::dynamic_pointer_cast<DRAMWrapper>(components[name])->get_max_bw();
                pre_estimated_cycles_per_iter[name] = {};
                pre_estimated_op_count_per_iter[name] = {};
                break;
            case ComponentType::NOC:
                components[name] = std::make_shared<NoCInterface>(
                    core_name, arch_config,
                    _threshold,
                    _core_id,
                    _send_queue, _received_queue, 
                    send_buffer, received_buffer,
                    _pipe_open
                );
                // Since NoC's simulation depends on runtime status, which cannot be pre-estimated,
                // we do not introduce pre-simulate NoC components.
                break;
            default:
                std::cerr << "[ERROR] Unsupported component type: " << name << "." << std::endl;
        }
    }
    pre_estimated_cycles_per_iter["Total"] = {};
}


void Core::reset_execution_status() {
    e2e_stats = Stats();
    cur_task_stats = Stats();
    all_task_stats.clear();

    is_all_task_finished = false;
    cur_task_id = 0;
    cur_task_simulated = false;
    cur_task_iteration_count = 0;
    cur_iteration = 0;
    cur_iteration_cycles = 0;
    cur_iteration_elapsed_cycles = 0;

    for(auto& p: pre_estimated_cycles_per_iter) {
        p.second.clear();
    }
    for(auto& p: pre_estimated_op_count_per_iter) {
        p.second.clear();
    }
    pre_generated_noc_packets.clear();
    
    for(auto p: components) {
        p.second->reset_execution_status(true);
    }
}


void Core::pre_simulate(std::shared_ptr<OperatorTask> task) {
    for(auto p: components) {
        p.second->reset_execution_status(false);
    }
    for(auto& p: pre_estimated_cycles_per_iter) {
        p.second.clear();
    }
    for(auto& p: pre_estimated_op_count_per_iter) {
        p.second.clear();
    }
    pre_generated_noc_packets.clear();

    std::vector<uint64_t> on_chip_pipeline_iters;

    if(task->type == OperatorType::GEMM) {
        cur_task_iteration_count = task->iteration + 2;
        std::shared_ptr<ComputationTask> computation_task = std::dynamic_pointer_cast<ComputationTask>(task);

        //// Non-DRAM components' latency can be directly derived from task input
        // (1) Matrix
        ComponentInput matrix_input = make_component_input({});
        for(auto m: computation_task->matrix_tasks) {
            matrix_input->emplace_back(make_component_input_item({
                {"mac_count", m.mac_count},
            }));
        }
        int matrix_cycle_per_iter = components[matrix_name]->simulate(matrix_input);
        double matrix_op_count_per_iter = components[matrix_name]->get_cur_op_count();

        // (2) Vector
        ComponentInput vector_input = make_component_input({});
        for(auto v: computation_task->vector_tasks) {
            vector_input->emplace_back(make_component_input_item({
                {"vec_count", v.vec_count}
            }));
        }
        int vector_cycle_per_iter = components[vector_name]->simulate(vector_input);
        double vector_op_count_per_iter = components[vector_name]->get_cur_op_count();

        // (3) Buffer
        ComponentInput buffer_input = make_component_input({
            {{"read", 0}},
            {{"write", 0}},
        });
        ComponentInputItem& read_item = buffer_input->at(0);
        ComponentInputItem& write_item = buffer_input->at(1);
        for(auto bl: computation_task->buffer_load_tasks) {
            read_item["read"] += bl.byte_count;
        }
        for(auto bs: computation_task->buffer_store_tasks) {
            write_item["write"] += bs.byte_count;
        }
        int buffer_cycle_per_iter = components[buffer_name]->simulate(buffer_input);
        double buffer_op_count_per_iter = components[buffer_name]->get_cur_op_count();

        // (4) Generate each iter's DRAM latency
        // First generate each iteration's DRAM access task description
        std::vector<ComponentInput> dram_inputs;
        for(int i=0; i<cur_task_iteration_count; ++i) {
            dram_inputs.push_back(make_component_input({}));
        }
        for(auto d: computation_task->DRAM_tasks) {
            // For each tensor to access
            // (i) Extract basic information
            const TensorPlacement& placement = placement_map->at(d.name);

            // (ii) Generate ComponentInputItem according to GEMM's tiling information
            int access_count = 0;
            std::vector<int64_t> cur_access_base = d.access_base;
            for(int i=d.init_iter; i<cur_task_iteration_count; i+=d.stride_iter) {
                append_dram_access_item(
                    dram_inputs[i],
                    placement,
                    d.is_write,
                    cur_access_base,
                    d.access_extent
                );

                access_count++;
                if(access_count >= d.total_iter)
                    break;
                for(size_t dim = 0; dim < cur_access_base.size(); ++dim) {
                    if(d.access_stride_add[dim] > 0 && access_count % d.access_stride_add[dim] == 0) {
                        cur_access_base[dim] = (cur_access_base[dim] + d.access_offset_add[dim]) % placement.shape[dim];
                    }
                }
            }
        }
        // Then get each iteration's DRAM access latency
        for(int i=0; i<cur_task_iteration_count; ++i) {
            components[dram_name]->reset_op_count();
            int cur_iter_cycles = components[dram_name]->simulate(dram_inputs[i]);

            pre_estimated_cycles_per_iter[dram_name].push_back(cur_iter_cycles);
            pre_estimated_op_count_per_iter[dram_name].push_back(components[dram_name]->get_cur_op_count());
        }

        //// Generate each iter's latency
        for(int i=0; i<cur_task_iteration_count; ++i) {
            // For the first/last iter , only process DRAM components
            if(i==0 || i==cur_task_iteration_count-1) {
                pre_estimated_cycles_per_iter[matrix_name].push_back(0);
                pre_estimated_cycles_per_iter[vector_name].push_back(0);
                pre_estimated_cycles_per_iter[buffer_name].push_back(0);

                pre_estimated_op_count_per_iter[matrix_name].push_back(0);
                pre_estimated_op_count_per_iter[vector_name].push_back(0);
                pre_estimated_op_count_per_iter[buffer_name].push_back(0);
            }
            // For intermediate iter, process both non-DRAM and DRAM components
            else {
                pre_estimated_cycles_per_iter[matrix_name].push_back(matrix_cycle_per_iter);
                pre_estimated_cycles_per_iter[vector_name].push_back(vector_cycle_per_iter);
                pre_estimated_cycles_per_iter[buffer_name].push_back(buffer_cycle_per_iter);

                pre_estimated_op_count_per_iter[matrix_name].push_back(matrix_op_count_per_iter);
                pre_estimated_op_count_per_iter[vector_name].push_back(vector_op_count_per_iter);
                pre_estimated_op_count_per_iter[buffer_name].push_back(buffer_op_count_per_iter);
            }

            pre_estimated_cycles_per_iter["Total"].push_back(std::max(
                std::max(
                    pre_estimated_cycles_per_iter[matrix_name][i],
                    pre_estimated_cycles_per_iter[vector_name][i]
                ),
                std::max(
                    pre_estimated_cycles_per_iter[buffer_name][i],
                    pre_estimated_cycles_per_iter[dram_name][i]
                )
            ));
        }
    }
    else if(task->type == OperatorType::DecodeAttention) {
        std::shared_ptr<ComputationTask> computation_task = std::dynamic_pointer_cast<ComputationTask>(task);
        if(!computation_task || !computation_task->decode_attn_input) {
            std::cerr << "DecodeAttention task must have a decoded attention input." << std::endl;
            exit(-1);
        }
        std::shared_ptr<DAttnInput> decode_attn_input = computation_task->decode_attn_input;
        if(core_id >= static_cast<int>(decode_attn_input->core_input_list.size())) {
            std::cerr << "DecodeAttention input core_input_list has "
                      << decode_attn_input->core_input_list.size()
                      << " entries, but core " << core_id << " is requested." << std::endl;
            exit(-1);
        }
        DAttnCoreInput& cur_core_input = decode_attn_input->core_input_list[core_id];
        
        int single_head_iteration_count = std::accumulate(
            cur_core_input.request_tile_count.begin(), 
            cur_core_input.request_tile_count.end(), 
            0
        );
        int total_tiles = decode_attn_input->kv_head_num * single_head_iteration_count;

        //// Simulate each sub-operation individually

        // (1) QK matrix
        components[matrix_name]->reset_op_count();
        ComponentInput qk_matrix_input = make_component_input({});
        qk_matrix_input->emplace_back(make_component_input_item({
            {"mac_count", computation_task->matrix_tasks[0].mac_count},
        }));
        int qk_matrix_cycles = components[matrix_name]->simulate(qk_matrix_input);
        double qk_matrix_op_count = components[matrix_name]->get_cur_op_count();

        // (2) SV matrix
        components[matrix_name]->reset_op_count();
        ComponentInput sv_matrix_input = make_component_input({});
        sv_matrix_input->emplace_back(make_component_input_item({
            {"mac_count", computation_task->matrix_tasks[1].mac_count},
        }));
        int sv_matrix_cycles = components[matrix_name]->simulate(sv_matrix_input);
        double sv_matrix_op_count = components[matrix_name]->get_cur_op_count();

        // (3) Softmax vector
        components[vector_name]->reset_op_count();
        ComponentInput softmax_vector_input = make_component_input({});
        softmax_vector_input->emplace_back(make_component_input_item({
            {"vec_count", computation_task->vector_tasks[0].vec_count}
        }));
        int softmax_vector_cycles = components[vector_name]->simulate(softmax_vector_input);
        double softmax_vector_op_count = components[vector_name]->get_cur_op_count();

        // (4) Accumulation vector
        components[vector_name]->reset_op_count();
        ComponentInput accum_vector_input = make_component_input({});
        accum_vector_input->emplace_back(make_component_input_item({
            {"vec_count", computation_task->vector_tasks[1].vec_count}
        }));
        int accum_vector_cycles = components[vector_name]->simulate(accum_vector_input);
        double accum_vector_op_count = components[vector_name]->get_cur_op_count();

        // (5) Buffer per sub-operation
        auto sim_buf = [&](int idx, int& cyc, double& op) {
            components[buffer_name]->reset_op_count();
            ComponentInput br = make_component_input({{{"read", computation_task->buffer_load_tasks[idx].byte_count}}});
            int rc = components[buffer_name]->simulate(br);
            double rop = components[buffer_name]->get_cur_op_count();
            components[buffer_name]->reset_op_count();
            ComponentInput bw = make_component_input({{{"write", computation_task->buffer_store_tasks[idx].byte_count}}});
            int wc = components[buffer_name]->simulate(bw);
            // cyc = std::max(rc, wc);
            cyc = rc + wc;
            op = rop + components[buffer_name]->get_cur_op_count();
        };
        int qk_buf_cyc, sm_buf_cyc, sv_buf_cyc, ac_buf_cyc;
        double qk_buf_op, sm_buf_op, sv_buf_op, ac_buf_op;
        sim_buf(0, qk_buf_cyc, qk_buf_op);
        sim_buf(1, sm_buf_cyc, sm_buf_op);
        sim_buf(2, sv_buf_cyc, sv_buf_op);
        sim_buf(3, ac_buf_cyc, ac_buf_op);

        // Build pipeline schedule for overlapped sub-operation execution
        struct ComputeStage {
            int matrix_tile;
            bool matrix_is_qk;
            int vector_tile;
            bool vector_is_softmax;
        };
        std::vector<ComputeStage> compute_stages;

        if(total_tiles == 1) {
            compute_stages.push_back({0, true,  -1, false});
            compute_stages.push_back({-1, false, 0, true});
            compute_stages.push_back({0, false, -1, false});
            compute_stages.push_back({-1, false, 0, false});
        } else if(total_tiles >= 2) {
            int pending_accum = -1;
            for(int i = 0; i < total_tiles; i += 2) {
                if(i + 1 < total_tiles) {
                    int t0 = i, t1 = i + 1;
                    if(pending_accum < 0)
                        compute_stages.push_back({t0, true, -1, false});
                    else
                        compute_stages.push_back({t0, true, pending_accum, false});
                    compute_stages.push_back({t1, true,  t0, true});
                    compute_stages.push_back({t0, false, t1, true});
                    compute_stages.push_back({t1, false, t0, false});
                    pending_accum = t1;
                } else {
                    int t = i;
                    if(pending_accum < 0)
                        compute_stages.push_back({t, true, -1, false});
                    else
                        compute_stages.push_back({t, true, pending_accum, false});
                    compute_stages.push_back({-1, false, t, true});
                    compute_stages.push_back({t, false, -1, false});
                    pending_accum = t;
                }
            }
            if(pending_accum >= 0)
                compute_stages.push_back({-1, false, pending_accum, false});
        }

        int num_compute_stages = (int)compute_stages.size();
        std::vector<int> tile_qk_stage(total_tiles, -1);
        std::vector<int> tile_accum_stage(total_tiles, -1);
        for(int s = 0; s < num_compute_stages; s++) {
            if(compute_stages[s].matrix_tile >= 0 && compute_stages[s].matrix_is_qk)
                tile_qk_stage[compute_stages[s].matrix_tile] = s;
            if(compute_stages[s].vector_tile >= 0 && !compute_stages[s].vector_is_softmax)
                tile_accum_stage[compute_stages[s].vector_tile] = s;
        }

        // Total iterations: prologue (DRAM load) + compute stages + epilogue (DRAM store)
        cur_task_iteration_count = 2 + num_compute_stages;

        //// Generate DRAM inputs mapped to new pipeline iterations
        // DRAM for tile i is loaded at iteration tile_qk_stage[i] (one iter before tile's qk compute)
        // Output is stored at iteration tile_accum_stage[last_tile_of_head] + 2
        std::vector<ComponentInput> dram_inputs;
        for(int i = 0; i < cur_task_iteration_count; ++i) {
            dram_inputs.push_back(make_component_input({}));
        }
        TensorPlacement& input = placement_map->at(decode_attn_input->input_tensor_name);
        TensorPlacement& output = placement_map->at(decode_attn_input->output_tensor_name);
        TensorPlacement& kv_cache = placement_map->at(decode_attn_input->kv_cache_tensor_name);
        require_decode_attention_rank2_dim1_contiguous(input, "input");
        require_decode_attention_rank2_dim1_contiguous(output, "output");
        require_decode_attention_rank2_dim1_contiguous(kv_cache, "kv_cache");
        int num_requests = (int)cur_core_input.request_indices.size();

        // Q/output tensors retain global batch row IDs after redistribution.
        // A core owning requests {2, 7} must not access the prefix rows {0, 1}.
        // Coalesce adjacent rows to preserve the native contiguous-batch traffic.
        auto access_request_rows = [&](ComponentInput target, const TensorPlacement& tensor, bool write) {
            auto requests = cur_core_input.request_indices;
            std::sort(requests.begin(), requests.end());
            for(size_t start = 0; start < requests.size();) {
                size_t end = start + 1;
                while(end < requests.size() && requests[end] == requests[end - 1] + 1) ++end;
                if(requests[start] < 0 || requests[end - 1] >= tensor.shape[0] ||
                   (start > 0 && requests[start] == requests[start - 1])) {
                    throw std::invalid_argument("DecodeAttention request rows are invalid or duplicated");
                }
                std::vector<int64_t> base(tensor.rank(), 0), extent = tensor.shape;
                base[0] = requests[start];
                extent[0] = end - start;
                append_dram_access_item(target, tensor, write, base, extent);
                start = end;
            }
        };
        access_request_rows(dram_inputs[0], input, false);

        // (ii) Load KV vectors for each tile
        int flat_tile_idx = 0;
        for(int r=0; r<num_requests; ++r) {
            ReqTileInfo& cur_req_tile_info = cur_core_input.request_tile_info_list[r];
            int cur_req_tile_count = cur_core_input.request_tile_count[r];

            for(int h=0; h<decode_attn_input->kv_head_num; ++h) {
                for(int t=0; t<cur_req_tile_count; ++t) {
                    int kv_dram_iter = tile_qk_stage[flat_tile_idx];
                    std::vector<int>& cur_tile_token_list = cur_req_tile_info[t];
                    for(int i=0; i<(int)cur_tile_token_list.size(); ++i) {
                        int cur_token_offset = cur_tile_token_list[i];
                        std::vector<int64_t> kv_access_base(kv_cache.rank(), 0);
                        std::vector<int64_t> kv_access_extent = kv_cache.shape;
                        kv_access_base[0] = h * decode_attn_input->total_slot_num + cur_token_offset;
                        kv_access_extent[0] = 1;
                        append_dram_access_item(dram_inputs[kv_dram_iter], kv_cache, false, kv_access_base, kv_access_extent);
                    }
                    flat_tile_idx++;
                }
            }
        }

        // (iii) Store all output in the last stage
        int last_dram_iter = cur_task_iteration_count - 1;
        access_request_rows(dram_inputs[last_dram_iter], output, true);

        // Simulate DRAM for each iteration
        for(int i=0; i<cur_task_iteration_count; ++i) {
            components[dram_name]->reset_op_count();
            int cur_iter_cycles = components[dram_name]->simulate(dram_inputs[i]);
            double cur_iter_op_count = components[dram_name]->get_cur_op_count();

            pre_estimated_cycles_per_iter[dram_name].push_back(cur_iter_cycles);
            pre_estimated_op_count_per_iter[dram_name].push_back(cur_iter_op_count);
        }

        for(int i=0; i<cur_task_iteration_count; ++i) {
            int matrix_cyc = 0, vector_cyc = 0, buffer_cyc = 0;
            double matrix_op = 0, vector_op = 0, buffer_op = 0;
            int m_buf_cyc = 0, v_buf_cyc = 0;

            if(i > 0 && i <= num_compute_stages) {
                int si = i - 1;
                const ComputeStage& stage = compute_stages[si];

                if(stage.matrix_tile >= 0) {
                    if(stage.matrix_is_qk) {
                        matrix_cyc = qk_matrix_cycles;
                        matrix_op = qk_matrix_op_count;
                        m_buf_cyc = qk_buf_cyc;
                        buffer_cyc += m_buf_cyc;
                        buffer_op += qk_buf_op;
                    } else {
                        matrix_cyc = sv_matrix_cycles;
                        matrix_op = sv_matrix_op_count;
                        m_buf_cyc = sv_buf_cyc;
                        buffer_cyc += m_buf_cyc;
                        buffer_op += sv_buf_op;
                    }
                }

                if(stage.vector_tile >= 0) {
                    if(stage.vector_is_softmax) {
                        vector_cyc = softmax_vector_cycles;
                        vector_op = softmax_vector_op_count;
                        v_buf_cyc = sm_buf_cyc;
                        buffer_cyc += v_buf_cyc;
                        buffer_op += sm_buf_op;
                    } else {
                        vector_cyc = accum_vector_cycles;
                        vector_op = accum_vector_op_count;
                        v_buf_cyc = ac_buf_cyc;
                        buffer_cyc += v_buf_cyc;
                        buffer_op += ac_buf_op;
                    }
                }
            }

            pre_estimated_cycles_per_iter[matrix_name].push_back(matrix_cyc);
            pre_estimated_cycles_per_iter[vector_name].push_back(vector_cyc);
            pre_estimated_cycles_per_iter[buffer_name].push_back(buffer_cyc);

            pre_estimated_op_count_per_iter[matrix_name].push_back(matrix_op);
            pre_estimated_op_count_per_iter[vector_name].push_back(vector_op);
            pre_estimated_op_count_per_iter[buffer_name].push_back(buffer_op);

            uint64_t on_chip_pipeline_cyc = std::max(
                std::max((uint64_t)matrix_cyc, (uint64_t)vector_cyc),
                (uint64_t)buffer_cyc
            );
            on_chip_pipeline_iters.push_back(on_chip_pipeline_cyc);

            pre_estimated_cycles_per_iter["Total"].push_back(std::max(
                on_chip_pipeline_cyc,
                pre_estimated_cycles_per_iter[dram_name][i]
            ));
        }
    }
    else if(task->type == OperatorType::Communication) {
        cur_task_iteration_count = task->iteration + 2;
        for(auto& p: pre_estimated_cycles_per_iter) {
            p.second.assign(cur_task_iteration_count, 0);
        }
        for(auto& p: pre_estimated_op_count_per_iter) {
            p.second.assign(cur_task_iteration_count, 0);
        }
        pre_generated_noc_packets.assign(
            cur_task_iteration_count, 
            std::make_pair(std::vector<Packet>(), std::vector<Packet>())
        );

        // Extract current PE's NoC task description
        std::shared_ptr<CommunicationTask> communication_task = std::dynamic_pointer_cast<CommunicationTask>(task);
        CoreNoCTask& core_noc_task = communication_task->core_noc_tasks[core_id];

        // Generate each iter's non-DRAM actions
        for(int i=0; i<communication_task->iteration; ++i) {
            NoCTxTask& cur_tx_task = core_noc_task.per_iter_tx_tasks[i];
            NoCRxTask& cur_rx_task = core_noc_task.per_iter_rx_tasks[i];

            // (1) Vector in Rx Task
            components[vector_name]->reset_op_count();
            ComponentInput vector_input = make_component_input({});
            for(auto v: cur_rx_task.vector_tasks) {
                vector_input->emplace_back(make_component_input_item({
                    {"vec_count", v.vec_count}
                }));
            }
            int vector_cycle_per_iter = components[vector_name]->simulate(vector_input);
            double vector_op_count_per_iter = components[vector_name]->get_cur_op_count();
            // 0th iter is reserved for DRAM access
            pre_estimated_cycles_per_iter[vector_name][i+1] = vector_cycle_per_iter;
            pre_estimated_op_count_per_iter[vector_name][i+1] = vector_op_count_per_iter;

            // (2) Buffer in Tx/Rx Task (all buffer tasks are executed sequentially)
            // (2.1) Buffer in Tx Task
            ComponentInput tx_buffer_load_input = make_component_input({
                {{"read", 0}},
            });
            for(auto bl: cur_tx_task.buffer_load_tasks) {
                tx_buffer_load_input->at(0)["read"] += bl.byte_count;
            }
            components[buffer_name]->reset_op_count();
            int tx_buffer_cycle_per_iter = components[buffer_name]->simulate(tx_buffer_load_input);
            double tx_buffer_op_count_per_iter = components[buffer_name]->get_cur_op_count();
            pre_estimated_cycles_per_iter[buffer_name][i+1] += tx_buffer_cycle_per_iter;
            pre_estimated_op_count_per_iter[buffer_name][i+1] += tx_buffer_op_count_per_iter;
            // (2.2) Buffer in Rx Task
            // (2.2.1) Buffer load in Rx Task
            ComponentInput rx_buffer_load_input = make_component_input({
                {{"read", 0}},
            });
            for(auto bl: cur_rx_task.buffer_load_tasks) {
                rx_buffer_load_input->at(0)["read"] += bl.byte_count;
            }
            components[buffer_name]->reset_op_count();
            int rx_buffer_cycle_per_iter = components[buffer_name]->simulate(rx_buffer_load_input);
            double rx_buffer_op_count_per_iter = components[buffer_name]->get_cur_op_count();
            pre_estimated_cycles_per_iter[buffer_name][i+1] += rx_buffer_cycle_per_iter;
            pre_estimated_op_count_per_iter[buffer_name][i+1] += rx_buffer_op_count_per_iter;
            // (2.2.2) Buffer store in Rx Task
            ComponentInput rx_buffer_store_input = make_component_input({
                {{"write", 0}},
            });
            for(auto bs: cur_rx_task.buffer_store_tasks) {
                rx_buffer_store_input->at(0)["write"] += bs.byte_count;
            }
            components[buffer_name]->reset_op_count();
            int rx_buffer_store_cycle_per_iter = components[buffer_name]->simulate(rx_buffer_store_input);
            double rx_buffer_store_op_count_per_iter = components[buffer_name]->get_cur_op_count();
            pre_estimated_cycles_per_iter[buffer_name][i+1] += rx_buffer_store_cycle_per_iter;
            pre_estimated_op_count_per_iter[buffer_name][i+1] += rx_buffer_store_op_count_per_iter;

            // (3) NoC in Tx/Rx Task (send/recv packets)
            // (3.1) NoC in Tx Task (send packets)
            for(auto st: cur_tx_task.noc_tasks) {
                assert(st.is_send);
                assert(st.src == core_id);

                pre_generated_noc_packets[i+1].first.push_back(make_unicast_packet(st));
            }
            // (3.2) NoC in Rx Task (recv packets)
            for(auto rt: cur_rx_task.noc_tasks) {
                assert(!rt.is_send);
                assert(rt.dst == core_id);

                pre_generated_noc_packets[i+1].second.push_back(make_unicast_packet(rt));
            }
        }

        // Generate each iter's DRAM actions
        // First generate each iteration's DRAM access task description
        std::vector<ComponentInput> dram_inputs;
        for(int i=0; i<cur_task_iteration_count; ++i) {
            dram_inputs.push_back(make_component_input({}));
        }
        for(auto d: core_noc_task.DRAM_tasks) {
            // For each tensor to access
            // (i) Extract basic information
            const TensorPlacement& placement = placement_map->at(d.name);

            // (ii) Generate ComponentInputItem according to GEMM's tiling information
            int access_count = 0;
            std::vector<int64_t> cur_access_base = d.access_base;
            for(int i=d.init_iter; i<cur_task_iteration_count; i+=d.stride_iter) {
                append_dram_access_item(
                    dram_inputs[i],
                    placement,
                    d.is_write,
                    cur_access_base,
                    d.access_extent
                );

                access_count++;
                if(access_count >= d.total_iter)
                    break;
                for(size_t dim = 0; dim < cur_access_base.size(); ++dim) {
                    if(d.access_stride_add[dim] > 0 && access_count % d.access_stride_add[dim] == 0) {
                        cur_access_base[dim] = (cur_access_base[dim] + d.access_offset_add[dim]) % placement.shape[dim];
                    }
                }
            }
        }
        // Then get each iteration's DRAM access latency
        for(int i=0; i<cur_task_iteration_count; ++i) {
            components[dram_name]->reset_op_count();
            int cur_iter_cycles = components[dram_name]->simulate(dram_inputs[i]);
            double cur_iter_op_count = components[dram_name]->get_cur_op_count();

            pre_estimated_cycles_per_iter[dram_name][i] = cur_iter_cycles;
            pre_estimated_op_count_per_iter[dram_name][i] = cur_iter_op_count;
        }

        // Generate each iter's latency
        for(int i=0; i<cur_task_iteration_count; ++i) {
            // Since we have injected each iter's latency components,
            // we only need to generate "Total" statistics
            pre_estimated_cycles_per_iter["Total"][i] = std::max(
                std::max(
                    pre_estimated_cycles_per_iter[matrix_name][i],
                    pre_estimated_cycles_per_iter[vector_name][i]
                ),
                std::max(
                    pre_estimated_cycles_per_iter[buffer_name][i],
                    pre_estimated_cycles_per_iter[dram_name][i]
                )
            );
        }
    }
    else if(task->type == OperatorType::SPMDGeneral || task->type == OperatorType::MPMDGeneral) {
        cur_task_iteration_count = task->iteration + 2;
        for(auto& p: pre_estimated_cycles_per_iter) {
            p.second.assign(cur_task_iteration_count, 0);
        }
        for(auto& p: pre_estimated_op_count_per_iter) {
            p.second.assign(cur_task_iteration_count, 0);
        }
        pre_generated_noc_packets.assign(
            cur_task_iteration_count,
            std::make_pair(std::vector<Packet>(), std::vector<Packet>())
        );

        std::shared_ptr<GeneralTask> general_task = std::dynamic_pointer_cast<GeneralTask>(task);
        if(!general_task) {
            std::cerr << "General operator task has invalid task object." << std::endl;
            exit(-1);
        }
        const GeneralCoreTask& core_task = general_task->get_core_task(core_id);

        std::vector<ComponentInput> dram_inputs;
        for(int i=0; i<cur_task_iteration_count; ++i) {
            dram_inputs.push_back(make_component_input({}));
        }

        for(int raw_iter=0; raw_iter<task->iteration; ++raw_iter) {
            const GeneralIterationTask& iter_task = core_task.per_iter_tasks[raw_iter];
            int sim_iter = raw_iter + 1;

            components[matrix_name]->reset_op_count();
            ComponentInput matrix_input = make_component_input({});
            for(auto m: iter_task.matrix_tasks) {
                matrix_input->emplace_back(make_component_input_item({
                    {"mac_count", m.mac_count},
                }));
            }
            pre_estimated_cycles_per_iter[matrix_name][sim_iter] = components[matrix_name]->simulate(matrix_input);
            pre_estimated_op_count_per_iter[matrix_name][sim_iter] = components[matrix_name]->get_cur_op_count();

            components[vector_name]->reset_op_count();
            ComponentInput vector_input = make_component_input({});
            for(auto v: iter_task.vector_tasks) {
                vector_input->emplace_back(make_component_input_item({
                    {"vec_count", v.vec_count},
                }));
            }
            pre_estimated_cycles_per_iter[vector_name][sim_iter] = components[vector_name]->simulate(vector_input);
            pre_estimated_op_count_per_iter[vector_name][sim_iter] = components[vector_name]->get_cur_op_count();

            components[buffer_name]->reset_op_count();
            ComponentInput buffer_input = make_component_input({
                {{"read", 0}},
                {{"write", 0}},
            });
            for(auto bl: iter_task.buffer_load_tasks) {
                buffer_input->at(0)["read"] += bl.byte_count;
            }
            for(auto bs: iter_task.buffer_store_tasks) {
                buffer_input->at(1)["write"] += bs.byte_count;
            }
            pre_estimated_cycles_per_iter[buffer_name][sim_iter] = components[buffer_name]->simulate(buffer_input);
            pre_estimated_op_count_per_iter[buffer_name][sim_iter] = components[buffer_name]->get_cur_op_count();

            if(!arch_config->has_noc && (!iter_task.noc_tx_tasks.empty() || !iter_task.noc_rx_tasks.empty())) {
                std::cerr << "General operator contains NoC tasks, but architecture has no NoC." << std::endl;
                exit(-1);
            }
            for(auto tx: iter_task.noc_tx_tasks) {
                if(tx.src == core_id) {
                    pre_generated_noc_packets[sim_iter].first.push_back(make_unicast_packet(tx));
                }
            }
            for(auto rx: iter_task.noc_rx_tasks) {
                if(rx.dst == core_id) {
                    pre_generated_noc_packets[sim_iter].second.push_back(make_unicast_packet(rx));
                }
            }

            for(auto d: iter_task.DRAM_tasks) {
                int dram_iter = raw_iter + (d.is_write ? 2 : 0);
                const TensorPlacement& placement = placement_map->at(d.name);
                append_dram_access_item(
                    dram_inputs[dram_iter],
                    placement,
                    d.is_write,
                    d.access_base,
                    d.access_extent
                );
            }
        }

        for(int i=0; i<cur_task_iteration_count; ++i) {
            components[dram_name]->reset_op_count();
            int cur_iter_cycles = components[dram_name]->simulate(dram_inputs[i]);
            double cur_iter_op_count = components[dram_name]->get_cur_op_count();

            pre_estimated_cycles_per_iter[dram_name][i] = cur_iter_cycles;
            pre_estimated_op_count_per_iter[dram_name][i] = cur_iter_op_count;
        }

        for(int i=0; i<cur_task_iteration_count; ++i) {
            uint64_t on_chip_pipeline_cyc = std::max(
                std::max(
                    pre_estimated_cycles_per_iter[matrix_name][i],
                    pre_estimated_cycles_per_iter[vector_name][i]
                ),
                pre_estimated_cycles_per_iter[buffer_name][i]
            );
            on_chip_pipeline_iters.push_back(on_chip_pipeline_cyc);
            pre_estimated_cycles_per_iter["Total"][i] = std::max(
                on_chip_pipeline_cyc,
                pre_estimated_cycles_per_iter[dram_name][i]
            );
        }
    }
    else {
        std::cerr << "[ERROR] Unsupported operator type: " << optype2str.at(task->type) << std::endl;
    }

    // Compute matrix bubble cycles from pre-estimated data
    if(task->type == OperatorType::GEMM ||
       task->type == OperatorType::DecodeAttention ||
       task->type == OperatorType::SPMDGeneral ||
       task->type == OperatorType::MPMDGeneral) {
        uint64_t non_overlap = 0;
        uint64_t on_chip_bubble = 0;
        uint64_t dram_bubble = 0;
        for(int i = 0; i < cur_task_iteration_count; ++i) {
            uint64_t m = pre_estimated_cycles_per_iter[matrix_name][i];
            uint64_t v = pre_estimated_cycles_per_iter[vector_name][i];
            uint64_t b = pre_estimated_cycles_per_iter[buffer_name][i];
            uint64_t d = pre_estimated_cycles_per_iter[dram_name][i];
            non_overlap += (m > v) ? (m - v) : (v - m);
            uint64_t on_chip_max;
            if(!on_chip_pipeline_iters.empty()) {
                on_chip_max = on_chip_pipeline_iters[i];
            } else {
                on_chip_max = std::max(std::max(m, v), b);
            }
            if(on_chip_max > m)
                on_chip_bubble += on_chip_max - m;
            if(d > on_chip_max)
                dram_bubble += d - on_chip_max;
        }
        cur_task_stats.compute_non_overlap_cycles = non_overlap;
        cur_task_stats.matrix_bubble_on_chip_cycles = on_chip_bubble;
        cur_task_stats.matrix_bubble_dram_cycles = dram_bubble;
    }

    cur_task_simulated = true;
    cur_iteration = 0;
    cur_iteration_cycles = pre_estimated_cycles_per_iter["Total"][0];
    cur_iteration_elapsed_cycles = 0;
    // Insert 1st iter's NoC packets into send/receive buffers
    if(pre_generated_noc_packets.size() > cur_iteration) {
        send_buffer->insert(
            send_buffer->end(), 
            pre_generated_noc_packets[cur_iteration].first.begin(), 
            pre_generated_noc_packets[cur_iteration].first.end()
        );
        received_buffer->insert(
            received_buffer->end(), 
            pre_generated_noc_packets[cur_iteration].second.begin(), 
            pre_generated_noc_packets[cur_iteration].second.end()
        );
    }
}


void Core::update_cur_task_energy() {
    double elapsed_seconds = cur_task_stats.e2e_cycles / (arch_config->frequency * 1e6);
    cur_task_stats.controller_energy = elapsed_seconds * arch_config->controller_config.power;
    cur_task_stats.matrix_energy = components[matrix_name]->get_cur_energy();
    cur_task_stats.vector_energy = components[vector_name]->get_cur_energy();
    cur_task_stats.buffer_energy = components[buffer_name]->get_cur_energy();
    cur_task_stats.dram_energy = components[dram_name]->get_cur_energy();
    cur_task_stats.noc_energy = components[noc_name]->get_cur_energy();
    cur_task_stats.update_e2e_energy();
}


void Core::tick() {
    // Generate current task's per-iter latency in non-NoC components before its first iter begins
    if(!is_all_task_finished && !cur_task_simulated) {
        pre_simulate(operator_list->at(cur_task_id));
    }

    // Tick all components
    clk++;
    if(!is_all_task_finished) {
        cur_iteration_elapsed_cycles++;
        cur_task_stats.e2e_cycles++;

        // Judge whether all non-NoC components are finished
        bool non_NoC_finished = cur_iteration_elapsed_cycles >= cur_iteration_cycles;
        // NoC action is completed if all packets in send/receive buffers are processed
        bool NoC_finished =  send_buffer->empty() && received_buffer->empty();
        
        // Tick active non-NoC components
        for(const auto& p: pre_estimated_cycles_per_iter) {
            if(p.first == "Total")
                continue;
                
            if(cur_iteration_elapsed_cycles <= p.second[cur_iteration]) {
                components[p.first]->tick(true);
                if(p.first == matrix_name)
                    cur_task_stats.matrix_cycles++;
                else if(p.first == vector_name)
                    cur_task_stats.vector_cycles++;
                else if(p.first == buffer_name)
                    cur_task_stats.buffer_cycles++;
                else if(p.first == dram_name)
                    cur_task_stats.dram_cycles++;
            }
            else {
                components[p.first]->tick(false);
            }
        }
        
        // Tick NoC Interface
        components[noc_name]->tick(!NoC_finished);
        if(!NoC_finished)
            cur_task_stats.noc_cycles++;

        // If current iteration has passed all cycles, move on to the next cycle
        if(non_NoC_finished && NoC_finished) {
            cur_task_stats.flop_count += 2 * pre_estimated_op_count_per_iter[matrix_name][cur_iteration]
                                       + pre_estimated_op_count_per_iter[vector_name][cur_iteration];
            cur_task_stats.memory_access_bytes += pre_estimated_op_count_per_iter[dram_name][cur_iteration];

            cur_iteration++;
            cur_iteration_elapsed_cycles = 0;
            // Only update iteration state if there are remaining iterations
            if(cur_iteration < cur_task_iteration_count) {
                cur_iteration_cycles = pre_estimated_cycles_per_iter["Total"][cur_iteration];
                // Update NoC Buffers and states
                if(pre_generated_noc_packets.size() > cur_iteration) {
                    send_buffer->insert(
                        send_buffer->end(), 
                        pre_generated_noc_packets[cur_iteration].first.begin(), 
                        pre_generated_noc_packets[cur_iteration].first.end()
                    );
                    received_buffer->insert(
                        received_buffer->end(), 
                        pre_generated_noc_packets[cur_iteration].second.begin(), 
                        pre_generated_noc_packets[cur_iteration].second.end()
                    );
                }
            }
        }

        // If current task has passed all iterations, update statistics, and move on to the next task
        if(cur_iteration >= cur_task_iteration_count) {
            if(core_id==0) {
                std::cout << clk << " | Task " << operator_list->at(cur_task_id)->name << " is finished, ";
                std::cout << "spending " << cur_task_stats.e2e_cycles << " cycles, ";
                std::cout << "bandwidth utilization " << cur_task_stats.dram_bw(arch_config->frequency) / max_dram_bw * 100 << "%." << std::endl;
            }

            // Update statistics
            update_cur_task_energy();
            e2e_stats += cur_task_stats;
            all_task_stats.emplace_back(operator_list->at(cur_task_id)->name, cur_task_stats);
            cur_task_stats = Stats();

            cur_task_id++;
            cur_task_simulated = false;
            cur_task_iteration_count = 0;
            cur_iteration = 0;
        }

        // If all tasks are simulated, update status
        if(cur_task_id >= operator_list->size()) {
            if(core_id==0) {
                std::cout << clk << " | All tasks are finished." << std::endl;
            }
            is_all_task_finished = true;
        }
    }
    else {
        // Tick all components with busy = false
        for(auto c : components) {
            c.second->tick(false);
        }
    }
}


void Core::display_stats(std::ostream& os) {
    os << "Core " << core_id << ":" << std::endl;

    os << "E2E Simulation Statistics:" << std::endl;
    e2e_stats.display_stats(max_dram_bw, arch_config->frequency, os, 4);

    os << "Per Task Simulation Statistics:" << std::endl;
    for(int i=0; i<all_task_stats.size(); ++i) {
        os << "(" << i+1 << ") " << operator_list->at(i)->name << ":" << std::endl;
        all_task_stats[i].second.display_stats(max_dram_bw, arch_config->frequency, os, 4);
    }

    os << "Per Component Simulation Statistics:" << std::endl;
    int index = 0;
    for(auto c: components) {
        os << "(" << index+1 << ") " << c.first << std::endl;
        c.second->display_stats(os);
        index++;
    }

    os << std::endl;
}


}
