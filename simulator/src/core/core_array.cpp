#include "core/core_array.h"
#include "common/stats.h"
#include <algorithm>
#include <cassert>
#include <ostream>


namespace atlasim {

namespace {

void update_all_core_energy(Stats& stats, const ArchConfig& arch_config, int core_num) {
    double elapsed_seconds = stats.e2e_cycles / (arch_config->frequency * 1e6);
    stats.controller_energy = core_num * elapsed_seconds * arch_config->controller_config.power;
    stats.matrix_energy = core_num * elapsed_seconds * arch_config->matrix_config.power;
    stats.vector_energy = core_num * elapsed_seconds * arch_config->vector_config.power;
    stats.buffer_energy = core_num * elapsed_seconds * arch_config->buffer_config.power;
    stats.dram_energy = core_num * elapsed_seconds * arch_config->dram_config.power;
    stats.noc_energy = core_num * elapsed_seconds * (arch_config->has_noc ? arch_config->noc_config.power : 0.0);
    stats.update_e2e_energy();
}

}


CoreArray::CoreArray(
    const ArchConfig _arch_config, const OperatorList _operator_list, const PlacementMapCollection _placement_maps,
    int _threshold,
    PCNInterfaceSet _send_queues, PCNInterfaceSet _received_queues, std::shared_ptr<std::vector<bool>> _pipe_open
) : arch_config(_arch_config), operator_list(_operator_list), placement_maps(_placement_maps),
    send_queues(_send_queues), received_queues(_received_queues), pipe_open(_pipe_open) {
    
    // Init clocks
    prev_clk.clear();
    clk = 0;

    // Init core count
    core_num = arch_config->core_num;

    // Init cores
    for(int i=0; i<core_num; ++i) {
        cores.push_back(std::make_shared<Core>(
            i, arch_config, operator_list, placement_maps->at(i),
            _threshold,
            send_queues->at(i), received_queues->at(i), pipe_open
        ));
    }
}


void CoreArray::reset_execution_status() {
    prev_clk.emplace_back(clk);
    for(auto core: cores) {
        core->reset_execution_status();
    }
    // Clear history to ensure the first check in the new run doesn't match stale state
    last_cores_state.clear();
}


void CoreArray::tick() {
    clk++;
    for(auto core: cores) {
        core->tick();
    }
}


bool CoreArray::state_changed() {
    bool change = false;
    if(last_cores_state.size() != cores.size()) {
        change = true;
    }
    else {
        for(int i=0; i<cores.size(); ++i) {
            if(!last_cores_state[i]->is_equal_to(*cores[i])) {
                change = true;
                break;
            }
        }
    }

    last_cores_state.clear();
    for (const auto& core : cores) {
        last_cores_state.push_back(std::make_shared<Core>(*core));
    }

    return change;
}


bool CoreArray::all_core_finished() {
    bool all_finished = true;
    for(auto core: cores) {
        if(!core->is_finished()) {
            all_finished = false;
            break;
        }
    }
    return all_finished;
}


Stats CoreArray::get_e2e_stats() {
    Stats e2e_stats;
    for(auto core: cores) {
        e2e_stats.e2e_cycles = std::max(e2e_stats.e2e_cycles, core->get_e2e_stats().e2e_cycles);
        e2e_stats.matrix_cycles = std::max(e2e_stats.matrix_cycles, core->get_e2e_stats().matrix_cycles);
        e2e_stats.vector_cycles = std::max(e2e_stats.vector_cycles, core->get_e2e_stats().vector_cycles);
        e2e_stats.buffer_cycles = std::max(e2e_stats.buffer_cycles, core->get_e2e_stats().buffer_cycles);
        e2e_stats.dram_cycles = std::max(e2e_stats.dram_cycles, core->get_e2e_stats().dram_cycles);
        e2e_stats.noc_cycles = std::max(e2e_stats.noc_cycles, core->get_e2e_stats().noc_cycles);

        e2e_stats.compute_non_overlap_cycles = std::max(e2e_stats.compute_non_overlap_cycles, core->get_e2e_stats().compute_non_overlap_cycles);
        e2e_stats.matrix_bubble_on_chip_cycles = std::max(e2e_stats.matrix_bubble_on_chip_cycles, core->get_e2e_stats().matrix_bubble_on_chip_cycles);
        e2e_stats.matrix_bubble_dram_cycles = std::max(e2e_stats.matrix_bubble_dram_cycles, core->get_e2e_stats().matrix_bubble_dram_cycles);
        e2e_stats.matrix_bubble_noc_cycles = std::max(e2e_stats.matrix_bubble_noc_cycles, core->get_e2e_stats().matrix_bubble_noc_cycles);
        e2e_stats.matrix_bubble_dram_noc_cycles = std::max(e2e_stats.matrix_bubble_dram_noc_cycles, core->get_e2e_stats().matrix_bubble_dram_noc_cycles);

        e2e_stats.flop_count += core->get_e2e_stats().flop_count;
        e2e_stats.memory_access_bytes += core->get_e2e_stats().memory_access_bytes;
    }
    update_all_core_energy(e2e_stats, arch_config, core_num);
    return e2e_stats;
}


OperatorStatsCollection CoreArray::get_all_op_stats() {
    OperatorStatsCollection all_op_stats;
    for(auto p: cores[0]->get_all_op_stats()) {
        all_op_stats.emplace_back(p.first, Stats());
    }

    for(auto core: cores) {
        OperatorStatsCollection core_op_stats = core->get_all_op_stats();
        assert(all_op_stats.size() == core_op_stats.size());
        for(int i=0; i<all_op_stats.size(); ++i) {
            assert(all_op_stats[i].first == core_op_stats[i].first);

            Stats& cur_stats = all_op_stats[i].second;
            const Stats& core_stats = core_op_stats[i].second;
            cur_stats.e2e_cycles = std::max(cur_stats.e2e_cycles, core_stats.e2e_cycles);
            cur_stats.matrix_cycles = std::max(cur_stats.matrix_cycles, core_stats.matrix_cycles);
            cur_stats.vector_cycles = std::max(cur_stats.vector_cycles, core_stats.vector_cycles);
            cur_stats.buffer_cycles = std::max(cur_stats.buffer_cycles, core_stats.buffer_cycles);
            cur_stats.dram_cycles = std::max(cur_stats.dram_cycles, core_stats.dram_cycles);
            cur_stats.noc_cycles = std::max(cur_stats.noc_cycles, core_stats.noc_cycles);

            cur_stats.compute_non_overlap_cycles = std::max(cur_stats.compute_non_overlap_cycles, core_stats.compute_non_overlap_cycles);
            cur_stats.matrix_bubble_on_chip_cycles = std::max(cur_stats.matrix_bubble_on_chip_cycles, core_stats.matrix_bubble_on_chip_cycles);
            cur_stats.matrix_bubble_dram_cycles = std::max(cur_stats.matrix_bubble_dram_cycles, core_stats.matrix_bubble_dram_cycles);
            cur_stats.matrix_bubble_noc_cycles = std::max(cur_stats.matrix_bubble_noc_cycles, core_stats.matrix_bubble_noc_cycles);
            cur_stats.matrix_bubble_dram_noc_cycles = std::max(cur_stats.matrix_bubble_dram_noc_cycles, core_stats.matrix_bubble_dram_noc_cycles);

            cur_stats.flop_count += core_stats.flop_count;
            cur_stats.memory_access_bytes += core_stats.memory_access_bytes;
        }
    }
    for(auto& op_stats: all_op_stats) {
        update_all_core_energy(op_stats.second, arch_config, core_num);
    }
    return all_op_stats;
}


CoreStatsCollection CoreArray::get_core_stats() {
    CoreStatsCollection core_stats;
    for(auto core: cores) {
        core_stats.emplace_back(core->get_e2e_stats());
    }
    return core_stats;
}


std::vector<std::pair<std::string, CoreStatsCollection>> CoreArray::get_operator_stats_per_core() {
    std::vector<std::pair<std::string, CoreStatsCollection>> operator_stats_per_core;
    for(auto p: cores[0]->get_all_op_stats()) {
        operator_stats_per_core.emplace_back(p.first, CoreStatsCollection());
    }

    for(auto core: cores) {
        OperatorStatsCollection core_op_stats = core->get_all_op_stats();
        assert(operator_stats_per_core.size() == core_op_stats.size());
        for(int i=0; i<operator_stats_per_core.size(); ++i) {
            assert(operator_stats_per_core[i].first == core_op_stats[i].first);
            operator_stats_per_core[i].second.push_back(core_op_stats[i].second);
        }
    }
    return operator_stats_per_core;
}
Performance CoreArray::get_performance() {
    Performance performance;

    performance.core_bw = get_max_dram_bw_per_core();
    performance.chip_bw = get_max_dram_bw();
    performance.chip_frequency = arch_config->frequency;

    performance.e2e_stats = get_e2e_stats();
    performance.operator_stats = get_all_op_stats();
    performance.core_stats = get_core_stats();
    performance.operator_stats_per_core = get_operator_stats_per_core();

    return performance;
}


void CoreArray::display_stats(std::ostream& os) {
    os << "[Core Array Status]" << std::endl << std::endl;
    for(auto core: cores) {
        os << "[Core " << core->get_core_id() << "]" << std::endl;
        core->display_stats(os);
    }
}


}
