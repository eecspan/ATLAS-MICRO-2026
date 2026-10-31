#ifndef _ATLASIM_CORE_H
#define _ATLASIM_CORE_H

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>
#include "common/arch.h"
#include "common/task.h"
#include "common/data.h"
#include "common/stats.h"
#include "common/packet.h"
#include "core/component.h"


namespace atlasim {


class Core {
public:
    Core(): core_id(-1), core_name("Core") { }
    Core(
        // Basic settings
        int _core_id, ArchConfig _arch_config, OperatorList _operator_list, PlacementMap _placement_map,
        // Booksim settings
        int _threshold,
        // Interface classes between Core and NoC wrapper
        CNInterface _send_queue, CNInterface _received_queue, std::shared_ptr<std::vector<bool>> _pipe_open
    );

    void reset_execution_status();
    bool is_finished() { return is_all_task_finished; };
    void tick();

    const int get_core_id() const { return core_id; }
    const double get_max_dram_bw() const { return max_dram_bw; }

    const Stats& get_e2e_stats() const { return e2e_stats; }
    const Stats& get_cur_op_stats() const { return cur_task_stats; }
    const OperatorStatsCollection& get_all_op_stats() const { return all_task_stats; }
    void display_stats(std::ostream& os = std::cout);

private:
    // Basic settings
    int core_id;
    std::string core_name;
    ArchConfig arch_config;
    OperatorList operator_list;
    PlacementMap placement_map;

    // Architecture components
    ComponentCollection components;
    std::string matrix_name = ctype2str.at(ComponentType::MATRIX);
    std::string vector_name = ctype2str.at(ComponentType::VECTOR);
    std::string buffer_name = ctype2str.at(ComponentType::BUFFER);
    std::string dram_name = ctype2str.at(ComponentType::DRAM);
    std::string noc_name = ctype2str.at(ComponentType::NOC);
    // Auxiliary components for NoC
    CNInterface send_queue;
    CNInterface received_queue;
    std::shared_ptr<std::vector<Packet>> send_buffer;
    std::shared_ptr<std::vector<Packet>> received_buffer;
    
    // Simulation records
    uint64_t clk;
    double max_dram_bw;
    Stats e2e_stats;
    Stats cur_task_stats;
    OperatorStatsCollection all_task_stats;

    // Runtime status of task execution
    bool is_all_task_finished;
    int cur_task_id; // Current task's index in OperatorList
    bool cur_task_simulated;
    int cur_task_iteration_count; // Current task's total iteration count 
    int cur_iteration; // Current task's ongoing execution iteration
    int cur_iteration_cycles;
    int cur_iteration_elapsed_cycles; // Elapsed cycle count of current iteration
    // Used for storing pre-simulated latency for non-NoC components
    // Key: Matrix, Vector, Buffer, DRAM, Total
    std::unordered_map<std::string, std::vector<uint64_t>> pre_estimated_cycles_per_iter;
    std::unordered_map<std::string, std::vector<double>> pre_estimated_op_count_per_iter;
    // Used for storing pre-generated NoC packets in each iteration
    std::vector<std::pair<std::vector<Packet>, std::vector<Packet>>> pre_generated_noc_packets;

    // General operators use the same Ramulator batch service times as before,
    // on one FIFO DRAM resource per core. Only annotated reads may survive an
    // iteration boundary; consumers and the operator epilogue wait for them.
    bool scheduled_dram = false;
    uint64_t dram_busy_until = 0;
    std::vector<int> dram_deadline;
    std::vector<uint64_t> dram_ready_before;
    void issue_dram_iteration();

    void pre_simulate(std::shared_ptr<OperatorTask> task);
    void update_cur_task_energy();

public:
    // Used for deadlock checking, so we only compare the runtime status of the core
    bool is_equal_to(const Core& other) const {
        return core_id == other.core_id &&
               is_all_task_finished == other.is_all_task_finished &&
               cur_task_id == other.cur_task_id &&
               cur_task_iteration_count == other.cur_task_iteration_count &&
               cur_iteration == other.cur_iteration &&
               cur_iteration_cycles == other.cur_iteration_cycles &&
               cur_iteration_elapsed_cycles == other.cur_iteration_elapsed_cycles;
    }
};


}


#endif // _ATLASIM_CORE_H
