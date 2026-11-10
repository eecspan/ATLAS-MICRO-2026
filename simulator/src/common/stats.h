#ifndef _ATLASIM_STATS_H
#define _ATLASIM_STATS_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>


namespace atlasim {


#define _KB std::pow(2, 10)
#define _MB std::pow(2, 20)
#define _GB std::pow(2, 30)
#define _MHz 1e6
#define _GHz 1e9


struct Stats {
    // Latency statistics (the unit is cycle)
    uint64_t e2e_cycles = 0;
    uint64_t matrix_cycles = 0;
    uint64_t vector_cycles = 0;
    uint64_t buffer_cycles = 0;
    uint64_t dram_cycles = 0;
    uint64_t noc_cycles = 0;
    
    // Energy statistics (the unit is Joule)
    double e2e_energy = 0.0;
    double controller_energy = 0.0;
    double matrix_energy = 0.0;
    double vector_energy = 0.0;
    double buffer_energy = 0.0;
    double dram_energy = 0.0;
    double noc_energy = 0.0;

    double flop_count = 0.0;
    double memory_access_bytes = 0.0;

    uint64_t compute_non_overlap_cycles = 0;
    uint64_t matrix_bubble_on_chip_cycles = 0;
    uint64_t matrix_bubble_dram_cycles = 0;
    uint64_t matrix_bubble_noc_cycles = 0;
    uint64_t matrix_bubble_dram_noc_cycles = 0;

    double matrix_util() const { 
        return matrix_cycles*1.0 / e2e_cycles;
    }

    double vector_util() const {
        return vector_cycles*1.0 / e2e_cycles;
    }

    double buffer_util() const {
        return buffer_cycles*1.0 / e2e_cycles;
    }

    double dram_util() const {
        return dram_cycles*1.0 / e2e_cycles;
    }

    double noc_util() const {
        return noc_cycles*1.0 / e2e_cycles;
    }

    // The unit of frequency is MHz
    // Return value is GB/s
    double dram_bw(double frequency = 1000) const {
        return (memory_access_bytes/_GB) / (e2e_cycles/(frequency*_MHz));
    }

    double dram_active_bw(double frequency = 1000) const {
        if (dram_cycles == 0) return 0.0;
        return (memory_access_bytes/_GB) / (dram_cycles/(frequency*_MHz));
    }

    double arithmetic_intensity() const {
        return flop_count*1.0 / memory_access_bytes;
    }

    double compute_non_overlap_ratio() const {
        return (e2e_cycles > 0) ? compute_non_overlap_cycles * 1.0 / e2e_cycles : 0.0;
    }

    void update_e2e_energy() {
        e2e_energy = controller_energy + matrix_energy + vector_energy + buffer_energy + dram_energy + noc_energy;
    }

    Stats& operator += (const Stats& s) {
        e2e_cycles += s.e2e_cycles;
        matrix_cycles += s.matrix_cycles;
        vector_cycles += s.vector_cycles;
        buffer_cycles += s.buffer_cycles;
        dram_cycles += s.dram_cycles;
        noc_cycles += s.noc_cycles;

        flop_count += s.flop_count;
        memory_access_bytes += s.memory_access_bytes;
        compute_non_overlap_cycles += s.compute_non_overlap_cycles;
        matrix_bubble_on_chip_cycles += s.matrix_bubble_on_chip_cycles;
        matrix_bubble_dram_cycles += s.matrix_bubble_dram_cycles;
        matrix_bubble_noc_cycles += s.matrix_bubble_noc_cycles;
        matrix_bubble_dram_noc_cycles += s.matrix_bubble_dram_noc_cycles;

        e2e_energy += s.e2e_energy;
        controller_energy += s.controller_energy;
        matrix_energy += s.matrix_energy;
        vector_energy += s.vector_energy;
        buffer_energy += s.buffer_energy;
        dram_energy += s.dram_energy;
        noc_energy += s.noc_energy;

        return *this;
    }

    void display_stats(double max_bw = 0, double frequency = 1000, std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "" << "(1) Latency Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "E2E Cycles: " << e2e_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Cycles: " << matrix_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Vector Cycles: " << vector_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Buffer Cycles: " << buffer_cycles << std::endl;
        os << std::setw(indent+4) << "" << "DRAM Cycles: " << dram_cycles << std::endl;
        os << std::setw(indent+4) << "" << "NoC Cycles: " << noc_cycles << std::endl;

        os << std::setw(indent) << "" << "(2) Energy Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "E2E Energy: " << e2e_energy << std::endl;
        os << std::setw(indent+4) << "" << "Controller Energy: " << controller_energy << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Energy: " << matrix_energy << std::endl;
        os << std::setw(indent+4) << "" << "Vector Energy: " << vector_energy << std::endl;
        os << std::setw(indent+4) << "" << "Buffer Energy: " << buffer_energy << std::endl;
        os << std::setw(indent+4) << "" << "DRAM Energy: " << dram_energy << std::endl;
        os << std::setw(indent+4) << "" << "NoC Energy: " << noc_energy << std::endl;

        os << std::setw(indent) << "" << "(3) Arithmetic Intensity Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "FLOP Count: " << flop_count << std::endl;
        os << std::setw(indent+4) << "" << "Memory Access Bytes: " << memory_access_bytes << std::endl;
        os << std::setw(indent+4) << "" << "Arithmetic Intensity: " << arithmetic_intensity() << std::endl;

        os << std::setw(indent) << "" << "(4) Compute Overlap Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "Compute Non-Overlap Cycles: " << compute_non_overlap_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Compute Non-Overlap Ratio: " << compute_non_overlap_ratio() << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Bubble On-Chip Cycles: " << matrix_bubble_on_chip_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Bubble DRAM Cycles: " << matrix_bubble_dram_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Bubble NoC Cycles: " << matrix_bubble_noc_cycles << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Bubble DRAM+NoC Cycles: " << matrix_bubble_dram_noc_cycles << std::endl;

        os << std::setw(indent) << "" << "(5) Utilization Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "Matrix Cycle Utilization: " << matrix_util() << std::endl;
        os << std::setw(indent+4) << "" << "Vector Cycle Utilization: " << vector_util() << std::endl;
        os << std::setw(indent+4) << "" << "Buffer Cycle Utilization: " << buffer_util() << std::endl;
        os << std::setw(indent+4) << "" << "DRAM Cycle Utilization: " << dram_util() << std::endl;
        os << std::setw(indent+4) << "" << "NoC Cycle Utilization: " << noc_util() << std::endl;
        os << std::setw(indent+4) << "" << "DRAM Bandwidth (GB/s): " << dram_bw(frequency) << std::endl;
        if(max_bw > 0) {
            os << std::setw(indent+4) << "" << "Max DRAM Bandwidth (GB/s): " << max_bw << std::endl;
            os << std::setw(indent+4) << "" << "DRAM Bandwidth Utilization: " << dram_bw(frequency) / max_bw << std::endl;
        }
        
    }
};


typedef std::vector<Stats> CoreStatsCollection;
typedef std::vector<std::pair<std::string, Stats>> OperatorStatsCollection;


struct Performance {
    double core_bw;
    double chip_bw;
    double chip_frequency;

    Stats e2e_stats;
    CoreStatsCollection core_stats;
    OperatorStatsCollection operator_stats;
    std::vector<std::pair<std::string, CoreStatsCollection>> operator_stats_per_core;

    void display_stats(std::ostream& os=std::cout, int indent=0) {
        os << std::setw(indent) << "" << "Full Chip E2E Statistics:" << std::endl;
        os << std::setw(indent+4) << "" << "(1) Latency Statistics:" << std::endl;
        os << std::setw(indent+8) << "" << "E2E Cycles: " << e2e_stats.e2e_cycles << std::endl;
        os << std::setw(indent+8) << "" << "Matrix Cycles: " << e2e_stats.matrix_cycles << std::endl;
        os << std::setw(indent+8) << "" << "Vector Cycles: " << e2e_stats.vector_cycles << std::endl;
        os << std::setw(indent+8) << "" << "Buffer Cycles: " << e2e_stats.buffer_cycles << std::endl;
        os << std::setw(indent+8) << "" << "DRAM Cycles: " << e2e_stats.dram_cycles << std::endl;
        os << std::setw(indent+8) << "" << "NoC Cycles: " << e2e_stats.noc_cycles << std::endl;
        os << std::setw(indent+4) << "" << "(2) Energy Statistics:" << std::endl;
        os << std::setw(indent+8) << "" << "E2E Energy: " << e2e_stats.e2e_energy << std::endl;
        os << std::setw(indent+8) << "" << "Controller Energy: " << e2e_stats.controller_energy << std::endl;
        os << std::setw(indent+8) << "" << "Matrix Energy: " << e2e_stats.matrix_energy << std::endl;
        os << std::setw(indent+8) << "" << "Vector Energy: " << e2e_stats.vector_energy << std::endl;
        os << std::setw(indent+8) << "" << "Buffer Energy: " << e2e_stats.buffer_energy << std::endl;
        os << std::setw(indent+8) << "" << "DRAM Energy: " << e2e_stats.dram_energy << std::endl;
        os << std::setw(indent+8) << "" << "NoC Energy: " << e2e_stats.noc_energy << std::endl;

        os << std::setw(indent) << "" << "Per Core E2E Statistics:" << std::endl;
        for(int i=0; i<core_stats.size(); ++i) {
            os << std::setw(indent+4) << "" << "Core " << i << ":" << std::endl;

            os << std::setw(indent+8) << "" << "(1) Latency Statistics:" << std::endl;
            os << std::setw(indent+12) << "" << "E2E Cycles: " << core_stats[i].e2e_cycles << std::endl;
            os << std::setw(indent+12) << "" << "Matrix Cycles: " << core_stats[i].matrix_cycles << std::endl;
            os << std::setw(indent+12) << "" << "Vector Cycles: " << core_stats[i].vector_cycles << std::endl;
            os << std::setw(indent+12) << "" << "Buffer Cycles: " << core_stats[i].buffer_cycles << std::endl;
            os << std::setw(indent+12) << "" << "DRAM Cycles: " << core_stats[i].dram_cycles << std::endl;
            os << std::setw(indent+12) << "" << "NoC Cycles: " << core_stats[i].noc_cycles << std::endl;

            os << std::setw(indent+8) << "" << "(2) Energy Statistics:" << std::endl;
            os << std::setw(indent+12) << "" << "E2E Energy: " << core_stats[i].e2e_energy << std::endl;
            os << std::setw(indent+12) << "" << "Controller Energy: " << core_stats[i].controller_energy << std::endl;
            os << std::setw(indent+12) << "" << "Matrix Energy: " << core_stats[i].matrix_energy << std::endl;
            os << std::setw(indent+12) << "" << "Vector Energy: " << core_stats[i].vector_energy << std::endl;
            os << std::setw(indent+12) << "" << "Buffer Energy: " << core_stats[i].buffer_energy << std::endl;
            os << std::setw(indent+12) << "" << "DRAM Energy: " << core_stats[i].dram_energy << std::endl;
            os << std::setw(indent+12) << "" << "NoC Energy: " << core_stats[i].noc_energy << std::endl;
        }

        os << std::setw(indent) << "" << "Per Operator Statistics:" << std::endl;
        for(int i=0; i<operator_stats.size(); ++i) {
            os << std::setw(indent+4) << "" << "Operator " << operator_stats[i].first << ":" << std::endl;
            operator_stats[i].second.display_stats(chip_bw, chip_frequency, os, indent+8);
        }

        os << std::setw(indent) << "" << "Per Operator Statistics per Core:" << std::endl;
        for(int i=0; i<operator_stats_per_core.size(); ++i) {
            os << std::setw(indent+4) << "" << "Operator " << operator_stats_per_core[i].first << ":" << std::endl;
            for(int j=0; j<operator_stats_per_core[i].second.size(); ++j) {
                os << std::setw(indent+8) << "" << "Core " << j << ":" << std::endl;
                operator_stats_per_core[i].second[j].display_stats(core_bw, chip_frequency, os, indent+12);
            }
        }
    }
};


}


#endif // _ATLASIM_STATS_H
