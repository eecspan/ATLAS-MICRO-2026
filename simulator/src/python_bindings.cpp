#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h> // Add this include

#include "common/arch.h"
#include "common/input.h"
#include "common/stats.h"
#include "common/task.h"
#include "chip/chip.h"

namespace py = pybind11;
using namespace atlasim;


PYBIND11_MODULE(atlasim, m) {
    m.doc() = "Python bindings for ATLAS simulator classes";

    // Redirect C++ streams to Python
    py::add_ostream_redirect(m, "ostream_redirect");

    // --- Enums ---

    py::enum_<OperatorType>(m, "SimulatorOperatorType")
        .value("GEMM", OperatorType::GEMM)
        .value("DecodeAttention", OperatorType::DecodeAttention)
        .value("Communication", OperatorType::Communication)
        .value("SPMDGeneral", OperatorType::SPMDGeneral)
        .value("MPMDGeneral", OperatorType::MPMDGeneral)
        .export_values()
        .def("to_string", [](OperatorType type) {
            return optype2str.at(type);
        })
        .def_static("from_str", [](std::string str) {
            return str2optype.at(str);
        });

    // --- Arch Configs ---

    py::class_<ControllerConfig>(m, "ControllerConfig")
        .def(py::init<>())
        .def_readwrite("power", &ControllerConfig::power)
        .def_readwrite("area", &ControllerConfig::area);

    py::class_<MatrixConfig>(m, "MatrixConfig")
        .def(py::init<>())
        .def_readwrite("mac_num", &MatrixConfig::mac_num)
        .def_readwrite("power", &MatrixConfig::power)
        .def_readwrite("area", &MatrixConfig::area);

    py::class_<VectorConfig>(m, "VectorConfig")
        .def(py::init<>())
        .def_readwrite("vec_num", &VectorConfig::vec_num)
        .def_readwrite("power", &VectorConfig::power)
        .def_readwrite("area", &VectorConfig::area);

    py::class_<BufferConfig>(m, "BufferConfig")
        .def(py::init<>())
        .def_readwrite("buffer_size", &BufferConfig::buffer_size)
        .def_readwrite("read_bw", &BufferConfig::read_bw)
        .def_readwrite("write_bw", &BufferConfig::write_bw)
        .def_readwrite("power", &BufferConfig::power)
        .def_readwrite("area", &BufferConfig::area);

    py::class_<DRAMConfig>(m, "DRAMConfig")
        .def(py::init<>())
        .def_readwrite("config_path", &DRAMConfig::config_path)
        .def_readwrite("power", &DRAMConfig::power)
        .def_readwrite("area", &DRAMConfig::area);

    py::class_<NoCConfig>(m, "NoCConfig")
        .def(py::init<>())
        .def_readwrite("topology", &NoCConfig::topology)
        .def_readwrite("config_path", &NoCConfig::config_path)
        .def_readwrite("flit_size", &NoCConfig::flit_size)
        .def_readwrite("power", &NoCConfig::power)
        .def_readwrite("area", &NoCConfig::area);

    py::class_<ChipConfig, std::shared_ptr<ChipConfig>>(m, "ChipConfig")
        .def(py::init<>())
        .def_readwrite("frequency", &ChipConfig::frequency)
        .def_readwrite("core_num", &ChipConfig::core_num)
        .def_readwrite("controller_config", &ChipConfig::controller_config)
        .def_readwrite("matrix_config", &ChipConfig::matrix_config)
        .def_readwrite("vector_config", &ChipConfig::vector_config)
        .def_readwrite("buffer_config", &ChipConfig::buffer_config)
        .def_readwrite("dram_config", &ChipConfig::dram_config)
        .def_readwrite("has_noc", &ChipConfig::has_noc)
        .def_readwrite("noc_config", &ChipConfig::noc_config)
        .def("display_stats", [](ChipConfig& self, int indent) {
            self.display_stats(std::cout, indent);
        }, py::arg("indent")=0);

    // --- Input / AttnInput ---
    
    // Abstract base class (AttnInput)
    py::class_<AttnInput, std::shared_ptr<AttnInput>>(m, "AttnInput")
        .def(py::init<>())
        .def_readwrite("q_head_num", &AttnInput::q_head_num)
        .def_readwrite("kv_head_num", &AttnInput::kv_head_num)
        .def_readwrite("q_head_num_per_kv", &AttnInput::q_head_num_per_kv)
        .def_readwrite("qk_head_dim", &AttnInput::qk_head_dim)
        .def_readwrite("v_head_dim", &AttnInput::v_head_dim)
        .def_readwrite("input_tensor_name", &AttnInput::input_tensor_name)
        .def_readwrite("output_tensor_name", &AttnInput::output_tensor_name)
        .def_readwrite("kv_cache_tensor_name", &AttnInput::kv_cache_tensor_name)
        .def_readwrite("total_slot_num", &AttnInput::total_slot_num)
        .def_readwrite("kv_head_addr_offset", &AttnInput::kv_head_addr_offset)
        .def("display_stats", [](AttnInput& self, int indent) {
            self.display_stats(std::cout, indent);
        }, py::arg("indent")=0);

    // DAttnCoreInput (used inside DAttnInput)
    py::class_<DAttnCoreInput>(m, "DAttnCoreInput")
        .def(py::init<>())
        .def_readwrite("core_id", &DAttnCoreInput::core_id)
        .def_readwrite("request_indices", &DAttnCoreInput::request_indices)
        .def_readwrite("request_tile_count", &DAttnCoreInput::request_tile_count)
        .def_readwrite("request_tile_info_list", &DAttnCoreInput::request_tile_info_list)
        .def("display_stats", [](DAttnCoreInput& self, int indent) {
            self.display_stats(std::cout, indent);
        }, py::arg("indent")=0);

    // DAttnInput (inherits from AttnInput)
    py::class_<DAttnInput, AttnInput, std::shared_ptr<DAttnInput>>(m, "DAttnInput")
        .def(py::init<>())
        .def_readwrite("token_tile_size", &DAttnInput::token_tile_size)
        .def_readwrite("block_size", &DAttnInput::block_size)
        .def_readwrite("total_block_num", &DAttnInput::total_block_num)
        .def_readwrite("core_input_list", &DAttnInput::core_input_list)
        .def("display_stats", [](DAttnInput& self, int indent) {
            self.display_stats(std::cout, indent);
        }, py::arg("indent")=0);

    // --- Stats / Performance ---

    py::class_<atlasim::Stats>(m, "Stats")
        .def(py::init<>())
        .def_readwrite("e2e_cycles", &atlasim::Stats::e2e_cycles)
        .def_readwrite("matrix_cycles", &atlasim::Stats::matrix_cycles)
        .def_readwrite("vector_cycles", &atlasim::Stats::vector_cycles)
        .def_readwrite("buffer_cycles", &atlasim::Stats::buffer_cycles)
        .def_readwrite("dram_cycles", &atlasim::Stats::dram_cycles)
        .def_readwrite("noc_cycles", &atlasim::Stats::noc_cycles)
        .def_readwrite("e2e_energy", &atlasim::Stats::e2e_energy)
        .def_readwrite("controller_energy", &atlasim::Stats::controller_energy)
        .def_readwrite("matrix_energy", &atlasim::Stats::matrix_energy)
        .def_readwrite("vector_energy", &atlasim::Stats::vector_energy)
        .def_readwrite("buffer_energy", &atlasim::Stats::buffer_energy)
        .def_readwrite("dram_energy", &atlasim::Stats::dram_energy)
        .def_readwrite("noc_energy", &atlasim::Stats::noc_energy)
        .def_readwrite("flop_count", &atlasim::Stats::flop_count)
        .def_readwrite("memory_access_bytes", &atlasim::Stats::memory_access_bytes)
        .def_readwrite("compute_non_overlap_cycles", &atlasim::Stats::compute_non_overlap_cycles)
        .def_readwrite("matrix_bubble_on_chip_cycles", &atlasim::Stats::matrix_bubble_on_chip_cycles)
        .def_readwrite("matrix_bubble_dram_cycles", &atlasim::Stats::matrix_bubble_dram_cycles)
        .def_readwrite("matrix_bubble_noc_cycles", &atlasim::Stats::matrix_bubble_noc_cycles)
        .def_readwrite("matrix_bubble_dram_noc_cycles", &atlasim::Stats::matrix_bubble_dram_noc_cycles)
        .def("matrix_util", &atlasim::Stats::matrix_util)
        .def("vector_util", &atlasim::Stats::vector_util)
        .def("buffer_util", &atlasim::Stats::buffer_util)
        .def("dram_util", &atlasim::Stats::dram_util)
        .def("noc_util", &atlasim::Stats::noc_util)
        .def("dram_bw", &atlasim::Stats::dram_bw, py::arg("frequency")=1000)
        .def("dram_active_bw", &atlasim::Stats::dram_active_bw, py::arg("frequency")=1000)
        .def("arithmetic_intensity", &atlasim::Stats::arithmetic_intensity)
        .def("compute_non_overlap_ratio", &atlasim::Stats::compute_non_overlap_ratio)
        .def("update_e2e_energy", &atlasim::Stats::update_e2e_energy)
        .def("display_stats", [](atlasim::Stats& self, double max_bw, double frequency, int indent) {
            self.display_stats(max_bw, frequency, std::cout, indent);
        }, py::arg("max_bw")=0, py::arg("frequency")=1000, py::arg("indent")=0);

    py::class_<Performance>(m, "Performance")
        .def(py::init<>())
        .def_readwrite("core_bw", &Performance::core_bw)
        .def_readwrite("chip_bw", &Performance::chip_bw)
        .def_readwrite("chip_frequency", &Performance::chip_frequency)
        .def_readwrite("e2e_stats", &Performance::e2e_stats)
        .def_readwrite("core_stats", &Performance::core_stats)
        .def_readwrite("operator_stats", &Performance::operator_stats)
        .def_readwrite("operator_stats_per_core", &Performance::operator_stats_per_core)
        .def("display_stats", [](Performance& self, int indent) {
            self.display_stats(std::cout, indent);
        }, py::arg("indent")=0);

    // --- Chip ---

    py::class_<Chip>(m, "Chip")
        .def(py::init<std::string, std::string, std::string, std::string>(),
            py::arg("arch_config_path"),
            py::arg("operator_list_path"),
            py::arg("placement_map_path"),
            py::arg("log_file_path") = "")
        .def("get_arch_config", &Chip::get_arch_config)
        .def("reset_execution_status", &Chip::reset_execution_status, py::arg("operator_list_path")="")
        .def("simulate", &Chip::simulate)
        .def("get_max_dram_bw_per_core", &Chip::get_max_dram_bw_per_core)
        .def("get_max_compute_capacity_per_core", &Chip::get_max_compute_capacity_per_core)
        .def("get_comp_bw_ratio", &Chip::get_comp_bw_ratio)
        .def("get_max_dram_bw", &Chip::get_max_dram_bw);
}
