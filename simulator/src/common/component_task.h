#ifndef _ATLASIM_COMPONENT_TASK_H
#define _ATLASIM_COMPONENT_TASK_H

#include <cstdint>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <yaml-cpp/yaml.h>


namespace atlasim {


struct MatrixTask {
    std::string name;
    int mac_count;

    MatrixTask() { }
    MatrixTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        mac_count = config["mac_count"].as<int>();
    }
};


struct VectorTask {
    std::string name;
    int vec_count;

    VectorTask() { }
    VectorTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        vec_count = config["vec_count"].as<int>();
    }
};


struct BufferTask {
    std::string name;
    bool is_write;
    int byte_count;

    BufferTask() { }
    BufferTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        is_write = config["is_write"].as<bool>();
        byte_count = config["byte_count"].as<int>();
    }
};


struct DRAMTask {
    std::string name;
    bool is_write;

    int init_iter = 0;
    int stride_iter = 0;
    int total_iter = 0;
    // General-task read: simulator iteration of its first SRAM consumer.
    // Absent means the legacy read-before-next-iteration barrier.
    int ready_iter = -1;

    std::vector<int64_t> access_base;
    std::vector<int64_t> access_extent;
    std::vector<int64_t> access_stride_add;
    std::vector<int64_t> access_offset_add;

    DRAMTask() { }
    DRAMTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        is_write = config["is_write"].as<bool>();
        access_base = config["access_base"].as<std::vector<int64_t>>();
        access_extent = config["access_extent"].as<std::vector<int64_t>>();
        access_stride_add = config["access_stride_add"].as<std::vector<int64_t>>();
        access_offset_add = config["access_offset_add"].as<std::vector<int64_t>>();

        init_iter = config["init_iter"].as<int>();
        stride_iter = config["stride_iter"].as<int>();
        total_iter = config["total_iter"].as<int>();
        if(config["ready_iter"]) {
            ready_iter = config["ready_iter"].as<int>();
            if(is_write || ready_iter < 0)
                throw std::invalid_argument("ready_iter requires a nonnegative read dependency");
        }

        if(
            access_base.empty() ||
            access_base.size() != access_extent.size() ||
            access_base.size() != access_stride_add.size() ||
            access_base.size() != access_offset_add.size()
        ) {
            std::cerr << "DRAMTask general access fields must be non-empty and have the same rank." << std::endl;
            exit(-1);
        }
    }

    int access_rank() const {
        return static_cast<int>(access_base.size());
    }
};


struct NoCTask {
    std::string name;
    bool is_send;
    int src;
    int dst;
    int flit_num;
    int init_flit_id;

    NoCTask() { }
    NoCTask(const YAML::Node config) {
        name = config["name"].as<std::string>();
        is_send = config["is_send"].as<bool>();
        src = config["src"].as<int>();
        dst = config["dst"].as<int>();
        flit_num = config["flit_num"].as<int>();
        init_flit_id = config["init_flit_id"].as<int>();
    }
};


}


#endif // _ATLASIM_COMPONENT_TASK_H
