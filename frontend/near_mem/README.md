# ATLAS 编译器扩展

统一维护 [主文档](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/ATLAS编译器扩展.md)：编译空间、公平对照、静态 KV/fused Attention、跨算子布局、运行命令和已验证结果。

入口：

- `python -m frontend.near_mem.compare --help`：独立 GEMM。
- `python -m frontend.near_mem.experiment --help`：固定工作负载的 dense Transformer prefill / 一步 decode。

原生 GEMM、预取和 Q/KV/FFN collective 保留；所有方案共用 ATLAS 后端。新入口的 baseline 包含共同的 KV/Attention 扩展，不等同于未修改的上游 decode 程序。
