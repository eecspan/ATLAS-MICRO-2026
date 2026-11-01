# ATLAS 编译器扩展

设计、公平对照规则和结果只维护在[主文档](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/ATLAS编译器扩展.md)。

| 目录/文件 | 职责 |
|---|---|
| `sim.py`、`tasks.py` | ATLAS 边界：硬件解析与指纹、独立进程执行；物理区域、Timeline、布局转换与线性 all-gather |
| `gemm/` | GEMM 编译器：`plan`/`ownership`（Ps/Pt 空间与代数）、`compiler`/`program`/`lower`（调度、回放、降为原生任务）、`space`（候选枚举与 proxy）、`native`（原生 ATLAS 编译：同一 proxy 选 tile 候选再模拟）、`distribution`（单份输入的复制计费） |
| `model/` | 模型算子：`attention`（静态 KV 与空间）、`attention_schedule`（公共融合调度、回放与 lowering）、`attention_flows`（DRAM/Flash 专属事件生成）、`attention_proxy`（共用排序）、`kernels`（GEMM 图接口及 norm/gate/residual）、`transformer`（建图与边界布局） |
| `output.py` | GEMM 与 attention 共用的归约输出方式：分片、汇聚到组端点、组内复制；整图搜索结合下游布局选择 |
| `search.py` | 通用搜索：`shortlist`、`diverse`（均支持下界剪枝）、`beam`、`anneal`、`measure_all` |
| `methods.py` | 方法注册表：`tp`（TP 基线）、`atlas`（上游检查）、`local`、`ours`、`dram`、`t10`、`flash`；新增 baseline 在此加一项 |
| `experiments/` | 实验入口：`gemm`、`attention`、`transformer`（每次一个方法）；`attention_space` 对照新旧融合调度空间；`compare` 计算加速比，`motivation` 汇总已有结果 |

Attention 和 GEMM 共用两阶段评估：proxy 排序，ATLAS 模拟器复核。Attention proxy 复用实际 lowering 任务，按硬件吞吐与依赖估算；单算子 `--budget`、整图 `--attention-budget` 默认 32，额外保留不同空间/输出族的代表，0 表示穷举。

各方法已有独立的 fused-attention 数据流：TP 按 context 并行；ours 允许 Q/在线状态、KV 及联合 SRAM 轮转；DRAM 按完整逻辑块落盘轮转；T10 在固定 DRAM 放置上做 SRAM 轮转，并计入实际 NUMA 访问；FlashGEMM 用固定 query/context 网格、线性收集后计算，比较 Q/状态常驻与 KV panel 常驻两种循环。后两者是数据流的融合迁移扩展。所有方法共用 proxy、模拟器和输出归约/边界布局搜索；均不将 score 矩阵写入 DRAM。具体规则维护在[统一约定](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/约定.md)。

刚性网格 baseline 遇到边界维度时，只用零填充补齐到各自规则布局要求，并按补齐后的完整 MAC 与 DRAM 流量排序和模拟；计划同时记录逻辑形状和执行形状。原生 ATLAS 单算子若没有精确 M/N 分区就补齐；TP 单算子在没有合法 M/N 网格时补齐到一个合法网格，整图 TP/原生 TP 只补固定网格要求的 K/N 轴；FlashGEMM 将 M/N/K 补齐到 mesh side。T10 的 `M=1` 用单份一维 DRAM 布局：A 沿 K 切分，B/C 沿 N 切分到所有 bank，不复制 A，远端访问由模拟器计费；其他不对齐维度才补齐到其规则网格。可自然改用其他空间轴的方案不强制 padding。

`dram` 只表示 DRAM 端轮转，不自动退回 `tp` 或 `local`。`ours` 保留 `tp/local` 作为可行的回退子空间；如果要单独比较 DRAM 与 SRAM 轮转，应分别运行这些方法，而不是把回退混进 DRAM baseline。

DRAM 搬运生成源 DRAM 读、NoC 发送/接收、目标 DRAM 写；与 ATLAS 一致，SRAM 计发送读和接收写，不额外计源端 staging 写及目标端 staging 读。搬运仍占 SRAM 缓冲，临时 DRAM 副本仍占容量；此 DMA 中转抽象可统一修订。历史 KV 的 owner 和内容不因轮转改变。

General task 的 DRAM 读可用 `ready_iter` 声明首次使用位置。后端将原有 Ramulator 批次延迟放入每核一条串行服务队列，允许预取跨 iteration，在首次使用及算子结束时等待。批次内仍由 Ramulator 建模，批次间不做请求重排，NoC 仍由 BookSim 模拟；这是保守的批次服务模型，不是逐请求异步 DRAM 后端。未注解的任务保留原有逐步等待语义。GEMM 只在双缓冲空闲时提前读取，attention 由 packet 生命周期检查 staging 与轮转缓冲容量。

```bash
.venv/bin/python -m frontend.near_mem.experiments.gemm --help
.venv/bin/python -m frontend.near_mem.experiments.attention --help
.venv/bin/python -m frontend.near_mem.experiments.transformer --help
.venv/bin/python -m frontend.near_mem.experiments.compare results/near_mem/...
.venv/bin/python -m pytest -q tests/near_mem
```
