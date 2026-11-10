# ATLAS 编译器扩展

设计、公平对照规则和结果只维护在[主文档](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/ATLAS编译器扩展.md)。

| 目录/文件 | 职责 |
|---|---|
| `sim.py`、`tasks.py` | ATLAS 边界：硬件解析与指纹、独立进程执行；物理区域、Timeline、布局转换与线性 all-gather |
| `gemm/` | GEMM 编译器：`plan`/`ownership`（Ps/Pt 空间与代数）、`compiler`/`program`/`lower`（调度、回放、降为原生任务）、`space`（候选枚举与 proxy）、`native`（原生 ATLAS 编译：同一 proxy 选 tile 候选再模拟）、`distribution`（单份输入的复制计费） |
| `model/` | 模型算子：`attention`（静态 KV 与空间）、`attention_schedule`（公共融合调度、回放与 lowering）、`attention_flows`（DRAM/Flash 专属事件生成）、`attention_proxy`（共用排序）、`kernels`（GEMM 图接口及 norm/gate/residual）、`transformer`（建图与边界布局） |
| `output.py` | GEMM 与 attention 共用的归约输出方式：分片、汇聚到组端点、组内复制；只有我们的空间在整图中结合下游布局选择 |
| `search.py` | 通用搜索：`shortlist`、`diverse`（均支持下界剪枝）、`beam`、`anneal`、`measure_all` |
| `methods.py` | 方法注册表：`tp`（TP 基线）、`atlas`（上游检查）、`local`、`ours`、`dram`、`t10`、`flash`；新增 baseline 在此加一项 |
| `experiments/` | 实验入口：`gemm`、`attention`、`transformer`（每次一个方法）；`inference` 按固定 batch/输入/输出折算整段推理；`attention_space` 对照新旧融合调度空间；`compare` 计算加速比，`motivation` 汇总已有结果；`dram_calibration` 为 proxy 的 DRAM 通道模型标定系数 |

Attention 和 GEMM 共用两阶段评估：proxy 排序，ATLAS 模拟器复核。GEMM proxy 的 DRAM 项按编译器实际的 tile 读写次数，经通道模型计费（最忙通道的 burst 数与打开的行数，系数按 DRAM 时序预设用 `experiments.dram_calibration` 标定一次，换通道数不必重标）。proxy 不区分 mesh 方向，所以只扫描一个方向，候选再补上转置的孪生方案，交给边界打分和整图模拟区分。Attention proxy 复用实际 lowering 任务，按硬件吞吐与依赖估算；单算子 `--budget`、整图 `--attention-budget` 默认 32，额外保留不同空间/输出族的代表，0 表示穷举。

各方法已有独立的 fused-attention 数据流：TP 按 context 并行；ours 允许 Q/在线状态、KV 及联合 SRAM 轮转；DRAM 按完整逻辑块落盘轮转；T10 在固定 DRAM 放置上做 SRAM 轮转，并计入实际 NUMA 访问；FlashGEMM 用固定 query/context 网格、线性收集后计算，比较 Q/状态常驻与 KV panel 常驻两种循环。后两者是数据流的融合迁移扩展。所有方法共用 proxy 和模拟器；均不将 score 矩阵写入 DRAM。具体规则维护在[统一约定](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/约定.md)。

刚性网格 baseline 遇到边界维度时，只用零填充补齐到各自规则布局要求，并按补齐后的完整 MAC 与 DRAM 流量排序和模拟；计划同时记录逻辑形状和执行形状。原生 ATLAS 单算子若没有精确 M/N 分区就补齐；TP 单算子在没有合法 M/N 网格时补齐到一个合法网格，整图 TP/原生 TP 只补固定网格要求的 K/N 轴；FlashGEMM 将 M/N/K 补齐到 mesh side。T10 的 `M=1` 用单份一维 DRAM 布局：A 沿 K 切分，B/C 沿 N 切分到所有 bank，不复制 A，远端访问由模拟器计费；其他不对齐维度才补齐到其规则网格。可自然改用其他空间轴的方案不强制 padding。

`ours` 是完整的 Ps/Pt 空间：Pt=1（不轮转）是其中的点，Ps 同时枚举 mesh 的两个方向，所以 TP 网格自然是 `ours` 的子集，不再拼接其他方法的候选；整个空间一次扫描、一次 shortlist，按 proxy 排名送入模拟器。`local` 只作消融（`ours` 限定 Pt=1），单独运行。`dram` 只表示 DRAM 端轮转，不退回 `tp` 或 `local`；比较 DRAM 与 SRAM 轮转时各自单独运行。

传递顺序：同一轴上两个张量周期不同（如 X 周期 4、W 周期 2）时，若该轴的共享层在最外、单独层紧随其后（只在这一轴轮转的计划按 ELK 规则即是如此），按 T10 的均匀环：长周期张量每步沿整个环移一格，短周期张量每 q 步移一次，起点为 `x_long + q*x_short`，每一跳都在环上相邻核之间；其余情况沿用各 gcd 层的混合进制顺序。两种顺序访问相同的块、传输量相同，只差链路负载。论文中把传递顺序写成计划的一部分（传递组上的循环顺序，起点由放置决定）。

算子边界：只有我们的空间（`ours`、`local`、`dram`，`Method.boundary`）会按计划改变 DRAM 驻留，所以只有它们展开归约输出方式（分片、汇聚到组端点、组内复制），并与相邻算子联合搜索（beam + 坐标下降）。TP 用其原生 collective 与方向对齐，T10/FlashGEMM 用固定二维分块，都保持自身数据流规定的边界：每个算子取单独模拟最快的计划，整图跑一次，边界不一致处照实计 relayout。`--no-joint` 让 `ours` 也这样做，作为边界协同的消融。论文中可把“归约终点结合下游需求选择”作为整图编译的一个小点。

FlashGEMM 尽量按原文：每步持有整段本地 M/N，只搜 K tile；放不下 SRAM 时只把 M 最小程度地切小（WS 下能放下就让 B 常驻），仍放不下才切 N panel；OS 与 WS 都交给模拟器选择。

DRAM 搬运生成源 DRAM 读、NoC 发送/接收、目标 DRAM 写；与 ATLAS 一致，SRAM 计发送读和接收写，不额外计源端 staging 写及目标端 staging 读。搬运仍占 SRAM 缓冲，临时 DRAM 副本仍占容量；此 DMA 中转抽象可统一修订。历史 KV 的 owner 和内容不因轮转改变。

快速实验口径（`inference`，各方法一致）：prefill 每个请求按 512-token chunk 切分（末块补齐），GEMM 的 M=chunk 与 batch 无关，attention 取各 chunk 前缀长度的平均值；decode 用给定 batch，attention 取中间生成位置的长度。每个阶段只搜索并模拟一层代表点，再乘步数和层数。这是私下取数的方式，论文按分段流程描述（静态片段按 token 桶，attention 按 KV 长度区间重规划）。

General task 的 DRAM 读可用 `ready_iter` 声明首次使用位置。后端将原有 Ramulator 批次延迟放入每核一条串行服务队列，允许预取跨 iteration，在首次使用及算子结束时等待。批次内仍由 Ramulator 建模，批次间不做请求重排，NoC 仍由 BookSim 模拟；这是保守的批次服务模型，不是逐请求异步 DRAM 后端。未注解的任务保留原有逐步等待语义。GEMM 只在双缓冲空闲时提前读取，attention 由 packet 生命周期检查 staging 与轮转缓冲容量。

```bash
.venv/bin/python -m frontend.near_mem.experiments.gemm --help
.venv/bin/python -m frontend.near_mem.experiments.attention --help
.venv/bin/python -m frontend.near_mem.experiments.transformer --help
.venv/bin/python -m frontend.near_mem.experiments.inference --help
.venv/bin/python -m frontend.near_mem.experiments.compare results/near_mem/...
.venv/bin/python -m pytest -q tests/near_mem
```
