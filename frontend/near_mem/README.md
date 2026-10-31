# ATLAS 编译器扩展

设计、公平对照规则和结果只维护在[主文档](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/ATLAS编译器扩展.md)。

| 目录/文件 | 职责 |
|---|---|
| `sim.py`、`tasks.py` | ATLAS 边界：硬件解析与指纹、独立进程执行；物理区域、Timeline、布局转换与线性 all-gather |
| `gemm/` | GEMM 编译器：`plan`/`ownership`（Ps/Pt 空间与代数）、`compiler`/`program`/`lower`（调度、回放、降为原生任务）、`space`（候选枚举与 proxy）、`native`（原生 ATLAS 编译：同一 proxy 选 tile 候选再模拟）、`distribution`（单份输入的复制计费） |
| `model/` | 模型算子：`attention`（静态 KV、空间与 proxy）、`attention_schedule`（融合调度、数值回放与 lowering）、`kernels`（GEMM 图接口及 norm/gate/residual）、`transformer`（建图、算子边界布局、原生 tile 调优） |
| `output.py` | GEMM 与 attention 共用的归约输出方式：分片、汇聚到组端点、组内复制；整图搜索结合下游布局选择 |
| `search.py` | 通用搜索：`shortlist`、`diverse`（均支持下界剪枝）、`beam`、`anneal`、`measure_all` |
| `methods.py` | 方法注册表：`tp`（TP 基线）、`atlas`（上游检查）、`local`、`ours`、`dram`、`t10`、`flash`；新增 baseline 在此加一项 |
| `experiments/` | 实验入口：`gemm`、`attention`、`transformer`（每次一个方法）；`attention_space` 对照新旧融合调度空间；`compare` 计算加速比，`motivation` 汇总已有结果 |

Attention 和 GEMM 共用两阶段评估：proxy 排序，ATLAS 模拟器复核。Attention proxy 复用实际 lowering 任务，按硬件吞吐与依赖估算；单算子 `--budget`、整图 `--attention-budget` 默认 32，额外保留不同空间/输出族的代表，0 表示穷举。

General task 的 DRAM 读可用 `ready_iter` 声明首次使用位置。后端将原有 Ramulator 批次延迟放入每核一条串行服务队列，允许预取跨 iteration，在首次使用及算子结束时等待。批次内仍由 Ramulator 建模，批次间不做请求重排，NoC 仍由 BookSim 模拟；这是保守的批次服务模型，不是逐请求异步 DRAM 后端。未注解的任务保留原有逐步等待语义。GEMM 只在双缓冲空闲时提前读取，attention 由 packet 生命周期检查 staging 与轮转缓冲容量。

```bash
.venv/bin/python -m frontend.near_mem.experiments.gemm --help
.venv/bin/python -m frontend.near_mem.experiments.attention --help
.venv/bin/python -m frontend.near_mem.experiments.transformer --help
.venv/bin/python -m frontend.near_mem.experiments.compare results/near_mem/...
.venv/bin/python -m pytest -q tests/near_mem
```
