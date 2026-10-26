# ATLAS 编译器扩展

设计、公平对照规则和结果只维护在[主文档](../../../Distributed_DRAM_SRAM/near_mem_gemm/docs/ATLAS编译器扩展.md)。

| 目录/文件 | 职责 |
|---|---|
| `sim.py`、`tasks.py` | ATLAS 边界：硬件解析与指纹、独立进程执行；物理区域、Timeline、布局转换与线性 all-gather |
| `gemm/` | GEMM 编译器：`plan`/`ownership`（Ps/Pt 空间与代数）、`compiler`/`program`/`lower`（调度、回放、降为原生任务）、`space`（候选枚举与 proxy）、`native`（原生 ATLAS 编译：同一 proxy 选 tile 候选再模拟）、`distribution`（单份输入的复制计费） |
| `model/` | 模型算子：`attention`（静态 KV 与 fused attention）、`kernels`（GEMM 图接口及 norm/gate/residual）、`transformer`（建图、算子边界布局、原生 tile 调优） |
| `search.py` | 通用搜索：`shortlist`、`diverse`（均支持下界剪枝）、`beam`、`anneal`、`measure_all` |
| `methods.py` | 方法注册表：`local`、`ours`；新增 baseline 在此加一项 |
| `experiments/` | 实验入口：`gemm`、`attention`、`transformer`；`motivation` 汇总已有结果 |

```bash
.venv/bin/python -m frontend.near_mem.experiments.gemm --help
.venv/bin/python -m frontend.near_mem.experiments.attention --help
.venv/bin/python -m frontend.near_mem.experiments.transformer --help
.venv/bin/python -m pytest -q tests/near_mem
```
