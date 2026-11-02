# Enhanced LatentMAS Advanced

LatentMAS 题内多角色 KV 交接压缩研究仓库。这里作为后续 LatentMAS 代码、协议和结果版本的统一提交位置。

## 当前版本：2026-11-02 冻结快照

已完成 GPQA-main **448题**，其中 Diamond **198题**为严格子集。固定 seed42、latent10/40，比较 Full、作者 HeadOBF、题内 JIVE 和接收方注意力选择。3584条唯一答案中复用1420条，只新增2164条；未重跑兼容旧答案。

本版包含与完成实验逐文件SHA一致的核心代码与运行支撑、无原始回答的逐题指标、16格汇总及归档证明。它是**历史研究源码快照**：离线结果重算可直接运行，原云端管理器仍保留部署路径和私有文件前提，尚不是通用的一键GPU启动器。

## 正确率

| 方法 | Main 10步 | Main 40步 | Diamond 10步 | Diamond 40步 |
|---|---:|---:|---:|---:|
| Full | 51.56% | 52.90% | 55.05% | 57.07% |
| HeadOBF | 48.88% | 47.54% | 50.51% | 54.04% |
| 题内JIVE | 45.54% | 46.43% | 46.46% | 49.49% |
| 接收方注意力 | 50.89% | 49.11% | 53.03% | 52.02% |

## KV字节指标（与正确率分开）

三种压缩方法预算相同。统计最终交接到Judger前的KV，包含受保护latent及sink：

| 范围 | 10步保留 / 减少 | 40步保留 / 减少 |
|---|---:|---:|
| Main448 | 13.22% / 86.78% | 20.49% / 79.51% |
| Diamond198 | 12.85% / 87.15% | 19.98% / 80.02% |

压缩字节与实际tensor回执一致；Full字节按实际保留位置和锁定BF16维度推算。字节比不代表实测网络、GPU或FLOP收益。生成时间不作为本轮效果指标。

## 快速核对（无需GPU或数据集）

```bash
python scripts/verify_snapshot.py
python scripts/reproduce_results.py --check
python -m unittest discover -s tests -v
```

前两条仅依赖Python标准库。CPU张量机械测试需要PyTorch；它们不运行语言模型，也不代表重新生成实验结果。

## 目录

- `snapshot/run/`：完成实验的科学worker、KV选择、JIVE、冻结解析器及历史管理器。
- `snapshot/bundle/runtime/`：模型加载、计费/位置检查、原生推理封装与带许可证的上游LatentMAS源码。
- `configs/protocol.json`：不含私有路径的实验参数。
- `results/gpqa_main448/`：3584行逐题指标和16格汇总；不含题文、gold或模型原答。
- `provenance/`：源SHA、题ID、模型SHA锁、原归档核验摘要和软件版本。
- `docs/`：方法、结果口径、运行前提及研究边界。

详见 [方法说明](docs/METHOD.md)、[复现说明](docs/REPRODUCING.md)、[结果与限制](docs/RESULTS.md)、[第三方来源](THIRD_PARTY_NOTICES.md)。

## 研究边界

这是单seed、包含已使用开发题和历史评测题的探索结果，不是新盲测。接收方方法在Main两档优于本轮OBF/JIVE，但Diamond40低于OBF，尚不能声称全面胜出或稳定显著优势。接收方感知已有相关工作；本仓库不声称首次提出该概念。SVD/接收方输出修正属于后续假设，本版未混入这些未验证方案。
