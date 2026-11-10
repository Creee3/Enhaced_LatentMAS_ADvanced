# Role Geometry Pilot Protocol Lock

锁定日期：2026-11-10（北京时间）

## 研究用途

这是一个探索性角色KV几何与接力读出机制验证，不是最终准确率评测。所有当前候选数据集和既有576/705题结果都已经被查看；本pilot不得用于确认性优越性、等价性或普遍泛化结论。

## 数据集范围

每个数据集/分层选择20题，共120题：

- AIME24：20/30；
- GPQA-Diamond：20/198；
- GPQA-nonDiamond：20/250；
- HumanEval+：20/164；
- MedQA：20/300；
- MMLU-Pro：20/12032，按类别分层；
- GSM8K：明确排除，不纳入本pilot。

GPQA Diamond和non-Diamond分别报告；Diamond是GPQA子集，不能和GPQA总数重复相加。

## 确定性抽样

固定salt：`role_geometry_pilot_20261110_v1`。

稳定排序键：

`SHA256(salt + "|" + dataset_group + "|" + stable_id)`，按十六进制值升序。

抽样细节：

- AIME24、GPQA-Diamond、GPQA-nonDiamond、HumanEval+、MedQA：各组直接取最低hash的20题；
- MMLU-Pro：14个类别各取一个最低hash，再取全局剩余最低hash的6题；
- MedQA稳定键使用`source_row_index:idx:pid`，因为本地文件存在重复idx/pid；
- 选择时不读取答案正确性，不按模型结果或方法效果挑题。

## 已生成的私有清单

文件：`pilot_selection_manifest.json`

清单SHA256：`7756c583be41780cd60ae9d88b04a562d99246069d85e654e22982ccb1f344e5`

清单只保存稳定ID、源行号、类别/分层、排序hash和源文件SHA，不保存题目文本、答案或生成内容。120个stable ID已经过唯一性检查。

## 执行顺序

1. 六题工程检查：每个数据集/分层一题，只验证张量形状、原KV/RoPE位置、保护行、同状态回放、几何统计和账本结构；不以正确率放行。
2. 120题探索性pilot使用12方法：Full、Gen、H、L、H-OBF、L-OBF、G、T、R、R-shuffle、Pseudo-R、Random-matched。每个方法对每题生成一条完整三交接轨迹和一个最终答案记录；G/T/R及其控制器在每一跳重新应用相应选择规则，H/L/OBF/Gen也按其定义运行完整链路。
3. 结果键数为120题×12方法=1440。每批30题是调度批大小，故共4批；不改变每数据组20题，也不是每数据组30题。六个常驻worker跨批处理，避免按批卸载/重载权重。
4. 单跳几何、误差抵消和下一接收者KL作为每个完整方法记录中的逐跳诊断字段，不额外增加答案臂或答案键数。
5. 主要指标是来源读出失真、有效秩、子空间重合、误差夹角、跨来源误差交叉项、下一接收者相对Full的KL/hidden-state差异，以及各方法累计KV字节和接力行为；答案正确率只作探索性描述，并按六个数据组独立计算。
4. 如机制信号稳定，再另行设计新的留出题集做确认性评估。

## 独立统计

- AIME24、GPQA-Diamond、GPQA-nonDiamond、HumanEval+、MedQA、MMLU-Pro分别计算和报告；不把六组汇总成一个准确率或一个主效应。
- 每组内部使用自己的题目作为独立重采样单位；层、头、token和decode位置是重复测量。
- 几何指标、来源读出失真、R/R-shuffle差值、下一接收者KL/hidden-state差值和答案正确率都按组单独汇总。
- GPQA Diamond与non-Diamond始终分开；Diamond不再作为额外独立样本加到GPQA总准确率。
- 若需要展示总体趋势，只能作为六组结果之后的描述性汇总，不作为显著性检验或主结论。

## 资源与启动边界

- 不修改已完成的旧结果、科学锁、原答、队列或账本；
- 不与当前云端seven-methods705运行并发抢占GPU0/1；
- 用户已明确授权启动，实施与零前向预算门槛通过后继续执行，无需重复询问启动许可；
- GPU1优先，单卡最多4、两卡合计最多6个常驻worker；实时显存与前一模型ready决定渐进加载；
- 六题工程键包含于1440目标键，验收后直接复用；worker跨30题调度批持续驻留；
- 本文件和manifest的生成本身没有启动模型推理。

## Frozen OBF baseline provenance

The local frozen OBF dependency is available at `D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_within_question_jive_20261031/runtime`, pinned to upstream revision `36f43183e84d2fbf3b84c9bf4e3e06b7c442a523`. The pinned `HOBF` and `BaseKVCompressor` file hashes are recorded in its `author_manifest.json`. The pilot must use paper-aligned ranks H-OBF=2 and L-OBF=4; the historical local HOBF class default rank8 is not the paper default and must be overridden in the new adapter. No cloud launch is valid until the adapter records the exact source hashes, rank, injection mode and value-protection checks.

The three missing controls are new project code: R-shuffle, Pseudo-R and Random-matched. They must preserve candidate counts, mandatory rows, source-block sizes and deterministic RNG, and must be unit-tested before any model worker is admitted.

## 2026-11-10启动接续

实时SSH核验GPU0/1均空闲；旧HL1410/1410、extension645/645，全部自有manager/worker退出，reserved0。累计actual95582768+unknown61211，115M余量19356021。用户已授权先启动就绪方法，后补其余实现。

论文H/L和OBF保留逐角色新prompt32、保留已有历史/latent/sink的预算；G/T/R和三个匹配控制对历史prompt竞争96、保护最新发送者末8。早期交接字节预算不同，按实际字节报告；同预算机制比较限于G/T/R及三个控制。
