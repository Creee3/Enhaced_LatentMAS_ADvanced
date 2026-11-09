# 三篇论文的基线核对与RCLC源码下载

核对日期：2026-11-09，北京时间。范围仅为用户指定的三篇论文正文、附录及官方仓库，不是系统性综述。源码下载版本与文件校验另见 download_receipt.json。

下载已完成，两份源码快照均已在云端逐文件读回SHA核验，gzip CRC、归档SHA和精确成员集合通过：

- 当前接口版：`/home/ubuntu/ICML2027/reference_repos/rclc-official_c9cbefaf54df`，固定提交 `c9cbefaf54dfd8f13fdc952143698c4aab99f768`，102文件。
- 论文复现版：`/home/ubuntu/ICML2027/reference_repos/rclc-paper-v1_15bc3f7a2f93`，官方 `paper-v1` 固定提交 `15bc3f7a2f93229b11ea982a534ce21528552fbc`，335文件；全部下载文件另与Git提交blob ID核对。
- 本地源码在 `D:/liuxin_workspace/ICML2027/04_参考代码与历史实验/上游参考代码/rclc-official` 与同级 `rclc-paper-v1`。该参考代码目录在本机解析到 `coding_base`。
- 云端为可直接阅读的源码快照，附 `SOURCE_SNAPSHOT.json`；模型权重、数据集压缩包和大型演示媒体不在本次代码包内。没有安装依赖或运行模型。

## 论文与官方来源

| 论文 | 固定版本 | 官方仓库 |
|---|---|---|
| Receiver-Conditioned Latent Communication gives 94% CacheBack | [arXiv:2609.32046v1](https://arxiv.org/html/2609.32046v1)；Maximillian Rossi等，与用户给出的标题及作者一致 | 用户地址 https://github.com/maxr0ssi/rclc 当前跳转到 https://github.com/agentcacheback/cacheback |
| Latent Collaboration in Multi-Agent Systems | [arXiv:2511.20639v4](https://arxiv.org/html/2511.20639v4) | https://github.com/Gen-Verse/LatentMAS |
| When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration | [arXiv:2604.13349v3](https://arxiv.org/html/2604.13349v3)；Yiping Li、Zhiyu An、Wan Du | https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay |

论文身份直接与arXiv官方标题、作者、版本及其代码链接交叉核对。原文HTML私有留存，SHA见source_manifest.json。没有新增BibTeX条目，也没有执行模型推理。

## 1. RCLC / CacheBack

研究问题：分布在不同agent上的长文档证据，如何按接收者需求选成较小消息。端到端任务为FanOutQA并行fan-in，以及LongBench v2顺序文档链。

### 端到端对照：正文4.1—4.3与附录E

| 对照 | 定义 |
|---|---|
| Same-size text | 与CacheBack同模型尺寸的发送者，根据接收者问题生成文字报告 |
| Smaller-sender text | 同模型家族内用更小发送者生成文字，接收者固定为该家族最大模型 |
| Full rows control | 上下文容许时发送全部位置，使用和CacheBack相同的表示；不是逐层原KV直接传输的Full-KV基线 |

Full rows在文中明确报告于FanOutQA的Ministral/Gemma上下文可容纳配置。不同压缩率及Relative/Fixed预算为CacheBack配置变化。

### 选择器对照：正文3.2、附录D.1；13题开发诊断的证据召回

| 选择器 | 论文中的适配方式 |
|---|---|
| StreamingLLM | 按位置新近性排序 |
| H2O | 累加发送者因果prompt注意力 |
| ChunkKV | 用末32个prompt位置观察，宽5平均窗口，按16位置chunk聚合 |
| KVzip | 用上下文重建过程的注意力评分 |
| Mean query attention | SnapKV式接收者query适配，跨query/head聚合并做局部池化；CacheBack无集中支持修正的对照 |
| CacheBack（论文方法） | 在mean query attention上加query局部集中支持修正，默认p=alpha=2 |

以上选择器比较不能全部称作完整任务准确率/延迟基线。另有unrelated-query对照、修正参数敏感性，以及消息编码比较。附录B讨论OBF的任务设定差异，没有把OBF列入相同端到端实验表；LatentMAS、KVComm、C2C等在相关工作出现，不等于已作直接实验对照。

重要实现区别：附录C.1产生跨头、跨层共享的位置排名；C.3明确实测传连续行，在接收端fresh prefill，使用新位置编码。没有实测原始逐层KV直接迁移。Token-ID加latent连续向量是进一步编码比较，不是所有端到端实验都使用的编码。

## 2. LatentMAS

研究问题：把多智能体的推理与通信从文字切换到latent，是否改善准确率、输出长度和推理时长。

正文4节Models and Baselines明确列出：

| 基线 | 定义 |
|---|---|
| Single | 单个LLM正常自回归解题 |
| Sequential TextMAS | 顺序角色链，使用文本推理和文本通信 |
| Hierarchical TextMAS | 层次结构的领域agent与汇总者，使用文本推理和通信 |

LatentMAS是被评估的方法本身，不是原论文中的一种压缩器。附录E.3另比较两种混合消融：Latent Reasoning + Text Communication（传最后128个解码token）；Text Reasoning + Latent Communication。latent步数、对齐方式等属于组件/配置研究，不能与上述主基线混列为不同公开算法。

评估覆盖GSM8K、AIME24/25、GPQA-Diamond、MedQA、ARC-E/C、MBPP+、HumanEval+。Sequential与Hierarchical是组织结构，Single/TextMAS/LatentMAS是相应比较维度。

## 3. OBF论文：When Less Latent Leads to Better Relay

研究问题：在LatentMAS顺序链中压缩交接KV，并用正交残差回填减轻删除损失。正文5.1相比方法如下。

| 主表方法 | 身份与定义 |
|---|---|
| Full | 完整历史KV交接基线 |
| LMAS-Attn (H) | 每层每头独立按当前latent推理状态的注意力选prompt，硬删除，无回填 |
| LMAS-Attn (L) | 每层先跨头汇总注意力，再使该层各头共享位置，硬删除，无回填 |
| LMAS-Sink (Gen) | 保留sink及生成的latent KV，不作prompt位置选择 |
| H-OBF | 论文方法：H选位后对保留V作正交低秩残差回填 |
| L-OBF | 论文方法：L选位后作同类回填 |

附录E表7另外有：Cache Merging（CaM启发的自定义合并）、Full + int8、Full + int4、L-OBF + int8。文中明确合并与量化不是直接运行CaM/KVQuant原代码，应称改编基线。

附录D组件消融为No Projection、Max-P、No Scaling、Naive Aggregation；替代设计为Attn-inject、Per-token、EVR-adaptive。预算与rank扫参、fast SVD属于配置/实现研究。

正文v3固定H-OBF rank2、L-OBF rank4。我们早期项目的HeadOBF冻结rank8，是已有项目配置，不能称作逐项复现当前论文默认设置。

## 与当前实验的关系

1. LatentMAS回答“latent协作相对文本/单模型怎样”；OBF回答“同类latent链上的KV怎么压缩与补偿”；RCLC回答“按接收者需求传哪些长文档证据及怎样编码”。
2. 当前Full是直接原KV参照；当前G/T/R按层、头选原KV，保留原位置RoPE；与CacheBack的共享位置排名和重新prefill不同。
3. 当前G与Mean query attention有接收者注意力选位这一共同方向，但query采样、跨层/头聚合、池化、span预算与表示传输不同，不能视为同一个实现。
4. 题内JIVE、核范数、当前T/R及旧receiver是我们项目的历史实现/候选，不是这三篇论文都运行过的公共基线。
5. 本次仅下载参考代码并核对文献，没有启动新增实验、安装包或修改正在运行的模型环境。
