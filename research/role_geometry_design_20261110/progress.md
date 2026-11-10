# 设计进度

2026-11-10：读取research-ideation、planning-with-files和仓库AGENTS。新设计位于独立目录，不修改正在运行的705题源。

2026-11-10：核对冻结Full/GGG/RRR完整题集统计：GPQA GGG比Full高10题（配对精确p=.0129），RRR高9题（p=.093）；G/R只差1题。MMLU-Pro RRR比Full低1题、GSM8K低1题。结果支持追踪GPQA正向信号，但不支持普遍超过Full的结论；跨运行比较均探索性。

2026-11-10：离线扫描576题正式归档共1728条答案，得到三交接C指标：TTT每跳较高；RRR比GGG在第2、3跳低，第1跳高。补充限制：T的目标直接最小化整体误差，所以TTT的高C是目标内生诊断；第一跳block0还包含固定任务前缀，不能说成角色间抵消；后跳轨迹已分叉。确认旧归档无原始K/V或逐来源误差向量。

2026-11-10：完成研究问题卡与分阶段方案。无新模型推理、无源/队列/锁/账本/原答修改；执行只需后续单独授权。

2026-11-10：核对本地先例与论文小规模诊断。CacheBack 13题是满足证据可定位条件的selector开发集；LatentMAS/OBF主结果是benchmark套件，非20题烟测；本地先例为4题结构工程试跑。记录候选pilot为3题工程+GPQA/MMLU-Pro/GSM8K各20题SHA分层探索样本；dataset_selection_gate.md标记尚待用户锁定，未启动实验。

2026-11-10：用户将范围改为AIME24、GPQA-Diamond、GPQA-nonDiamond、HumanEval+、MedQA、MMLU-Pro，排除GSM8K。按salt `role_geometry_pilot_20261110_v1`与`sha256(salt|dataset_group|stable_id)`生成6×20=120题私有manifest；MedQA使用源行号+idx+pid解决重复索引，全部120 stable ID唯一。当前云端其他实验仍占用GPU0/1，未启动新pilot。

2026-11-10：将本次规则完整写入`pilot_protocol_lock.md`。明确六题工程检查→120题探索pilot的顺序、G/T/R/R-shuffle/伪来源对照、连续机制指标为主、答案只作描述性、当前云端运行释放前不抢占GPU且需单独启动授权。

2026-11-10：用户确认120题上按数据集独立计算。协议已补充六组分别汇总几何指标、KL/hidden-state差值、R/R-shuffle差值和答案正确率；不做池化主效应或池化显著性检验，GPQA Diamond/non-Diamond继续分开。

2026-11-10：用户指出主方法应是RRR而不是单次R。已修正协议：局部G/T/R/R-shuffle只做机制诊断；120题链级主实验运行真实GGG/RRR/TTT，必要时增加三跳RRR-shuffle作为角色边界空对照。链级结果仍按六个数据集组独立统计。

2026-11-10：用户要求在诊断实验中跑多方法并找关键机制。新增`diagnostic_matrix.md`：局部层扩展为Full/Gen/H/L/G/T/R/R-shuffle/Pseudo-R/Random-matched；链级主表改为Full/Gen/H/L/H-OBF/L-OBF/GGG/RRR。重点寻找“角色几何→边界特异性→下一接收者失真→链级收益”的证据链。

2026-11-10：用户指出原论文的H-OBF/L-OBF也必须出现在诊断层。已补入局部矩阵，并明确它们继承H/L选位、只修改保留V的OBF回填；这样可以拆分选位几何贡献与value recovery贡献。

2026-11-10 16:16：12法全量原生tokenizer零forward上界14946801，总累计上界110590780<115M；作者HOBF/LOBF/Gen均同固定commit源、三跳BF16张量和新控制测试通过。云端判分器NumPy依赖已适配，20负例拒绝，19/20参考正例通过，HumanEval/32原HF脚本参数TypeError在定位；尚无GPU前向或新答案。

2026-11-10 16:47：用户启动授权已执行，实际run位于role_geometry120_s42_l10_20261110T073338Z；代码research/role_geometry_run_20261110。六组120题十二完整方法已进入工程队列，GPU1依次四模型、GPU0最多二，总六。旧design/controls.py是未接入GPU的早期CPU草稿，正式controls/diagnostics以新run锁为准。
