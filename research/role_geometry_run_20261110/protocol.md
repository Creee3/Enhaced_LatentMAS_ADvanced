# 120题角色几何与接力诊断协议

用户授权范围：AIME24、GPQA-Diamond、GPQA-nonDiamond、HumanEval+、MedQA、MMLU-Pro各20题；固定选择清单SHA7756c583be41780cd60ae9d88b04a562d99246069d85e654e22982ccb1f344e5。

12完整链级方法：Full、Gen、H、L、H-OBF、L-OBF、G、T、R、R-shuffle、Pseudo-R、Random-matched。每方法每题三次交接并生成最终答案，120×12=1440键。六题工程来自120题，72键复用；队列每30题一批，共4批。

GPU1优先4worker，GPU0最多2worker，总上限6。模型常驻跨工程/正式与调度批。依次加载、每份新模型前余量至少21GiB，首份卡利用率<=30且free>=70%。GPU2/3不使用。BF16、模型15文件SHA、identity/norm_only/mean-input-L2，seed42/latent10/cap8192/context32768、temperature.6/top_p.95/top_k20、原KV/RoPE保持。

论文六法依赖固定commit36f43183e84d2fbf3b84c9bf4e3e06b7c442a523，运行作者原函数，仅缓存构造import适配；H-OBF rank2、L-OBF rank4，uniform/ad_over_as/no-centering。H/L及OBF每发送角色新增prompt32，保留已有history、全latent、4sink，常见46/88/130。Gen仅4sink+累积10/20/30latent，即14/24/34；Full保留所有原位置。

G/T/R及三个空对照竞争全历史prompt96，保护4sink、所有latent与最新发送者末8prompt，常见110/120/130。H/L与G/R早期预算不同，报告实际字节，不称等预算。新空对照使用独立CPU RNG，G/R/T沿用冻结bank<=2B及固定原KV gather。R-shuffle保持每头来源计数和block0、随机打乱成员归属，拒绝仅重命名；单来源退化显式记。Pseudo-R在来源内相对prompt位置四分位匹配bin内置换，保持每bin标签计数；它不是全局绝对年龄匹配。Random-matched匹配G的来源×上述bin保留数，强制与保护行相同。

每题共享一次Planner与第一receiver probe，所有方法从相同KV/RNG分支，之后各走三跳轨迹。每方法7个RID引用，去重链级62RID；Full状态在后两跳各对11个压缩法额外receiver prefill，共22RID。因此每题84去重请求，总10080；84是调用数，费用按actual_positions真实计，不按调用数替代。全量原生tokenizer最坏14946801位置，总计上界110590780<115M。

诊断：每方法每跳保存prompt/latent分开的K/V参与率、熵有效秩、top4能量、特征子空间重合；每来源等数量按相对位置bin抽样最多64，配伪来源空对照。读出诊断保留真实来源误差、交叉项矩阵、夹角、抵消指标与代数恒等校验；不保留原始K/V。Full完整轨迹提供公共状态的12方法局部选择与后两跳receiver KL/hidden差异；各链另记相对其自身删除前状态的receiver偏差，两种分析明确区分。receiver采样位置是prompt末token产生的首个next-token分布，native prefill直接hook，不增加生成。

计时分原生阶段、prompt prefill、latent/decode、required probe、geometry、selector/gather、额外公共状态probe。仪器开销不称生成加速，任务吞吐另算。每方法实际KV字节与Full原位置分母逐跳记录。

六组独立统计，题目是重采样单位，层/头/位置不是独立N。无中途正确率。完整1440键、结构/源SHA/rawRID/token/原KV位置/保护/费用核验、manager/worker/MPS退出、自有预留0，先完整原始归档并验证远端流SHA/gzipCRC/精确成员/逐成员/安全全解包读回，再最终评分并最终归档。HumanEval按冻结测试与显式测试适配器在Landlock/seccomp子进程判分；不将gold/test/canonical放入prompt。所有比较为探索性。

故障只处理自有失败worker，停止现场与费用先完整归档核验；不盲重载、不重排complete、不清零/迁移/手工健康结算。源锁冻结后不热改；同一故障恢复需新的明确证据。用户明确暂停立即停止本run自有运行。
