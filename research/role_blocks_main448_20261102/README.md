# main448角色块固定比较

管理接续：2026-11-03T03:25Z，原manager650899因读取退出中worker的/proc/environ产生PermissionError退出。完整故障归档核验后，以manager_recovery.py管理包装入口保守处理无权限/异UID进程并委托原manager.py，接管健康驻留worker、不重启模型。原16文件科学锁和worker源码不改。具体PID、缺失键单次派发与费用见progress及management恢复记录。

2026-11-02用户授权启动完整实验，最多六卡。

范围：同一GPQA-main448，包含Diamond198；seed42、latent10。T总读出匹配、R真实四块读出匹配、R-shuffle等大小错位连续边界，各448个唯一答案，共1344。完整复用两题试跑第一遍6条，新增446题1338条；后两遍只作工程证据，不当新增样本。旧Full/HeadOBF/JIVE/receiver的latent10共1792条只读复用，最终七方法主集3136键；Diamond子集1386键，不另生成。latent40不在本轮范围。

科学实现沿用通过试跑的版本：Qwen3-8B BF16/norm_only、seed42、context32768、cap8192、temperature.6/top_p.95/top_k20、作者prompt与冻结程序解析器。前两个交接用原receiver，末次从仍可用prompt中按相同bank选96行，保护30latent+4sink，每层每KV头130行，GQA四查询头×16prompt query，原K/V gather无V回填。R包含self/sink块；假角色只更换来源分组。没有新增分解、超参扫描、跨题库、AI评分或新seed。

`build_worker.py`产生新队列入口，并逐函数AST验证stage/probe/judge/final_select/validate及同步timer与试跑相同，选择器逐字节相同。新版本仅改变排队/持久提交、单次重复、结果元数据和实际KV张量字节计数。正式六条复用的源题文、源码、token/位置三次一致性及父归档SHA在prepare中核对。

唯一云端入口`manager.py`。GPU0–7候选，最多同时6个自有worker，一卡完整模型；每10秒逐卡fresh util≤30%且free/total≥70%准入。健康worker不迁移；每卡仅自动尝试一次，真实容器PID/UID/start_ticks/argv/CVD确认所有权，不碰他人进程。旧GPU4/5加载失败的特定历史证据需单独清除：GPU4不因相同24081MiB快照盲试，GPU5需恢复到已有证明的24081MiB级可用资源。它们仍在候选池，故障规则不是新的全局资源门槛。

所有模型forward（包括必需probe及失败）计入既有共享账本，actual+reserved+独立unknown≤115M；不迁移、清零或重复结算。部署前使用冻结prompt和真实tokenizer计算446题的精确最大新位置上界及original context界限，不进行模型forward。失败只冻结对应任务，保留已提交答案；不得自动重试，先完整本地归档核验，再凭明确修复/资源改善恢复缺失键。

科学主指标为冻结程序正确率与实际KV保留字节；生成耗时辅助记录，不作为方法效果判定依据。预登记主检验为R对T的双侧精确McNemar，α=.05；声称真实角色边界有收益还须R优于R-shuffle并通过同类双侧检验。Diamond是同一批题的预定子集描述，不把它当独立重复；多方法比较不声称全部确认性显著。main包含既往开发题，不叫新盲测。压缩字节不等于实测通信/GPU/FLOP收益。

计时保留试跑口径：CUDA两端同步，模型加载单列，选择器外层包含诊断，必需probe计入成本，共享prefix/probe实际只算一次；单臂总时间明确是重建值。显存peak字段按每题累计。无需重跑旧基线来补计时，不能据旧硬件/旧计时直接宣称加速。

本地私有包`advanced_MOC/experiments/latentmas_role_blocks_main448_20261102`保存cloud_active、原答快照和完整inspect；代码仓库不含题文/原答/日志/账本。`capture_current.py`一次性保存完整UTF8 inspect，只输出覆盖、真实进程和异常，不输出未完成准确率。

全部1344新方法覆盖、旧1792快照完好、无held/failed缺失键、pending0、全部自有进程退出且自有RID预留0后，完整本地归档本run、六条父复用与旧四臂来源、bundle37/模型SHA锁（不复制权重），验证远端流SHA/gzipCRC/tar精确集合/逐成员及安全全解包SHA。随后按冻结解析器在main448与Diamond198分别汇总七方法正确率、KV保留/减少比和RID去重费用。重要记录UTF8同步两地管理目录，本地Git提交不push。沿用每30分钟巡检，全部完成后删除本监控；用户暂停立即停止自有运行和自动推进。
