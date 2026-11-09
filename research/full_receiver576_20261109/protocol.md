# Full与原receiver同题576题补跑

用户授权只增加Full、旧receiver，沿用同一questions.json SHA，GPQA96（Diamond48/非Diamond48）、MMLU-Pro280、GSM8K200，共1152新键。旧GGG/RRR/TTT1728键冻结，只读作五方法同题参考。

Qwen3-8B15权重SHA、BF16/norm_only、identity与mean-input-L2、seed42/latent10、cap8192/context32768、temperature.6/top_p.95/top_k20、作者模板及上一轮任务适配/程序解析器不变。Full每次交接保留全部原KV与原RoPE。旧receiver逐字节调用冻结stage_ops.compress(receiver_attention_sink_recent)，仅当前发送方新prompt选32（末8强制），此前行/latent/sink保留，典型三次46/88/130行；不是RRR的110/120/130，报告不混淆总预算。

同题只计算一次Planner，记录KV SHA和分叉RNG；两臂恢复同一起点，独立上游路径。Full无接收者probe，原receiver各交接必要probe都计成本。每题4+7-1=10真实RID，整体5760；工程三题6键30RID包括在整体中，不额外重跑。计时CUDA两端同步，重建单臂时间明确标记；峰值每题累计。原K/V gather、每层每头位置与storage indices存npz、实际字节和latent/sink原位置独立审计。

工程先用最终清单index0/96/376（每题集1题），三个数据集至少一个程序可解析输出，不以正确率或准确率选题/放行。工程审计与完整本地流归档读回验证后，首遍6键复用再正式新增1146键。

GPU0/1共最多6worker、每卡最多3本次新分配（用户每卡最多4总界限内），独立BF16模型与私有MPS；首份fresh util<=30/free>=70%，后续自有模型下free>=21GiB，上个ready才加。动态cpuset分配。取消上一轮仅管理层600秒加载抛错；模型生成原1800秒超时不改，STOP仍检查。不盲重载，异常先保留现场和费用、全面本地归档核验，有明确修复/实证资源改善才恢复真实缺失键。预算原共享actual+reserved+独立unknown<=115M，无手工健康结算或额外硬调用限。

完成须1152精确键、无held/pending/running、自有进程/MPS退出和自有RID0，旧参考快照SHA完好；完整归档远端流SHA/gzipCRC/tar精确集合/逐成员/安全全解包读回通过后才冻结程序评分。分数据集和GPQA子集报告正确数/分母、Wilson区间、程序无效、cap、KV实字节，五方法按同题描述；全部新增配对检验探索性，不挪用上一轮预登记显著性。跨runPlanner SHA匹配数单独报告，不保证历史逐token相同，不称共享RID。无R-shuffle不主张角色边界因果特异性。不作中途准确率/AI语义审计。重要中文UTF8两地记录、本地Git不push，每小时巡检完成后删除。
