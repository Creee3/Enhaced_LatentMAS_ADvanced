# 记录

2026-11-06：用户明确“不跑R-shuffle，其他三个补一下”，范围T/R/G225。旧main448/G/两题pilot保持冻结，不热改旧锁，不清零累计账本。新GPU1最多4worker，初始仍须util<=30/free>=70%，后续自有驻留模型允许利用率豁免，剩余显存至少21GiB/待加入worker且上一个加载完成。21GiB来自试跑约16GiB分配峰值加KV/碎片余量的资源保护，不改科学参数。失败不自动重载，保留已有键与费用。

实际新run TRG225_a100_s42_l10_20261106T155639Z，锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f。32文件+bundle37校验、8个核心函数AST与试跑一致、T/R/G选位分派CPU检查及671缺失键/零forward预检通过；保守上界7087329，启动预算74917051/0/58106。

GPU1逐次加载4worker完成，真实UID1000/start_ticks/argv/CVD1均独立核验，模型15文件SHA、BF16CUDA16381470720B、identity及target norm1.3758320808410645全部通过，现场无失败。首两个worker分别已有1条R新提交，raw RID/tokenSHA、36x8x130位置及原KV/保护行通过，但尚无本attempt全部T/R/G，不称正式三标签worker。两道pilot复用题的R均严格匹配pilot前缀位置并提交，pilot4条不重跑。

最新启动验收6/675唯一（新2+复用4），2case complete/4running/219pending/0held；实际账本74932122/reserved33919/unknown58106，在途预留正常不手工结算。一次完整UTF8 inspect私有落地；后续仅轻量真实进程/计数用于启动检查。startup_verified.json两地保存。监控gpqa-448已更新ACTIVE、每40分钟、仅新TRG225run；旧所有run/原始答案不改。

2026-11-06T16:38Z首轮40分钟巡检：一次完整inspect私有落地，20/675唯一（新16+复用4），剩655；5complete/4running/216pending/0held，无重复越界，源及参考/复用快照SHA通过。manager28169与四真实worker健康。四worker独立新attempt核验：28172的8条与28985的4条已覆盖T/R/G并正式通过；28440检查3条已有R/G、待T，28695检查2条已有T/R、待G，不能冒称全部正式。预算capture74997837/reserved33899/unknown58106，健康在途预留正常，无手工或重复结算，无新故障、未算未完成准确率。证明及中文接续两地保存，监控继续ACTIVE40分钟。首次本地管理记录命令引号产生SyntaxError，未执行任何科学操作；改用here-string后保存成功。

2026-11-06T17:18Z巡检：一次完整UTF8 inspect私有落地，39/675唯一（新35+复用4），剩636；11complete/4running/210pending/0held，源及全部参考/复用快照SHA通过，无重复或越界。manager28169及四worker真实身份与启动一致，无新故障。28440的8条本attempt新答、28695的5条分别完成T/R/G全部独立核验，raw RID/token SHA、36x8x130位置/保护行/实际KV及native位置通过；至此四worker均已正式三标签核验（此前28172和28985已通过），两地证明保存。账本actual75097100/reserved34189/unknown58106，健康在途预留正常，无手工或重复结算，无未完成准确率。继续ACTIVE40分钟，常规健康进展不打扰。

2026-11-06T17:58Z巡检：一次完整UTF8 inspect私有落地，59/675唯一（新55+复用4），剩616；18complete/4running/203pending/0held，无重复越界，源锁fa21c45d及所有参考/复用快照SHA完好。manager28169及四正式worker28172/28440/28695/28985健康，各提交15/16/10/14，均较上轮推进，无新故障。实际RID预留归属与真实存活worker/本run一致；账本75195247/reserved34260/unknown58106，在途正常，无手工/重复结算，无未完成准确率。中文管理接续两地保存，原科学锁和暂停run不变，ACTIVE40分钟继续。

2026-11-06T18:38Z巡检：一次完整UTF8 inspect私有落地，80/675唯一（新76+复用4），剩595；24complete/4running/197pending/0held，无重复越界或新故障。源锁及所有快照SHA完好，manager28169与四正式worker真实身份健康，各提交18/21/17/20；RID预留均属于真实自有在途worker。账本75307948/reserved34920/unknown58106，无手工结算，继续ACTIVE40分钟，两地接续保存。

完成阶段final_summary.py已在b974247冻结SHA两地保存；没有读取/计算当前正确率。须完整675精确键、全部队列完成/进程退出/预留0及归档通过，才调用冻结作者解析器复算。225/D106/nonD119分别报告，仅all225 R对T是预登记primary，子组和G比较均探索性；无shuffle不主张角色边界因果。KV是实际保留字节对原位置未压缩参照的加权比，费用按新run/pilot/前置A100失败的settled RID去重并对账本增量。4项合成测试与30源/3584公开快照复核通过，0模型调用。首次管理记录误用inspect basename导致远端FileNotFoundError，未改科学文件；改用active绝对路径后保存成功。

2026-11-06T19:18Z巡检：一次完整UTF8 inspect私有落地，109/675唯一（新105+复用4），剩566；35complete/4running/186pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交25/27/26/27。实际RID预留均归属本run存活worker；账本75419606/reserved34080/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T19:58Z巡检：一次完整UTF8 inspect私有落地，140/675唯一（新136+复用4），剩535；45complete/4running/176pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交33/33/34/36。实际RID预留均归属本run存活worker；账本75538998/reserved34718/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T20:38Z巡检：一次完整UTF8 inspect私有落地，165/675唯一（新161+复用4），剩510；54complete/4running/167pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交38/43/37/43。实际RID预留均归属本run存活worker；账本75653350/reserved34229/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T21:18Z巡检：一次完整UTF8 inspect私有落地，188/675唯一（新184+复用4），剩487；62complete/4running/159pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交42/49/42/51。实际RID预留均归属本run存活worker；账本75750767/reserved34128/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T21:58Z巡检：一次完整UTF8 inspect私有落地，211/675唯一（新207+复用4），剩464；69complete/4running/152pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交52/52/47/56。实际RID预留均归属本run存活worker；账本75856360/reserved25734/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。
