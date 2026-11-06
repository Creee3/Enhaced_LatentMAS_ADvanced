# 记录

2026-11-06：用户明确“不跑R-shuffle，其他三个补一下”，范围T/R/G225。旧main448/G/两题pilot保持冻结，不热改旧锁，不清零累计账本。新GPU1最多4worker，初始仍须util<=30/free>=70%，后续自有驻留模型允许利用率豁免，剩余显存至少21GiB/待加入worker且上一个加载完成。21GiB来自试跑约16GiB分配峰值加KV/碎片余量的资源保护，不改科学参数。失败不自动重载，保留已有键与费用。

实际新run TRG225_a100_s42_l10_20261106T155639Z，锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f。32文件+bundle37校验、8个核心函数AST与试跑一致、T/R/G选位分派CPU检查及671缺失键/零forward预检通过；保守上界7087329，启动预算74917051/0/58106。

GPU1逐次加载4worker完成，真实UID1000/start_ticks/argv/CVD1均独立核验，模型15文件SHA、BF16CUDA16381470720B、identity及target norm1.3758320808410645全部通过，现场无失败。首两个worker分别已有1条R新提交，raw RID/tokenSHA、36x8x130位置及原KV/保护行通过，但尚无本attempt全部T/R/G，不称正式三标签worker。两道pilot复用题的R均严格匹配pilot前缀位置并提交，pilot4条不重跑。

最新启动验收6/675唯一（新2+复用4），2case complete/4running/219pending/0held；实际账本74932122/reserved33919/unknown58106，在途预留正常不手工结算。一次完整UTF8 inspect私有落地；后续仅轻量真实进程/计数用于启动检查。startup_verified.json两地保存。监控gpqa-448已更新ACTIVE、每40分钟、仅新TRG225run；旧所有run/原始答案不改。
