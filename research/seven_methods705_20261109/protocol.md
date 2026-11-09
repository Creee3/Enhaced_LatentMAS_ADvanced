# 已授权七方法705题补齐协议

固定同题GPQA225（Diamond106/非Diamond119）、MMLU-Pro280、GSM8K200，七方法4935键。既有五方法576题2880原答只读复用；新增H/L705题1410、五方法GPQA129题645，共2055。没有R-shuffle、OBF回填、核范数、其他seed或latent40；不能将旧末次T/R/G225答案作为全交接结果。

Qwen3-8B15文件SHA/BF16/norm_only、identity与mean-input-L2、seed42/latent10/cap8192/context32768、temperature.6/top_p.95/top_k20、原模板与task_adapter程序解析器均沿用。H/L作者3源固定36f43183e84d2fbf3b84c9bf4e3e06b7c442a523，仅适配缓存构造，完整原函数执行且独立注意力topk对照；H每头、L层内共用位置，每角色新prompt32，无末8强制，无回填；全部历史/latent/sink保留，典型46/88/130。原receiver新32+末8，GGG/RRR/TTT共同96prompt+末8并保护latent/sink，为110/120/130。不同早期预算如实报告。

HL仅共享Planner与其latent注意力，2×4−1=7RID/题。extension五方法共享Planner，G/T/R额外共享首probe，共26RID/题。各分支恢复共同fork RNG，各自继续上游；不同子run实际重算prefix分别计费，不声称跨run调用共享或生成逐token一致。原K/V gather与原RoPE、全部latent/sink保护逐层逐头验证，位置文件36×8×长度与实际字节匹配。Timing沿CUDA双端同步，必要probe计成本，重建单臂耗时注明非独立实测。

两个子run串行共用总6worker上限，GPU0/1每卡3；共享全局manager锁，任一阶段其他run不能并行启动。首次fresh util<=30/free>=70%，自有ready后free>=21GiB且上一个ready才加载；动态CPU亲和，私有MPS，不碰他人及GPU2/3。每卡阶段最多4槽位尝试，无盲重载，1800秒原生成超时不改。只按实际UID/start_ticks/argv/CVD认领。

全局budget.lock/cumulative_budget.json actual+reserved+unknown<=115M，两个run合计预检最坏106987494位置，零forward。probe/失败真实计费，健康预留不手工结算、不重复结算、不清零迁移。新增故障保留全现场及费用，完整本地流归档远端SHA/gzipCRC/精确成员/逐成员/安全全解包读回后，有明确修复或实际资源改善才补真实缺失方法，不重排complete。

各run先三题工程（HL6键、extension15键），程序解析与结构/费用/退出审计和完整归档通过后复用首遍，正式补剩余，不以正确率放行。正式全精确键/零held/pending/running/全退出/MPS停/自有RID0，原始归档通过后才评分；两个run原始归档均通过后冻结score一次合并4935。五题集范围（GPQA225/MMLU280/GSM200及GPQA两子集）分别报告正确率、实际KV与Full比、去重RID成本、跨run Planner SHA匹配数及探索性配对统计，不把不显著当等价、无R-shuffle不声称角色边界因果。最后最终归档，小时监控删除。真人暂停立即停止本scope自有运行及监控，明确恢复才继续。
