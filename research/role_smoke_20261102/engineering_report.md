# 两题三次工程检查报告

状态：18个结果键齐全，工程检查完成。

- 完成18/18条；真实请求54个，逐RID原始输入/输出与结算记录一致。
- 实际新增位置费用41472；新增unknown=0。历史费用不重新结算。
- 冻结解析器非空输出18/18；触及8192输出上限0条。
- 实测36层×8个KV头，每头130行；按锁定BF16成对K/V维度计算为19,169,280字节，不是网络传输测量。K/V与原位置的gather校验，以及全部latent/sink保护，逐臂执行。
- seed42固定；每题三次重复，三臂轮换顺序。两题来自同一main集合，一题属于Diamond；不把它们当作独立的两套数据集。

设备：NVIDIA GeForce RTX 4090；15个文件SHA、全BF16 CUDA及identity norm通过。模型加载/哈希/初始化用时921.89秒，独立于下表。

| 题ID | 方法 | 次数 | 位置三次一致 | token序列三次一致 | 选择与gather均值(s) | Judger均值(s) | 生成tokens |
|---|---|---:|---|---|---:|---:|---|
| gpqa_main_minus_diamond_0000 | T | 3 | True | True | 2.665 | 71.219 | 1878, 1878, 1878 |
| gpqa_main_minus_diamond_0000 | R | 3 | True | True | 2.444 | 55.407 | 1505, 1505, 1505 |
| gpqa_main_minus_diamond_0000 | R-shuffle | 3 | True | True | 2.729 | 38.245 | 1094, 1094, 1094 |
| gpqa_diamond_test_147 | T | 3 | True | True | 2.482 | 34.185 | 975, 975, 975 |
| gpqa_diamond_test_147 | R | 3 | True | True | 2.469 | 42.493 | 1181, 1181, 1181 |
| gpqa_diamond_test_147 | R-shuffle | 3 | True | True | 2.671 | 90.500 | 2462, 2462, 2462 |

计时范围：CUDA两端同步；选择与gather是包含目标诊断及标量回读的外层压缩时间，模型审计hook仍在。必须的probe计入成本，并由三臂真实共享一次。每臂重建总时间不是独立端到端实测；不同答案长度不作为方法速度胜负。显存峰值字段按每次重复累计。

这轮不作准确率优劣、显著性、创新有效性或加速比结论；未运行Full/OBF/JIVE/旧receiver新答案，未启动全量新实验。

完整归档SHA256：`df93680182551bd4f14f92dc18298e9cd34df799808d13b1ffb06f34e2d9bb8b`；82个成员。云端流SHA、gzip CRC、tar精确集合、逐成员及安全全解包SHA全部通过。私有题文/原答/tokens不进入仓库。
