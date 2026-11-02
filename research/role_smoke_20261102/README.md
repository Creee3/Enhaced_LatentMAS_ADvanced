# 两题三次工程检查

这是2026-11-02用户明确授权的小规模试跑，不是全量实验，也不计算方法胜负。

- 两个固定GPQA题ID：main顺序中第一个非Diamond条目和第一个Diamond条目；不看答案选题。
- latent10，seed42。每题重复三次，每次重新建立缓存。
- 每次前两个交接沿用冻结receiver；最后一个交接跑T、R、R-shuffle，三次重复轮换执行顺序。总计18次末端生成。
- 三臂共用一次真实前缀及接收方probe；每次Judger前恢复相同CPU/CUDA RNG状态。重复用于发现选择/生成不稳定，不形成六个独立效果样本。
- 保护全部30latent及4sink，每层每KV头保留96个prompt行，最终36×8×130位置，原K/V gather，不回填V。Refiner最后8个prompt位置强制保留，其他候选注意力预选，候选池不超过192。

T匹配总注意力输出；R匹配self/sink和三个来源块的输出，目标是四块均方误差的平均。R-shuffle用等大小、错移连续边界的假来源块，保留self/sink标签。三个方法从完全相同的当前可用缓存选择，不能把这个局部teacher称为历史Full基线。

每个KV头的四个GQA查询头×最后16个prompt query作为64行独立观测。后向贪心删除使用共同归一化分母的闭式损失，双精度求值；每头保持相同长度，实际位置可不同。

计时在CUDA同步前后包围实际调用，记录共享前缀、必须的probe、选择与gather、Judger、独立张量校验和位置文件I/O。单臂总时间是共享部分与单臂部分的**重建和**，不是独立实测端到端延迟。原模型审计hook仍在，不能与旧数据直接作加速比。

模型、prompt、解析器和共享账本复用原冻结依赖；模型加载与SHA检查单独记时。所有真实forward走原Engine账本，包含probe。题文、原答、tokens及完整日志只放独立私有实验目录，不进入代码仓库。

运行CPU检查（不调用云端）：

```powershell
python -m unittest discover -s research/role_smoke_20261102 -p test_block_selector.py -v
```

`deploy_smoke.py`是显式启动入口，只有执行该命令才有云端副作用；发现已有active会拒绝重复启动。`capture.py`只读取一次状态。`finish.py`要求自有进程已退出，再完整归档、校验gzip CRC、tar精确成员集、逐成员及全安全解包SHA，并检查18键与RID唯一结算。

GPU池0–7，fresh util≤30%且free/total≥70%才准入；本轮最多一个worker。失败不自动重启，不触碰他人进程，不新增监控或启动大实验。结果完成后再更新工程报告。
