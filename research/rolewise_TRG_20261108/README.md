# 全交接 GGG / RRR / TTT

用户已确认96 GPQA＋280 MMLU-Pro＋200 GSM8K，最多6个worker。仅此新run；旧实验与原答冻结，不运行混合臂或Full。详细协议见protocol.md，进展见task_plan.md/progress.md。

代码不含题文、选项、gold、原答或模型权重。私有状态唯一入口：`D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_rolewise_TRG576_20261108/cloud_active.json`。预算使用旧共享115M账本，不清零。原生生成/计费/位置hook依赖已冻结32文件和bundle37；新角色选位代码单独版本化。

- `prepare_data.py`：官方固定revision与SHA、按ID哈希选题；只做数据准备。
- `deploy.py prepare`：仅CPU机械测试、原生tokenizer上界和源锁，不启动模型；已经开始的run拒绝覆盖。
- `orchestrate.py --poll-seconds 60`：**显式执行入口**。执行已授权工程阶段，独立审计与本地全归档后进入正式阶段，最终退出/归档/冻结评分。该进程是一次实验执行器，不是定时自动化，不创建app任务。发生新故障保留现场、停止新增加载，不盲重试。
- `inspect_current.py`：只抓一次实时状态，完整私有落地，stdout仅计数/真实进程/异常，不输出原答或未完成准确率。

最多6是总并发上限，每卡最多4份模型；不能把6份BF16权重塞进一张80GiB卡。GPU0/1按fresh准入，模型逐份加载；同卡并发使用自有0700 MPS服务，原位置与实际K/V运行时逐层核验。CPU亲和性依据当时容器cpuset。

暂停在远端**本run**创建STOP即可；原生每forward计费hook会检查它，管理器停止自有加载并保留费用与缺失键，执行器不自动恢复。不要触碰旧run或其他人的进程。实验完成条件、费用和三交接指标口径以protocol.md为准。
