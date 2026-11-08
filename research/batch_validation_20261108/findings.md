# 单卡并发验证：2026-11-08

可进入下一步四题完整 T/R/G 工程试跑的方案是：GPU0 私有 NVIDIA MPS 服务，四个独立模型进程，各自保持 B1 前向、KV 和 RNG。尚未启动该四题试跑，也未重跑已完成的 225 题。

## 吞吐与轨迹

| 实现 | 总输出 token/s | 与原 B1 参考逐 token 一致 |
|---|---:|---|
| 原四独立进程、无 MPS | 45.46 | 四题、两轮全部一致 |
| 真正 batch4 | 75.47 | 失败：四题中三题分叉 |
| 单模型轮转、四 CUDA streams | 22.26 | 通过短生成与人工 EOS 调度检查 |
| 单模型四 Python 线程及 streams | 13.76 | 通过，但更慢 |
| 私有 MPS、四独立 B1 进程 | **78.86** | **四题、两轮全部一致** |

MPS 短解码吞吐约为原四进程的 1.73 倍（提高约 73%）。这是跨时段的固定 128 步工程对照，不能外推为完整四角色流水线或整套题的同等提速，也不是准确率结果。

batch4 的 teacher-forcing 检查在相同 token 历史和相同逐题 RNG 状态下，从第一步就出现 logits 差异；无 padding 的四份同题也出现差异。因此不能仅归因于 RNG 串扰或左 padding。证据支持 batch 形状改变数值路径这一解释，但不等于排除了所有实现问题。本轮不采用 batch4 替换冻结 B1 轨迹。[PyTorch 数值精度说明](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)也说明批量与逐条运算不保证逐位相同。

## 原生长生成门槛

在私有 MPS 下重新构造四题上游前缀：全部 36 层 K/V、选位、输入 IDs、绝对位置和 RNG 均与旧工程 packet 精确相同。源 run 的 32 个锁定文件和 bundle37 校验通过。

另用两个独立进程执行原始 HF generate，保持 cap8192、context32768、BF16/norm_only、seed42、temperature .6、top_p .95、top_k20：

| 题目 packet | 自然 EOS 输出长度 | 生成时间 | 与原 A100 R 完整 token 数组 |
|---|---:|---:|---|
| 0 | 2805 | 133.62 秒 | 完全一致 |
| 1 | 2097 | 99.55 秒 | 完全一致 |

两题均未触达 cap；15 文件 SHA、16381470720 字节 BF16 CUDA 参数、identity 与 mean-input-L2 target norm 1.3758320808410645、原 KV gather/RoPE/保护行、逐 forward SDPA 数及费用核验通过。这是两题原生长轨迹验证，不代表 225 题或新完整 T/R/G 流水线已经验证。

MPS 用途与隔离方式参考 [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html)。只创建自有 0700 私有 pipe，显式绑定 GPU0 UUID，未修改系统 compute mode、时钟、功率或其他 GPU/进程。

## 费用、退出和归档

本轮五组验证合计 99 个唯一 RID、44105 个实际输入/输出位置；新增 independent unknown 为 0。完成快照 actual=78225351、reserved=0、unknown=58106，总和未超 115000000。所有自有验证 worker、控制器及私有 MPS 服务均已退出。旧科学结果和旧 active 不变，未新增监控。

各归档都通过远端流 SHA、gzip CRC、tar 精确成员集合、逐成员 SHA 和安全全解包读回 SHA：

| 证据包 | SHA256 | 字节/成员 |
|---|---|---|
| batch_validation | `09a45d329814f4dba958ed2a9841c15a61511c7cffb1a4f029f1ceb6a3f56aed` | 22800 / 18 |
| stream_validation | `c230efc428e409930d42579fb9ee79e88697a9df2847441cf3184afecd7ee24f` | 9542 / 9 |
| threaded_stream_validation | `3a1073e18908d0071cce676b979e3a69e50eef0406a97c8fce07c2e1f3be8a46` | 8252 / 9 |
| mps_validation | `1d24eafce577bf1c3f60a5d5637453b01d2821d206a648e6b7bd6ad507ff3b7d` | 17596 / 56 |
| mps_native_smoke | `ed97af545da8ff2965e7a945351b28834f56fedd96e0bf8a5751deb5f4ea3b33` | 61504278 / 35 |

原答、token 数组、KV、日志、原账本和完整证明只在私有实验包及对应远端 evidence 下，未写入 Git。

## 已准备的下一步入口

私有元数据：`D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/batch_validation_20261108/launch_prepared.json`。

远端：`/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/trg4_mps_smoke_prepared_1794108404388723128`。

`manager.py` 默认只检查，不加载模型；`--execute` 还要求明确选定实验范围并令 plan 的 launch_authorized 为 true。当前为 false。准备包只支持固定首四题 T/R/G，共 12 个键；每题共享原 prefix/probe/RNG，原科学函数通过 AST 提取使用，原 cap8192 和选位不变，独立 audit 不计算准确率。保守费用预检上界为 4×9×32768=1179648 位置，仅用于预算可用性检查，不新增调用硬限。

CPU 负向检查确认 blocked 状态、错误 cap、错误 worker SHA、失败验证报告均拒绝启动；恢复原文件后默认检查通过。无 forward、无新增费用、无 manager/worker 启动副作用。后续完整四题的三方法新答、独立结构审计、费用退出和归档仍应作为工程试跑的完成条件，不能由两题 R 长生成替代。

用户已授权加速验证及启动准备。工程四题、全量另建 run 或仅保留入口的范围问题尚无回复；当前不默认重跑已完成的 225 题。
