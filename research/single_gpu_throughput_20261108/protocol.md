# 单卡吞吐排查（工程代理测试）

用户授权调查并尝试加速单卡并发；已完成TRG225及全部旧原答冻结，不恢复旧监控。

假设H1：一份BF16权重、batch4/8的总吞吐高于batch1。H2：原位置审计中的GPU到CPU标量读取有开销。先用同一Qwen3-8B、SDPA、130个固定合成token输入、greedy固定128步，做batch1/4/8×原observe审计开/关×2次交错重复，另各一次16步预热。比较总output-token/s、单流token/s、CUDA同步墙钟、显存峰值和token SHA；不计算题目准确率。

全部native forward按实际batch token positions用原Ledger预留/结算，不重置共享账本。硬上界20000positions。GPU0只有实时util<=30%、free>=70%且无本次旧进程时准入；只一份模型，不改功率、时钟、MPS或其他进程。模型15文件SHA，BF16CUDA验证。每种模式保留原trace，失败不重载。

局限：合成短上下文、无采样、无真实压缩KV，不代表8192长回答、完整角色pipeline或四独立进程基准。原observe(latent=False)保留原实现；关审计仅作为工程对照，不作为科学结果。batch浮点/RNG顺序可能影响输出；token SHA匹配只说明此代理样本，不能承诺真实题集等价。

另离线精确比对旧225与新225题目键、T/R token/prefix SHA和正确性变化，解释差异，不合并不兼容原答。不用新模型调用挑选性能测试题。

先记录计划提交，再启动一次短测试；测试结束本轮复盘，不创建后台调度。
