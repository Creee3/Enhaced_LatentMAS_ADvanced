# 方法与固定协议

四角色按Planner→Critic→Refiner→Judger交接。Qwen3-8B，BF16，全CUDA，norm_only真实identity变换及mean-input-L2目标范数；seed42、context32768、GPQA输出8192、temperature0.6/top_p0.95/top_k20。作者prompt和程序解析器冻结。

每次交接保护已经保留的历史、所有latent及最初4个sink（仅一次），当前角色另保留32个prompt位置。三次交接的总位置数在latent10时为46/88/130，在latent40时为76/148/220。保留原RoPE位置；缓存存储槽与原位置分开。

- Full：全部KV。
- HeadOBF：作者attention headwise选择、exact-SVD rank8、正交残差V回填。外部原文件URL和SHA固定于`author_manifest.json`。
- 题内JIVE：共同rank4、特有rank4、固定20轮近似几何分解。仅使用同题实际保留的既往prompt V与当前prompt V。不是完整统计AJIVE，也无收敛最优保证。
- receiver_attention_sink_recent：读取下一角色prompt最后16个真实query的注意力；保护8个recent位置，再按每层/每KV头的分数选24个位置；原KV直接gather，无SVD、无V回填。probe在cache clone上进行，费用单独计入。

各case共享Planner，相同Judger RNG条件。receiver代码的legacy `source_v_equal=False`只是`method == jive`方法名标签，不是V发生变化的测量；原代码保留此字段，不将其冒充额外完整V等值证明。

完整实验允许GPU0–7作为候选，最多6个自有worker，一卡完整模型；fresh utilization≤30%且free/total≥70%准入，健康worker不迁移，不盲重载故障卡。共享账本累计actual+reserved+unknown≤115M，失败/probe保留费用。发布与测试均不启动云端推理。
