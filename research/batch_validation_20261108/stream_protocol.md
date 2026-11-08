# 条件分支：共享权重但保留B1形状

Teacher-forcing已显示：原HF短参考4/4匹配；B4在第0步logits已不同，同步每题RNG仍不同，无padding的同题B4也复现。因此先不启动B4正式实验。

本分支仍使用相同4packet、同Qwen3-8B BF16/SDPA，一份模型。对照：同一stream交错B1 forward vs 四独立CUDA stream交错B1 forward。固定128步各两轮，16步预热；每题独立KV/RoPE/RNG，不改变GEMM批次形状。保留前两次实际K/V和每forward36层SDPA检查。追加人工EOS（真实EOS token注入）在8/16/32/64步退出，检查剩余流前缀不变和独立结算。

GPU0仍fresh util<=30/free>=70；只一模型、不动其他进程，源与packet SHA验证，所有native forward原账本真实计入，本分支<=50000positions。比较输出与已核对HF的逐题参考，全部4题×2轮需精确匹配；EOS检查也需通过。速度若未胜过原四进程45.46 token/s，不宣称并发优化成功；即便通过也只允许准备小样本完整cap8192 smoke草案，不能凭128步四题宣称225题/长上下文全部验证。

不创建后台监控，结束退出。计划在本分支启动前提交。
