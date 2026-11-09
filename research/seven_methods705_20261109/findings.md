# 方法与实现核对

H/L直接执行固定作者Headwise/Layerwise AST，仅替换DynamicCache构造适配器；用独立sender prompt_scores+topk复核原作者metadata，并核对每层/每头原KV实际gather。H每头独立，L层内共用位置；全部继承历史/latent/sink保留。CPU夹具用真实36×8×128/BF16三交接。

extension五方法使用冻结stage/probe/judge核心函数，G/T/R选择器沿原逐字节relay_select，原Full/receiver函数保留。Planner全方法共享，G/T/R额外共享第一次Critic probe；后续分叉独立。HL每题7RID，extension每题26RID。联合最坏位置上界为HL逐题 p+2c+2r+2j+50+2×8192；extension p+7c+9r+9j+110+5×8192。

旧工程门槛keys==12错误在本次新manager源中修成工程题数×方法数，支持HL6键与extension15键。两个子run禁止并发启动，控制器串行且各阶段全退出/MPS收尾/归档核验才进下一阶段。代码版本仅新目录，不修改任何旧科学锁。
