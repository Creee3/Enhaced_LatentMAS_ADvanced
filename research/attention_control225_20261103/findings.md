# 事实与边界

原run暂停时675条新三臂唯一结果，对应共同225题。用户现已授权新增G，未授权补齐原448题。G是注意力选位对照，不是已验证新方法。

新worker从冻结main448 worker派生，仅最终选择器和方法枚举改变；前两次receiver交接、prompt、seed42、latent10、温度/cap/context、Engine收费和原KV gather保持一致。所有实际prefix/probe/生成计入同一共享账本。

225题已经查看阶段结果，后续比较属于探索性机制分析。主比较G对T，双侧精确McNemar；Diamond106/非Diamond119分别展示，不作独立重复；不显著不代表等效。原答及原run_lock不修改。
