# 真实KV单卡批处理工程对照

用户授权“先试试”。正式TRG225及旧原答不改，不恢复监控。范围固定为原225顺序首4题、仅R选位，选题不参考正确性。由于未保存真实KV张量，使用原冻结stage/probe/final_select函数重建四题prefix和130行真实KV，仅工程用途；不重新报告准确率。

主比较：一张空闲A100的4独立进程 vs 1模型batch4，固定128步decode、2次重复、16步预热；另单模型逐题参考。模型15SHA/BF16/SDPA，温度.6/topk20/topp.95。保留每题独立CUDA采样RNG与原RoPE，batch前缀按题独立、左padding仅prompt，mask隐藏padding。原KV无拟合/回填。固定步数不按EOS提前退出，属于时长工程对照，不是原8192协议的新科学结果。

核验：每题130行/19169280B、原K/V gather、36×8选位SHA、30latent+4sink+末8prompt；batched位置IDs按每题original_end生成。前2次forward逐层核对实际注意力读入的原K/V，全部forward计36层SDPA；采样按题单独恢复RNG。比较batch与逐题token SHA并报告首个不匹配，不能用平均准确率掩盖。

所有forward计原共享账本，合计<=100000 positions的保守预检（含padding/预热/参考）；失败保留现场与真实费用，不盲重载。总worker最多4，GPU0首次util<=30/free>=70，后续逐模型加载需free>=21GiB；不碰别人的进程、不改MPS/时钟/功率。两端CUDA同步，加载/准备/预热分别记录，不计入decode吞吐；四进程用barrier同时起跑，计算总完成token/共同墙钟。短测试结束退出，不后台调度。若GPU0不符合准入则不启动。

结果是探索性工程证据，不能宣称batch与原HF generate/长回答数值等价；先完成短对照，再决定正式调度集成。
