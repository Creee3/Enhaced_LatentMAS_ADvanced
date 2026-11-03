# 事实与边界

原run暂停时675条新三臂唯一结果，对应共同225题。用户现已授权新增G，未授权补齐原448题。G是注意力选位对照，不是已验证新方法。

新worker从冻结main448 worker派生，仅最终选择器和方法枚举改变；前两次receiver交接、prompt、seed42、latent10、温度/cap/context、Engine收费和原KV gather保持一致。所有实际prefix/probe/生成计入同一共享账本。

225题已经查看阶段结果，后续比较属于探索性机制分析。主比较G对T，双侧精确McNemar；Diamond106/非Diamond119分别展示，不作独立重复；不显著不代表等效。原答及原run_lock不修改。

实际启动发现：显存/利用率合格不保证CUDA可初始化。GPU2/7 index与UUID测试均失败；cuInit999与UVM设备open EIO为直接观测。驱动/libcuda版本相同，不能把故障武断归因到版本不匹配或仅编号变化。需要恢复CUDA环境的明确证据；当前无G准确率、无模型加载成功。

参考JSONL含合法U+2028，必须按LF分隔；冻结bundle自身已有相同格式教训。v1暂存保持STOP和原锁；v2采用明确格式修复与按卡telemetry，不热改运行中的科学锁。旧字节拷贝在Git用-text保留CRLF，避免重新checkout改变冻结解析器SHA。
