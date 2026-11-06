# 记录

2026-11-06：旧G首次在新机前缀位置SHA与旧T不一致而退出，完整归档通过、原manager/worker全退出、预算74906188/0/58106。用户明确授权A100小样本T/G，另已授权GPU1同卡双worker。新实验只配对本机共享前缀，不宣称与旧4090 T逐位相同；旧差异作为诊断如实记录。

已完成：run paired_TG_a100_two_s42_l10_20261106T154043Z，manager26112、GPU1 slot1 PID26115/slot2 PID26376，均UID1000。两模型独立15文件SHA/BF16CUDA/identity/norm通过，观察到两者同时Judger，整卡快照34171MiB/81920MiB，没有OOM。

两题四答案全部完成：Diamond T正确2454token/G正确1343token；非Diamond T正确1813token/G错误1069token。仅两题机制部署验证，不估计泛化准确率，不做显著性。每题两臂共享6个prefix/probe RID、相同prefix stages/transfers/位置，最终选位不同；两题旧T前缀均不匹配如实记录。原始K/V gather/RoPE、全部latent/sink、Refiner末8prompt、36x8x130位置SHA、19169280B、token/raw RID/parser SHA/native位置全部独立通过。

16个唯一RID、实际10863位置、unknown新增0/失败请求0/手工结算0；预算74917051/0/58106。所有本实验manager/worker已退出，active reservations0，旧run仍暂停。完整run+bundle37+账本私有归档SHA4e8d5af933a6af8ea6434128968401357840d80156f0f58d4acca411303a94e5，1057775B/126成员，远端流SHA/gzipCRC/tar精确集合/逐成员与安全全解包读回SHA通过，另存最终cost/scope/active补充记录，两地一致。冻结plan部分未使用的225题文字标签继承旧模板，在management/scope_clarification.json明确，仅新task_plan/绑定的两题队列与4目标生效；不热改冻结锁。一次完整inspect私有落地。
