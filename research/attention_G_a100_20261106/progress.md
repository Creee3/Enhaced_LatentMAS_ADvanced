# 记录

2026-11-06：真人授权最多两个G worker，原实验不恢复。初始仅GPU1符合准入，GPU0约35GiB可用但低于70%，GPU2/3被占用。新A100 UUID/80GiB/驱动595是与原4090不同的机器，旧按数字GPU索引的失败封锁不直接移植；保留旧失败现场，以新UUID与新准入记录独立管理。旧科学worker逐字节复用，只适配新run管理策略。

实际启动manager23131、GPU1 worker23142（UID1000）。15模型SHA、BF16 CUDA16381470720B、identity和target norm1.3758320808410645独立通过；在首题前缀与旧T选位SHA比较失败，于G Judger前退出，0新答案。5个请求由原Engine结算actual1426、unknown无新增；当前74906188/0/58106，无active reservation，无手工结算。

真人随后明确允许GPU1同卡双worker，已记录资源策略例外授权；因前缀协议失败没有启动第二个。新run写STOP，manager与worker均实际退出，224pending/1failed_held，无重复越界；旧run保持暂停。完整现场及账本/管理/旧快照/bundle37归档17228070B/92成员，SHA b71e754229fc936b37dc69a3e521cafac7c69b492da53d41f7e6dac711613c82，流SHA/gzipCRC/精确集合/逐成员/安全全解包读回SHA通过，两地证明保存。一次完整UTF8 inspect私有落地，只输出计数/进程/异常。

当前是协议校验hold，不是显存失败；具体数值差异原因尚未定位，不能直接宣称A100导致。不得跳过匹配检查或盲重试。若要重新生成A100上的T作为G配对对照，属于新增T生成，需真人扩展授权；旧基线不改。
