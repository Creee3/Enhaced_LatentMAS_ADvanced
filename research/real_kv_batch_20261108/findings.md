# 真实Judger KV单卡对照结果（2026-11-08）

结论：batch4总吞吐比四独立进程快约66%，但没有通过逐token轨迹一致性门槛；仍保留为工程分支，不替换正式生成、不更改旧准确率。

## 协议和有效范围

同一张GPU0 A10080GB、Qwen3-8B BF16/SDPA/15文件SHA。原225顺序首4题，不按正确性挑题。因旧run未保存KV张量，用冻结的stage/probe/final_select函数重建R臂真实KV：两次上游receiver压缩、最终96prompt+30latent+4sink=130行/19169280B；核对每层每头原K/V gather、位置SHA、全部latent/sink及Refiner末8prompt。四题prompt长304/243/250/331，原位置按题独立。各条件加载相同packet文件，而非各自重建前缀。

新解码器使用温度.6/topk20/topp.95、每题独立CUDA Generator及相同保存的RNG起点，固定128步不按EOS提前退出，各2轮重复和16步预热。batch左pad prompt，以mask隐藏padding，position_ids按每题原absolute_end生成；缓存存储坐标与RoPE分开。每轮前2个forward逐层核对实际SDPA读入的前130行K/V，全部forward检查36层调用数、缓存实际长度。padding真实forward也计账。

参考是本次工程解码器的逐题执行，不是已证明与原HF generate逐token等价的正式实现。短测试不测原8192完整时长、不计算任何正确率。

## 性能（总output tokens / 同步墙钟）

| 实现 | 总token/s（2轮均值） | 每轮总512个输出token的墙钟 |
|---|---:|---|
|单模型逐题串行|21.42|23.893s/23.917s|
|四独立模型进程，同屏障起跑|45.46|11.257s/11.267s|
|单模型batch4|75.47|6.785s/6.783s|

主比较batch4/四进程吞吐比1.660；不能把先前合成batch8相对batch1的7.92倍套到这里。模型加载/真实prefix准备/预热独立，不计入表中；计时包括本轮KV输入准备、采样、记账和结果整理。四进程按最早start到最晚end的共同墙钟，避免把各流token/s直接相加。

batch4 PyTorch峰值分配约16.01GiB；四独立进程各自峰值分配合计约61.71GiB。后者为各进程峰值之和，非NVML同一时刻峰值，两者均不含CUDA上下文等额外占用。未开启MPS/改频/改功率/量化。

## 一致性结果和限制

逐题参考两轮重复相同。四独立进程各题各轮128token全部匹配参考（8/8条）。batch4只有packet1匹配；其余packet0/2/3分别在零基token69/53/123首次分叉，两轮均同样复现（2/8条匹配）。KV读入/保护行/原位置与SDPA调用检查没有报错，但这不构成整个新解码器已数学等价的证明。

BF16批量矩阵计算与逐样本计算可出现数值差异，这是一个候选解释；还不能排除新实现的padding、采样或位置处理问题。PyTorch官方说明batch计算与逐切片计算不保证逐位一致：https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html 。该说明不能代替本例的因果定位。

下一步：在分叉前强制相同token输入，比较每步逐题/batch logits、softmax及采样RNG状态，找到首次差异来自模型计算还是采样/对齐；随后做原HF generate短参考与EOS不等长退出验证。通过这些门槛后再考虑完整方法实验，不恢复旧run、不拼接旧原答。

## 费用、现场和退出

所有阶段60唯一RID、20360实际token positions（含6392的真实prefix重建），unknown新增0、自有预留0；总保守上界20396，小于协议100000。四个GPU worker均退出，没有后台调度。旧科学源32文件/bundle37 SHA验证通过。

控制器曾在阶段切换的即时资源准入检查停止；原阈值未放宽。保留首次源码/日志/故障记录后，改为有界等待fresh准入，已完成serial不重复；一次计划管理接管保留当时batch子进程，模型不重启。后续四进程无worker故障、无重复结算。完整现场/源/packet/采样token/计时/trace私有归档两地可追溯，未把题目、原答、token或KV张量写入Git。

归档远端流SHA、gzip CRC、tar精确集合、逐成员SHA、安全全解包读回均通过。

归档SHA：`4e1462811bbf1dffbfaad7e88cbfaa2888acc8e27ddeec49eccba6ca14aaefa0`，61509706B/82成员。
