# 云端CPU单题可用性测试

2026-11-04真人仅授权CPU先测一道题。独立技术试跑，不运行G、旧三臂或自动化，不改冻结科学代码和旧答案。

Qwen3-8B同15文件SHA、BF16完整CPU参数、16计算线程/1 interop、batch1、seed42；取已冻结225题顺序第一题。单模型非思考直答，仅允许提示词请求boxed选项，cap64；尚未使用约束解码，必须检查实际输出和是否触顶。此配置与原长推理/多角色协议不同，不能混入原准确率或作公平速度比。

`control.py launch`显式启动一次；`status`只读摘要；`collect`仅在实际退出且预留零后将所有文件按远端SHA完整落本地私有包。每个真实forward通过未修改的原Engine（注入CPU model）计入同一115M共享账本。加载/来源与15模型SHA、首forward、整体生成/解码吞吐、峰值RSS分别记录。

私有包：`advanced_MOC/experiments/latentmas_cpu_single_question_20261104`，含题目与原答，不进入Git。原实验/自动化继续暂停。只试一道，不按是否答对重试，不自动扩大样本。
