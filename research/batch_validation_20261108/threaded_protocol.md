# 四线程驱动四stream诊断

B1形状共享模型已token/EOS通过，但单CPU驱动吞吐22.25，未超过45.46四进程。以相同4packet、16预热+2×128+EOS8/16/32/64，新增ThreadPoolExecutor四CPU线程，各自CUDA stream/独立KV/RNG；SDPA审计请求上下文改为thread-local，防止全局指针串线。只一模型，GPU0原准入，不修改MPS/功率/线程数学精度，原账本<=50000positions，所有worker退出。主门槛仍是逐token匹配+EOS+吞吐；未通过不启动正式实验。
