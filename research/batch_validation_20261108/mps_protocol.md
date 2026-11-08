# GPU0私有MPS四进程对照

前两条共享权重B1路线token/EOS均匹配，但单线程4stream22.25、四线程4stream13.76总token/s均未超过原四进程45.46。用户授权继续验证优化可行性，因此另做资源调度对照。

复用已归档的real_kv_batch worker.py（decode函数不改），相同4真实R packet，4独立进程、16预热+2轮128步，共同barrier。仅新增MPS环境：CUDA_VISIBLE_DEVICES指定GPU0 UUID，私有0700的CUDA_MPS_PIPE_DIRECTORY/LOG_DIRECTORY，使用当前UID启动服务；不改compute mode/频率/功率、不用sudo、不触碰GPU1/2/3或其他进程。

GPU0初始fresh util<=30/free>=70%，逐个加载需free>=21GiB，上个ready才加下个。若已有可见MPS服务则不启动；服务初始化失败不强行修复。实际容器PID/UID/start_ticks/argv/私有管道环境验证归属，不按NVML宿主PID认领。全部forward原账本<=20000positions；失败不盲重载，停止/清理仅本试验资源。

输出必须4题×2轮逐token匹配已验证HF短参考，prefix/KV/RoPE/账本核验需通过。性能是相对此前同代码/输入/硬件四进程短对照的探索性测量，不是同日交错大样本确认。仅当兼容且速度高于45.46时准备小样本长生成smoke草案；不因此直接重跑225或旧baseline。模型全部退出后发送私有MPS quit并核实自有服务退出。
