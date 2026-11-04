# 记录

2026-11-04：用户授权CPU单题；只读核对原Engine显式支持注入外部model/tokenizer，默认CUDA加载分支保持不动。开始独立版本，尚未运行。

首次远端启动脚本在Python解析阶段失败：过宽的PYTHON占位替换误匹配PYTHONUNBUFFERED；改为仅替换argv列表占位，未执行远端代码、未加载模型或调用forward。

2026-11-04 16:04（北京时间）：独立CPU单题进程PID678141/UID1033/start_ticks6466352，argv为CPU专属worker.py，CVD为空；原实验/自动化未恢复。启动预算actual74904532、reserved0、unknown58106，预检有足够余量。首次状态6/15模型文件已核对，权重I/O等待，推理尚未开始。

15/15模型SHA随后全部通过，进入依赖导入/CPU模型加载。只读诊断wchan=folio_wait_bit_common，已读取约16.58GB，当前等待torch/_dynamo/polyfills的pyc文件，仍未forward/收费。这是加载I/O等待，不可当CPU推理耗时。
