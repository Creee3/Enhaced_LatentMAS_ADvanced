# 记录

2026-11-04：用户授权CPU单题；只读核对原Engine显式支持注入外部model/tokenizer，默认CUDA加载分支保持不动。开始独立版本，尚未运行。

首次远端启动脚本在Python解析阶段失败：过宽的PYTHON占位替换误匹配PYTHONUNBUFFERED；改为仅替换argv列表占位，未执行远端代码、未加载模型或调用forward。

2026-11-04 16:04（北京时间）：独立CPU单题进程PID678141/UID1033/start_ticks6466352，argv为CPU专属worker.py，CVD为空；原实验/自动化未恢复。启动预算actual74904532、reserved0、unknown58106，预检有足够余量。首次状态6/15模型文件已核对，权重I/O等待，推理尚未开始。

15/15模型SHA随后全部通过，进入依赖导入/CPU模型加载。只读诊断wchan=folio_wait_bit_common，已读取约16.58GB，当前等待torch/_dynamo/polyfills的pyc文件，仍未forward/收费。这是加载I/O等待，不可当CPU推理耗时。

2026-11-04 16:09（北京时间）：单题实际完成，225输入token、6输出token、未触顶，冻结解析器判本题正确。全部参数CPU BF16、16381470720B、16线程；推理4.0663s，首forward2.4973s，后续解码约3.404token/s，峰值RSS16.120GiB。模型from_pretrained加载11.9746s；包括15文件SHA/依赖导入/加载/Engine准备的前置耗时322.882s，总326.949s。该速度仅单题、短输出，含冷缓存影响，不是最优性能或与旧GPU协议的速度比。

只有一次计量请求：6个native forward、230个实际位置；共享账本actual74904762/reserved0/unknown58106，新增actual230、unknown0，无手工结算。PID678141已退出、自有预留0。完整远端源/问题/原答/raw/trace/收据/日志按SHA安全落本地，离线核对格式/解析器、15SHA、raw生成ID、唯一请求和结算均通过。没有追加题、重试或恢复G/原三臂/自动化。
