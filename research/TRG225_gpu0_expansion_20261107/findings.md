# 运行约束

GPU1保留原四驻留worker。GPU0首worker须fresh util<=30/free>=70%，后续自有同卡并发允许util豁免但free>=21GiB/待加入模型，逐个加载。每卡最多4，每槽位仅一次自动尝试；失败阻止该卡继续自动加槽，不重载，不影响其他健康worker。所有进程真实UID/start_ticks/argv/CVD匹配；共享原queue.lock和115M账本；只发原始真实缺失臂。
