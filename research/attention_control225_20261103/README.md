# 注意力选位G对照（共同225题）

独立版本，用户2026-11-03明确授权。旧main448三臂保持暂停。详见protocol.md与task_plan.md。

本目录不含题目、gold、原答或日志；私有包为`advanced_MOC/experiments/latentmas_attention_control225_20261103`。`deploy.py prepare`创建一次新锁；`deploy.py launch`仅显式执行零forward预检及启动。`capture_current.py`一次完整UTF8私有捕获，stdout仅摘要；`verify_formal_worker.py --pid PID`独立验证实际新G提交。测试不会启动云端任务。
