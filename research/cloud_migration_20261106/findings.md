# 迁移发现

2026-11-06：新实例 SSH 别名 DevContainer（亦 liuxin-163），ubuntu@10.225.230.172，经 EdgeHost。已修复实例公钥认证，未上传私钥。

Ubuntu24.04/Python3.12 系统；/opt/anaconda3 为只读公共 base，Python3.14.6。旧 kv-graph Python3.10.21，torch2.6.0 / transformers4.53.3 / accelerate1.6.0 / numpy2.2.6 / scipy1.15.3 / tokenizers0.21.4。

4×A10080GB，128GiB RAM/8 CPU，约600GiB空间。GPU0/2/3正在被使用；本任务不派发实验或模型推理。

本地三归档共92648文件/11266078211原始字节/5749256900压缩字节，已验证。Qwen3-8B模型固定revision b968826d9c46dd6066d109eabc6255188de91218，15文件原SHA锁完整，权重原备份未携带，需重新下载并逐文件对锁。

目标研究根目录 /home/ubuntu/ICML2027；档案/验收根 /home/ubuntu/cloud_migration_20261106；用户级新环境 /home/ubuntu/.conda/envs/kv-graph。保持旧 /home/liuxin 路径为历史，不修改原锁。

确认新容器和 EdgeHost 的公网TLS连接均被校园网门户截获，证书CN=net.szu.edu.cn，不能通过换CA或禁用SSL来解决。本机公网路径正常，已临时回环CONNECT代理+SSH反向隧道为包/模型下载提供TLS透传；任务结束撤销。用户提供SSH账户/密钥不等于校园网认证已完成。

路径适配方案：若新容器不存在真实liuxin账户或/home/liuxin目录，新增/home/liuxin→/home/ubuntu兼容符号链接，在paths.json记录；不改原锁中绝对路径。只恢复指向已验证恢复范围的源符号链接，其他缺失目标留记录。新记录均在迁移目录，原STOP/manager/PID/队列只作历史证据，禁止自动认领。
