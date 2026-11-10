# 实际发现

原manager是不可热改的70文件科学锁之一，原资源上限硬编码6。新管理版本必须另存并接管原worker与私有MPS；原worker/队列/账本/科学锁保持不变。原controller按management/identity.json与workers.json检查，可原样接续收尾；audit无六worker硬编码。
17:08左右检查：原六均ready无failure，40/1440；GPU0剩47217MiB、util11%；GPU3剩61874MiB、util39%，已有外部进程。GPU3新模型须先满足util<=30及free>=70%准入，不操作既有GPU3进程。

最终八worker ready与独立冻结核验通过（2026-11-10 17:24:04），两个新增模型常驻等待工程门禁。GPU3有原外部模型，fresh util0/free61874MiB通过准入后才启动。
