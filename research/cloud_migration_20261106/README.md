# liuxin-163 迁移与使用

本目录只包含迁移程序、环境清单和不含原答的管理说明。真人授权的是配置新云端；原main448三臂与G实验、自动化均保持暂停。

## 位置

|内容|新实例位置|
|---|---|
|研究根目录|`/home/ubuntu/ICML2027`|
|当前代码仓库|`/home/ubuntu/ICML2027/Enhaced_LatentMAS_ADvanced`|
|完整历史累计实验|`/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025`|
|环境|`/home/ubuntu/.conda/envs/kv-graph`|
|Qwen3-8B固定权重|`/home/ubuntu/models/Qwen3-8B`|
|归档、环境记录与验收|`/home/ubuntu/cloud_migration_20261106`|
|VS Code工作区|`/home/ubuntu/ICML2027/LatentMAS.code-workspace`|

新容器中新增兼容链接 `/home/liuxin → /home/ubuntu`，使冻结源码中旧绝对路径仍能定位相同字节；不修改原科学锁、模型锁、原答或账本。链接只在该旧路径及真实liuxin账户都不存在时创建。旧PID/UID/start_ticks保留作历史证据，不能据此认领新机进程。

## 打开环境

新开SSH终端后执行：

```bash
latentmas
python scripts/verify_snapshot.py
python research/cloud_migration_20261106/status.py
```

若当前终端未重读bashrc：

```bash
source /home/ubuntu/cloud_migration_20261106/activate.sh
```

此入口只激活环境和定位代码；不启动manager/worker，不恢复队列。默认HF/Transformers使用离线文件。VS Code工作区已指定同一Python解释器。

## 恢复范围与验证

三份2026-11-04已核验归档包含92648原文件，原始11266078211字节，压缩5749256900字节。current/source/history文件集合互不重叠。恢复程序验证压缩包SHA、gzip CRC、tar精确集合、每成员SHA与独立落盘读回SHA；保留Linux原始名称，仅恢复已验证范围内且目标存在的符号链接。

原四方法latent10/40的3584答案、冻结latent10基线1792、当前三臂675答案/225共同题、G零答案，以及所有已归档失败现场均保留。shared budget必须仍为actual74904762/reserved0/unknown58106，零active reservation，不重新结算。

`verification/RESTORE_VERIFIED.json`、`EVIDENCE_VERIFIED.json`、`MODEL_VERIFIED.json`、`MIGRATION_VERIFIED.json`是阶段性实际验收文件。只有相应文件存在且passed为true，才能声称对应阶段完成。

当前LatentMAS核心环境按旧Python3.10.21、torch2.6.0/CUDA12.4、transformers4.53.3等版本重建。全量旧环境包清单另存，不声称310个历史包或其他项目的独立API环境逐一重建。历史其他项目的源码、数据、实验资产随归档完整恢复；API凭据仍需自行配置。当前Qwen3-8B按revision `b968826d9c46dd6066d109eabc6255188de91218`和15文件旧SHA逐项核对。Mistral历史模型的配置/来源信息保留，其完整预训练权重不属于本轮活跃LatentMAS环境。

安装参考[PyTorch官方旧版本说明](https://docs.pytorch.org/get-started/previous-versions/)，权重来源[Qwen官方固定revision](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218)。未用不同revision或量化权重代替原锁。

## 新服务器与后续实验

新机可见4张A10080GB；这是共享GPU，当前他人负载不属于本任务。迁移只做文件/依赖/离线CPU检查，不构造大模型或生成答案。新硬件上的模型BF16、identity/norm、真实KV和新答案三臂核验仍需在未来真人明确恢复后独立完成，不能复用旧4090的运行证明。

以后恢复必须使用新版本管理记录，保留原停止现场与缺失键，重新适配实际可见GPU范围及身份；不重跑旧基线，不清零预算，不把新机视作已完成的新盲测，也不直接混用两种硬件时长。

## 联网与临时通道

新机外网请求曾被校园网门户拦截，TLS返回net.szu.edu.cn证书。迁移期间使用仅回环地址监听、仅允许公开包源/模型站点443端口的临时CONNECT转发与SSH反向隧道，TLS保持端到端验证。临时通道在任务结束关闭，不写全局代理。后续在线安装仍需先完成云端校园网认证；已部署的离线模型、数据和结果不依赖此通道。

原始归档及运行记录属于私有数据，不提交Git或公开；本地Git只提交迁移代码/版本清单/摘要，不push。
