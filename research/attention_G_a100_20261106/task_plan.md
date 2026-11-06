# A100 G 双 worker

用户授权：检查并运行最多两个G worker。仅G225共同题，原T/R/shuffle及旧G失败run仍暂停，自动化不变。

- [x] 核对旧锁/原答、预算与零forward预检，建立新主机版本run。
- [ ] manager唯一入口，最多2 worker，一卡一模型，4卡候选，沿用util<=30/free>=70%门槛。
- [ ] 核验新worker15SHA/BF16CUDA/identity/norm与实际G提交；完整inspect一次私有落地。
- [ ] 中文两地管理记录、Git不push；若仅一张卡合格则第二worker等待，不放宽门槛。

当前状态：PROTOCOL_HOLD。首worker加载核验通过，但旧T前缀匹配失败，未成为正式G；第二worker未启动。故障归档与费用/停止审计完成。用户单卡双worker资源授权有效，不能代替新增T对照的科学范围授权。
