# A100 T/R/G 225共同题

真人授权补T/R/G，不跑R-shuffle；GPU1请求最多6并发，但6份BF16权重91.54GiB超过80GiB，部署可承载上限4，逐个实测接入。每40分钟更新原监控。

- [x] 冻结225题/106Diamond+119非Diamond；3方法675唯一键，复用两题pilot4条，仅新增671；旧基线只读。
- [x] 新版资源manager、CPU检查和0forward预算预检。原科学选位器、prompt/parser/模型/latent10/cap8192不改。
- [ ] 启动GPU1最多4独立worker，逐个独立15SHA/BF16/identity/norm及本attempt提交核验。
- [x] 两地中文管理记录，本地Git不push；原监控改为40分钟，仅本run。

223个新case三臂共享同一前缀/probe/RNG；两道复用题的T/G来自A100pilot，R重建相同模型/seed/协议前缀并严格核对位置SHA，RID因复算不同，费用分别计入。复用题不冒称三臂同一次prefix调用。三种方法按case索引循环换序。无shuffle控制，不能据此单独宣称正确角色边界是收益原因。

启动完成：manager28169，GPU1 slots PID28172/28440/28695/28985均独立15SHA/BF16CUDA/identity/norm通过，资源快照68830MiB/81920MiB；三方法本attempt新答正式核验仍由后续巡检继续，未冒称已全部正式。监控gpqa-448已ACTIVE，每40分钟。
