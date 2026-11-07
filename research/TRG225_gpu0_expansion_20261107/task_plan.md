# GPU0 扩展（同一TRG225队列）

用户明确选择“允许总共八个：两张卡各四个”。GPU1四健康模型不迁移。保留原科学锁与T/R/G225范围，不重跑已提交键，不改prompt/模型/选位/精度。

- [x] 只读测量两卡资源/性能与实际worker，记录原manager身份。
- [x] 新management目录manager96471接管同队列，停止原manager28169但不停止worker，保留旧管理状态/日志，原科学源不改。
- [ ] GPU0逐个准入加载、独立15SHA/BF16/identity/norm/实际新提交核验。
- [ ] 更新40分钟监控新manager/两卡资源策略，两地记录/Git不push。
