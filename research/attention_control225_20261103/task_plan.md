# G 注意力选位对照：启动计划

2026-11-03 真人明确授权启动G；授权仅覆盖暂停时共同225题的G臂，旧三臂仍暂停。

- [x] 固定225题（Diamond106、非Diamond119）、协议与参考答案来源SHA。
- [x] 新版本G选择器、兼容性证明与CPU机械测试。
- [x] 远端零forward输入token/context/共享115M账本预检、原run停止核验。
- [x] 独立manager按原动态GPU规则实际启动；两worker在CUDA初始化失败，已安全停止准入并归档。
- [ ] CUDA环境恢复后接续G；完成模型SHA和实际新G答案独立证明，不能把失败/加载当正式提交。
- [ ] 中文UTF8两地接续、30分钟仅G环境恢复监测、本地Git提交不push。

G使用同64个query观测、同<=192 bank、稳定排序；保留注意力质量最高的88个自由prompt+强制8个prompt+30latent+4sink=130。预算与T相同；不拟合V，不改原RoPE，不重跑任何旧方法。失败保留现场，不盲重试。原GPU3/4/6失败需独立实证clearance，GPU5须至少达到既有24081MiB恢复证据。

当前阻塞：GPU2/7 index与UUID隔离进程均CUDA unavailable/count0，直接驱动cuInit=999，/dev/nvidia-uvm打开errno5（EIO）。驱动与libcuda均580.173.02，普通设备节点2/7和ctl可打开。非显存门槛或G算法失败；未改宿主驱动/重启/触碰他人。G0/225、所有225pending、0held、0请求、0新增forward和费用、0自有预留。没有真人撤销G授权，STOP是故障准入护栏而非真人暂停；原三臂仍真人暂停。
