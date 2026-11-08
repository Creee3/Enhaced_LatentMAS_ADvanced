# 进展

2026-11-08：完成 batch 分叉定位、单模型 streams、线程 streams、四进程私有 MPS、MPS 前缀精确重建及原生 cap8192 长生成验证。五组证据均完整归档核验，费用一次结算、自有预留零、全部自有服务退出。

技术结论：采用保持 B1 的私有 MPS 作为下一步工程试跑候选；拒绝将 batch4 直接接入冻结科学实验。短解码 MPS 78.86 token/s，对照四进程 45.46；四题两轮逐 token 一致。四题完整前缀精确相同，两题长生成自然 EOS 且完整 token 与历史 A100 R 相同。

已部署四题完整 T/R/G 入口和独立结构审计，默认 check-only 通过，四类错误门槛负向检查通过，无启动副作用。当前 launch_authorized=false；范围问题待回复，未启动新实验。旧 TRG225 已完成且保持冻结。

完整数据、限制与归档索引见 findings.md；私有启动状态见对应实验目录 launch_prepared.json。

2026-11-08 新阶段：用户已回复选择首四题完整 T/R/G 工程试跑，范围已明确。启动前实测GPU0 util0/free81153MiB，无已有MPS；容器affinity与cpuset均允许8核；源锁通过，账本actual78225351/reserved0/unknown58106。按task_plan.md推进12键和独立审计/归档，不扩展到225题重跑。
