# G 注意力选位对照：启动计划

2026-11-03 真人明确授权启动G；授权仅覆盖暂停时共同225题的G臂，旧三臂仍暂停。

- [x] 固定225题（Diamond106、非Diamond119）、协议与参考答案来源SHA。
- [x] 新版本G选择器、兼容性证明与CPU机械测试。
- [ ] 远端零forward tokenizer/context/共享115M账本预检、原run停止核验。
- [ ] 独立manager按原动态GPU规则启动；真实进程与加载/新提交分开记录。
- [ ] 独立原KV/位置/raw RID/模型SHA证明、中文UTF8两地接续、本地Git提交不push。

G使用同64个query观测、同<=192 bank、稳定排序；保留注意力质量最高的88个自由prompt+强制8个prompt+30latent+4sink=130。预算与T相同；不拟合V，不改原RoPE，不重跑任何旧方法。失败保留现场，不盲重试。原GPU3/4/6失败需独立实证clearance，GPU5须至少达到既有24081MiB恢复证据。
