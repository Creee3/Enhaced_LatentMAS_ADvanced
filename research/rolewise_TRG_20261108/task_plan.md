# GGG/RRR/TTT全交接576题

用户已明确授权：只保留GGG/RRR/TTT；按96题GPQA开发集、280题MMLU-Pro、200题GSM8K推进；先极小工程检查再正式运行；最多6个worker。无其他混合臂、Full、R-shuffle、额外seed或latent40，不恢复旧实验，不创建自动监控。

- [x] 固定数据源revision/SHA及576题清单：GPQA Diamond/非Diamond各48，MMLU-Pro14领域各20，GSM8K test200；按ID哈希选择，不按答案或效果选择。
- [x] 新版三交接选择器：同query/候选bank/预算，96prompt+已有latent+4sink，发送方末8prompt强制保留，来源标签随原位置传播，原K/V gather/RoPE。CPU验证闭式选择、短prompt、三次mapping与不修改原缓存。
- [x] 模板/解析器：GPQA原作者模板与解析器；MMLU-Pro仅将原选择题Judger范围扩展到实际选项（A-J）；GSM8K作者数学模板和固定程序数值归一化。测试后冻结，不作AI语义判分。
- [x] 零forward真实tokenizer逐题上界与共享115M预算预检，源32/bundle37/model15 SHA检查，冻结源锁和协议并Git提交后才调用GPU。
- [ ] 4题工程阶段（2 GPQA、1 MMLU-Pro、1 GSM8K，来自最终清单），12键。完整结构/计费/退出与本地归档通过后，同代码同协议首遍12键复用，正式阶段不重复生成。
- [ ] 正式执行至1728唯一键，仅发布计数、真实进程、异常与重要变化。最多6自有worker，每卡最多4份BF16模型，首份fresh util<=30/free>=70%，后续自有模型下free>=21GiB且上个ready。优先GPU0/1空闲卡；不触碰他人或其他实验进程。独立私有MPS，仅在归属清晰且目标卡未被其他MPS占用时启用；同GPU并发权限已明确。故障不盲重载、不重排complete。
- [ ] 全部完成退出且自有预留0，再完整本地归档、SHA/CRC/精确成员/安全解包读回，冻结程序分题集及GPQA子集汇总、实际三交接KV与费用、统计检验；两地中文记录、本地Git不push。

本轮是全链不同上游路径的同题配对试验。共享Planner及第一次receiver probe；第一次选择分叉后，各臂保持真实独立上游，不强制相同prefix，不把历史逐token不一致视为自动失败。硬门槛仍是代码/结构/预算/原KV与位置/配对起点公平。初始种子42、latent10、cap8192/context32768、temperature.6/top_p.95/top_k20、Qwen3-8B BF16/norm_only不变。

主比较预登记RRR对GGG，双侧精确McNemar，三个题集构成一个检验族用Holm控制alpha.05；TTT对照和三方法其他配对作为次要结果报告。MMLU-Pro各领域等样本，报告总体及领域描述；不做中途准确率或按显著性加题。GPQA为已看过开发题，Diamond是子集；新题集固定子集不声称模型训练时未见。

启动时只统计实际可准入worker数量，6是上限而非必须占满。不将6份模型塞入一张80GiB卡。开始新阶段前复查本计划并记录关键事实；工程阶段后和全部完成后反思，遇到新的结构/预算故障即时保留现场。
