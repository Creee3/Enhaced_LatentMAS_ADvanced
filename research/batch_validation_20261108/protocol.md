# 批处理一致性定位与启动门槛

用户授权继续验证，能通过则准备实验。复用上一工程对照已验证归档中的首4题R真实KV packet，不重建、不读取正确率、不改旧结果。GPU0 fresh util<=30/free>=70%，只一模型，BF16/SDPA/15SHA，所有forward计同一115M账本，保守总上界100000positions。无后台监控、MPS或系统参数修改。

本轮一次模型加载，固定顺序：
1. 同一packet原HF generate(max_new_tokens128，原采样.6/.95/20，原observe与position fix)，与上轮逐题工程参考的共同有效前缀比较；自然EOS允许早停。
2. 固定上轮参考token历史，逐题与batch4逐步比较logits和采样：同样输入历史、每题独立RNG，记录第一次logits差异、最大绝对误差、第一次候选token差异及每步RNG state SHA。
3. 对packet0做单条左padding与4份相同无padding批次的第一步隔离，判断差异是否只出现在混合padding条件。
4. 全batch固定128步的同实现参考；保持batch形状4，用预登记人工EOS退出长度8/16/32/64验证活跃流不被其他流退出影响。人工EOS仅测调度，不是题目答案。

原KV实际读入前2forward逐层核对，所有forward SDPA数、存储长度、按题原position_ids和保护行检查。批量padding及EOS后的dummy计算均真实计账，独立标记。

启动门槛：不能只看吞吐。需HF短参考、原KV/RoPE、独立RNG、重复性、EOS退出、费用与退出证明通过；若逐token一致性仍失败或根因未隔离，就只生成BLOCKED启动草案，不启动新的准确率实验。不得将浮点近似一致偷换成原轨迹逐token一致。不同batch构成/后端必须另行验证，不能由四题外推225题。

本轮每阶段复盘一次，测试结束写两地记录和本地Git，不push。
