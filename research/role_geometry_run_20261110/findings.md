# 实施发现

2026-11-10：实时旧HL1410/1410、extension645/645，manager/worker均退出，GPU0/1空闲；ledger actual95582768 unknown61211 reserved0。预算余量19356021，完整逐题零前向上界待计算。

本地只冻结了作者部分文件；同固定commit的GitHub树存在LOBF.py/GenerationOnly.py，可补充相同修订依赖。旧runner数据白名单不能直接支持六组，需要新适配。三个旧CPU控制不满足位置匹配协议，将在新版本实现。

实际云端最终判分预检为20canonical通过、20负例拒绝；本地HE32已记录数值不稳定，官方参数顺序覆盖与独立见证固定，但云端实际没有触发替代正例。原HF源文件不改。

Random-matched按G在真实来源×来源内相对位置bin的保留数量精确匹配，从原完整prompt候选池随机采样；G/T/R及R-shuffle/Pseudo-R具有attention预筛bank<=2B。随机对照检验保留配额而非强行限制在相同2B预筛bank；后续解释不得把它称为完全相同bank的算法消融。边界特异性主对照R-shuffle/Pseudo-R仍用同bank。

正式诊断后处理位于独立版本role_geometry_postprocess_20261110，10CPU检查通过。远端以management/postprocessing_lock.json单独冻结调用入口及纯NumPy分析代码；科学生成70文件锁不变。完整键与原始归档通过后再汇总各题内层/头，然后按题bootstrap1000次、seed42，六组分别报告。
