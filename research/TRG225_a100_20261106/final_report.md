# A100 T/R/G 225题完整结果（2026-11-07）

675/675精确覆盖（新671+复用4），225 case complete；无pending/held/failed缺失，八个worker及manager均正常退出，自有预留0，原答未修改。

| 同题范围 | T | R | G |
|---|---:|---:|---:|
|全部225|104/225（46.22%）|114/225（50.67%）|114/225（50.67%）|
|Diamond106|53/106（50.00%）|58/106（54.72%）|57/106（53.77%）|
|非Diamond119|51/119（42.86%）|56/119（47.06%）|57/119（47.90%）|

主检验R对T：净多对10题（+4.44个百分点），不一致32对22，双侧精确McNemar p=0.220328，未达到alpha=.05。R对G均114/225，但各自独有正确24对24，p=1；这不是等价性证据。无R-shuffle，不能单独宣称正确角色边界的因果收益；Diamond为子集、非独立重复。这些题已被查看/开发过，属于探索性比较，不是新盲测。

三方法每条实际KV均19169280B；按原位置Full参考字节汇总加权保留13.38385%、减少86.61615%。该Full为未压缩位置字节参考，不是新增Full生成，也不是GPU/FLOP/通信实测速率收益。

本run 2021去重RID，实际3226701 token positions、unknown新增0、失败请求0；加复用pilot及A100前缀失败前置成本共2042 RID /3238990 positions。共享账本最终actual78143752/reserved0/历史unknown58106，未清零、迁移或重复结算。

完整结构核验：671本attempt原答的raw RID/token SHA、36×8×130位置npz、原K/V gather/RoPE及30latent+4sink+末8prompt保护、实际KV、同题prefix/probe RID、15权重SHA/BF16/norm、trace/raw全RID终态一致和费用对账通过；复用4条覆盖及前缀位置重建核验通过。冻结程序在全部退出及归档验证之后重算作者解析器。原科学锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f未改。

完整归档含run、复用pilot、前置失败、来源引用/冻结参考快照和bundle37/模型锁，不含预训练权重。远端流SHA、gzip CRC、tar精确集合、逐成员SHA、安全全解包读回SHA全部通过。

归档SHA：`0e54558e885eeafb998ef90a5a70fd5d7d0864802ddb15c1165bf7657ee187c4`；79075400B，1073成员。本地私有包：`D:\liuxin_workspace\ICML2027\advanced_MOC\experiments\latentmas_TRG225_a100_20261106`，机器结果`final_summary.json`，归档证明`completion_archive_verified.json`。
