# 约束

G225/seed42/latent10、cap8192与冻结prompt/解析器/选位/保护行/KV gather/RoPE不变；共享115M累计预算不重置，零重跑基线。新机时长不与旧4090直接合并。新版本仅调整4卡候选、max_workers2及Python路径，cloud_common默认探测range4；每卡仍仅自动尝试一次。
