# 约束

只比较新A100的T/R/G，旧4090结果保持各自范围。相同96prompt+30latent+4sink=130/19169280B，R按来源块目标，G仅注意力质量，T整体输出匹配。Qwen3-8B15文件SHA/BF16/norm_only、作者prompt/parser、seed42/latent10/context32768/cap8192/采样不变。Diamond106为225子集，分开报，不相加。旧1792 baseline不重跑。
