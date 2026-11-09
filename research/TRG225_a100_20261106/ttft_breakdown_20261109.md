# 同题GPQA96：记录的耗时拆分与TTFT缺口

没有精确TTFT记录。当前Engine只有每请求整体elapsed_seconds及最多每秒一次native_forward_progress，没有首个采样token完成时间戳或逐请求排队/分词/采样事件。不能把首个forward进度或Judger前准备称为TTFT。

| 方法 | 进入Judger前准备 | 其中T/R/G选位+gather | 其余前缀/probe/开销 | Judger全生成 | 重建整臂 | 平均输出token | token/Judger秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
|T|41.96s|35.64s|6.32s|435.75s|477.71s|3899.3|8.95|
|TTT|17.19s|14.16s|3.03s|260.12s|277.31s|3825.1|14.71|
|R|41.07s|34.75s|6.32s|439.92s|481.00s|3910.2|8.89|
|RRR|15.93s|13.07s|2.86s|263.04s|278.97s|3958.6|15.05|
|G|9.92s|3.61s|6.32s|421.15s|431.07s|3783.1|8.98|
|GGG|3.78s|0.69s|3.09s|256.48s|260.26s|3572.7|13.93|

旧T/R/G的前两次原receiver压缩计在“其余”列；新TTT/RRR/GGG的三次新选择均计入选位列，不能将这两列直接作纯算法消融。

两轮所有这96题的Judger初始历史KV都为130行。TTT相对T的总差中约88%落在Judger；RRR相对R约88%；GGG相对G约96%。这说明“多次压缩更快”不是由这张表支持的结论。运行配置、CPU亲和、并发/MPS与输出长度均混杂，不能定量归因于其中一项。RRR输出token略多于R，依然Judger更快，单靠长度不能解释。

阶段组件细表（平均秒）如下；标注shared的成本在原run实际共享，仅独立单臂重建中计入。三跳first probe已在shared内，不能再把components[0].required_probe重复加一次。

T: latent_P_C_R_seconds=3.9139, early_receiver_probes_seconds=0.9051, early_original_receiver_compression_seconds=0.9289, shared_Judger_probe_seconds=0.5695, unallocated_preparation_overhead=0.0016

TTT: shared_Planner_first_Critic_probe_seconds=0.9879, Critic_Refiner_latent_seconds=1.5653, remaining_two_probes_seconds=0.2774, unallocated_preparation_overhead=0.2021

R: latent_P_C_R_seconds=3.9173, early_receiver_probes_seconds=0.9055, early_original_receiver_compression_seconds=0.9289, shared_Judger_probe_seconds=0.5694, unallocated_preparation_overhead=0.0016

RRR: shared_Planner_first_Critic_probe_seconds=0.9879, Critic_Refiner_latent_seconds=1.4081, remaining_two_probes_seconds=0.2564, unallocated_preparation_overhead=0.2093

G: latent_P_C_R_seconds=3.9139, early_receiver_probes_seconds=0.9051, early_original_receiver_compression_seconds=0.9289, shared_Judger_probe_seconds=0.5695, unallocated_preparation_overhead=0.0016

GGG: shared_Planner_first_Critic_probe_seconds=1.0775, Critic_Refiner_latent_seconds=1.5446, remaining_two_probes_seconds=0.2755, unallocated_preparation_overhead=0.1967

Trace证据：576/576个Judger请求有进度；首条进度时已完成forward中位数13.0，范围2–22。进度事件没有保证首token采样或CUDA同步，不能还原真实TTFT。

完整warm请求TTFT应分：调度/等待→分词及准备→Planner latent→第一次probe及选位→Critic latent→第二次probe及选位→Refiner latent→第三次probe及选位→Judger prefill→首token采样/可用。当前精确缺少调度、Judger prefill/采样边界、首token事件。模型冷加载另记，不能混入warm TTFT。
同题分支原本依次执行，独立单臂重建不含排在前面的其他臂等待。当前冻结实验不加热插埋点；未来独立性能验证应在首个采样token可用处记录单调时钟/CUDA同步边界，再在相同MPS/并发/CPU/固定生成长度下作对照；不为取数重跑科学结果。
