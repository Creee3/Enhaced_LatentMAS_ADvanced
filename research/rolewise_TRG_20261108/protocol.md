# 全交接GGG/RRR/TTT：冻结协议

用户确认仅三臂及576题：GPQA96（Diamond/非Diamond各48，已看过开发范围）、MMLU-Pro280（14领域各20）、GSM8K test200。数据源版本、下载字节SHA与固定ID哈希分层清单在私有data_manifest/questions中，选择不看答案或方法效果。主比较RRR对GGG，三个题集双侧精确McNemar以Holm控制alpha .05；TTT及其他配对为次要分析。不合并不同题集准确率，GPQA Diamond为子集，无额外独立样本。

四角色Planner/Critic/Refiner/Judger、latent10、seed42、Qwen3-8B 15文件SHA/BF16/norm_only/identity/输入embedding平均L2，cap8192/context32768、temperature.6/top_p.95/top_k20固定。GPQA与GSM8K作者模板逐字保持，MMLU-Pro只扩展Judger选项范围，0-shot。GPQA与MMLU-Pro使用冻结作者boxed提取并校验合法选项；GSM8K使用冻结数值归一化（逗号/小数/分数），仅程序评分，解析失败计错误但单独报告。

三次交接分别都使用同一方法G/R/T。目标每次最多96prompt，最新发送方末8prompt强制保留，所有已有latent与4sink保护；候选不足96全留，绝不补假行。常规交接长度110/120/130，36层×8KV头各自可选不同原位置，头间保留长度相同。按原角色新增位置传播来源标签；接收方self/sink为块0，其余Planner/Critic/Refiner块1/2/3，当前未出现块为空。所有方法同候选attention bank<=192、同64个GQA query观测和精确重新归一化；G按attention mass，T总输出误差，R四块误差，直接gather原K/V、原RoPE，无V回填/拟合，无SVD、新训练或latent删除。

每题Planner latent和第一次Critic probe实际共享一次。第一次选择分叉前记录共同缓存/RNG SHA；三个分支各自恢复相同起点、运行Critic/Refiner及各自探针，后续不同前缀不强制共享。方法顺序按题目index轮换。每个分支7个请求（含共同2），整题去重17RID；所有forward/必须probe/失败都进同一115M账本。零forward预检为每题p+4c+6r+6j+70+3*8192，原位置上界p+c+r+j+30+8192<=32768。此是预算上界，不新增调用硬限。

工程阶段选清单中固定2题GPQA（各子集1）、1题MMLU-Pro、1题GSM8K，共12首遍结果；源/方法/模板/预算不变，完整独立结构与费用退出、本地全归档验证通过后只读复用这12键，正式阶段仅生成剩余真实缺失键。工程阶段不计算正确率。每种任务至少有一条程序可解析输出才放行，若全不可解析则保留现场定位格式问题，不按正确与否选题或改方法。旧225/main448/pilot原答与锁不改，不将跨协议旧答案混入本次配对。历史逐token不一致只作诊断，不作为自动失败；实际KV/位置、共享起点、RNG、源锁及公平预算是硬要求。

最多6个worker总并发，每卡最多4份完整BF16模型。仅GPU0/1按UUID候选；首份fresh util<=30/free>=70%，后续只有存在健康自有ready模型才可免util、仍需free>=21GiB，上一个ready才加载下一个。每卡每阶段最多4个自动槽位尝试，任何该卡worker故障即停止其新加载，不盲重载；其他健康worker继续。不碰其他进程、功率、时钟、allocator或模型量化。按实时容器cpuset分配自有CPU线程。私有0700短pipe MPS按卡隔离，并避开其他MPS占用；归属依据真实容器UID/start_ticks/argv/CVD/私有pipe，不用NVML宿主PID。

成功必须1728精确键、无pending/running/held/失败缺失、全部自有manager/worker/MPS退出及自有RID预留0。先流式归档run及来源/模型SHA锁（不复制权重），验证流SHA/gzipCRC/tar精确集合/逐成员/安全解包读回，再用冻结程序重算分题集与子集准确率、配对检验、各交接及累计实际KV/Full位置字节比、计时和去重RID费用。字节比不等于实测GPU/FLOP/网络收益，重建单臂耗时不冒充独立实测。中途只报计数、真实进程和新异常，不算未完成准确率，不做AI语义审计。用户暂停立即停止本run自有推进；无定时自动化。

MPS多控制器及UUID使用依据NVIDIA官方文档：https://docs.nvidia.com/deploy/mps/quick-start.html ，https://docs.nvidia.com/deploy/topics/topic_5_2_1.html 。使用服务器已验证的legacy控制接口，不因文档新版本更改本机驱动或实现。
