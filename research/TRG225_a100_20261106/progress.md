# 记录

2026-11-06：用户明确“不跑R-shuffle，其他三个补一下”，范围T/R/G225。旧main448/G/两题pilot保持冻结，不热改旧锁，不清零累计账本。新GPU1最多4worker，初始仍须util<=30/free>=70%，后续自有驻留模型允许利用率豁免，剩余显存至少21GiB/待加入worker且上一个加载完成。21GiB来自试跑约16GiB分配峰值加KV/碎片余量的资源保护，不改科学参数。失败不自动重载，保留已有键与费用。

实际新run TRG225_a100_s42_l10_20261106T155639Z，锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f。32文件+bundle37校验、8个核心函数AST与试跑一致、T/R/G选位分派CPU检查及671缺失键/零forward预检通过；保守上界7087329，启动预算74917051/0/58106。

GPU1逐次加载4worker完成，真实UID1000/start_ticks/argv/CVD1均独立核验，模型15文件SHA、BF16CUDA16381470720B、identity及target norm1.3758320808410645全部通过，现场无失败。首两个worker分别已有1条R新提交，raw RID/tokenSHA、36x8x130位置及原KV/保护行通过，但尚无本attempt全部T/R/G，不称正式三标签worker。两道pilot复用题的R均严格匹配pilot前缀位置并提交，pilot4条不重跑。

最新启动验收6/675唯一（新2+复用4），2case complete/4running/219pending/0held；实际账本74932122/reserved33919/unknown58106，在途预留正常不手工结算。一次完整UTF8 inspect私有落地；后续仅轻量真实进程/计数用于启动检查。startup_verified.json两地保存。监控gpqa-448已更新ACTIVE、每40分钟、仅新TRG225run；旧所有run/原始答案不改。

2026-11-06T16:38Z首轮40分钟巡检：一次完整inspect私有落地，20/675唯一（新16+复用4），剩655；5complete/4running/216pending/0held，无重复越界，源及参考/复用快照SHA通过。manager28169与四真实worker健康。四worker独立新attempt核验：28172的8条与28985的4条已覆盖T/R/G并正式通过；28440检查3条已有R/G、待T，28695检查2条已有T/R、待G，不能冒称全部正式。预算capture74997837/reserved33899/unknown58106，健康在途预留正常，无手工或重复结算，无新故障、未算未完成准确率。证明及中文接续两地保存，监控继续ACTIVE40分钟。首次本地管理记录命令引号产生SyntaxError，未执行任何科学操作；改用here-string后保存成功。

2026-11-06T17:18Z巡检：一次完整UTF8 inspect私有落地，39/675唯一（新35+复用4），剩636；11complete/4running/210pending/0held，源及全部参考/复用快照SHA通过，无重复或越界。manager28169及四worker真实身份与启动一致，无新故障。28440的8条本attempt新答、28695的5条分别完成T/R/G全部独立核验，raw RID/token SHA、36x8x130位置/保护行/实际KV及native位置通过；至此四worker均已正式三标签核验（此前28172和28985已通过），两地证明保存。账本actual75097100/reserved34189/unknown58106，健康在途预留正常，无手工或重复结算，无未完成准确率。继续ACTIVE40分钟，常规健康进展不打扰。

2026-11-06T17:58Z巡检：一次完整UTF8 inspect私有落地，59/675唯一（新55+复用4），剩616；18complete/4running/203pending/0held，无重复越界，源锁fa21c45d及所有参考/复用快照SHA完好。manager28169及四正式worker28172/28440/28695/28985健康，各提交15/16/10/14，均较上轮推进，无新故障。实际RID预留归属与真实存活worker/本run一致；账本75195247/reserved34260/unknown58106，在途正常，无手工/重复结算，无未完成准确率。中文管理接续两地保存，原科学锁和暂停run不变，ACTIVE40分钟继续。

2026-11-06T18:38Z巡检：一次完整UTF8 inspect私有落地，80/675唯一（新76+复用4），剩595；24complete/4running/197pending/0held，无重复越界或新故障。源锁及所有快照SHA完好，manager28169与四正式worker真实身份健康，各提交18/21/17/20；RID预留均属于真实自有在途worker。账本75307948/reserved34920/unknown58106，无手工结算，继续ACTIVE40分钟，两地接续保存。

完成阶段final_summary.py已在b974247冻结SHA两地保存；没有读取/计算当前正确率。须完整675精确键、全部队列完成/进程退出/预留0及归档通过，才调用冻结作者解析器复算。225/D106/nonD119分别报告，仅all225 R对T是预登记primary，子组和G比较均探索性；无shuffle不主张角色边界因果。KV是实际保留字节对原位置未压缩参照的加权比，费用按新run/pilot/前置A100失败的settled RID去重并对账本增量。4项合成测试与30源/3584公开快照复核通过，0模型调用。首次管理记录误用inspect basename导致远端FileNotFoundError，未改科学文件；改用active绝对路径后保存成功。

2026-11-06T19:18Z巡检：一次完整UTF8 inspect私有落地，109/675唯一（新105+复用4），剩566；35complete/4running/186pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交25/27/26/27。实际RID预留均归属本run存活worker；账本75419606/reserved34080/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T19:58Z巡检：一次完整UTF8 inspect私有落地，140/675唯一（新136+复用4），剩535；45complete/4running/176pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交33/33/34/36。实际RID预留均归属本run存活worker；账本75538998/reserved34718/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T20:38Z巡检：一次完整UTF8 inspect私有落地，165/675唯一（新161+复用4），剩510；54complete/4running/167pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交38/43/37/43。实际RID预留均归属本run存活worker；账本75653350/reserved34229/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T21:18Z巡检：一次完整UTF8 inspect私有落地，188/675唯一（新184+复用4），剩487；62complete/4running/159pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交42/49/42/51。实际RID预留均归属本run存活worker；账本75750767/reserved34128/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T21:58Z巡检：一次完整UTF8 inspect私有落地，211/675唯一（新207+复用4），剩464；69complete/4running/152pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交52/52/47/56。实际RID预留均归属本run存活worker；账本75856360/reserved25734/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T22:38Z巡检：一次完整UTF8 inspect私有落地，227/675唯一（新223+复用4），剩448；75complete/4running/146pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交58/55/49/61。实际RID预留均归属本run存活worker；账本75943573/reserved33779/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T23:18Z巡检：一次完整UTF8 inspect私有落地，251/675唯一（新247+复用4），剩424；83complete/4running/138pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交64/62/55/66。实际RID预留均归属本run存活worker；账本76049394/reserved34287/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-06T23:58Z巡检：一次完整UTF8 inspect私有落地，270/675唯一（新266+复用4），剩405；88complete/4running/133pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交69/65/59/73。实际RID预留均归属本run存活worker；账本76151288/reserved34207/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T00:38Z巡检：一次完整UTF8 inspect私有落地，290/675唯一（新286+复用4），剩385；95complete/4running/126pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交74/68/64/80。实际RID预留均归属本run存活worker；账本76249096/reserved33963/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T01:18Z巡检：一次完整UTF8 inspect私有落地，318/675唯一（新314+复用4），剩357；105complete/4running/116pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交78/76/73/87。实际RID预留均归属本run存活worker；账本76364992/reserved25625/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T01:58Z巡检：一次完整UTF8 inspect私有落地，341/675唯一（新337+复用4），剩334；112complete/4running/109pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager28169与4个真实自有worker正常，各提交87/81/79/90。实际RID预留均归属本run存活worker；账本76466521/reserved34009/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T02:23Z巡检：一次完整UTF8 inspect私有落地，352/675唯一（新348+复用4），剩323；115complete/8running/102pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交89/84/81/92/2/0/0/0。实际RID预留均归属本run存活worker；账本76537250/reserved51446/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07 GPU0扩展实际完成：真人明确授权GPU0/1各4、总8并发。新manager96471接管原GPU1驻留PID28172/28440/28695/28985，未重启迁移模型；GPU0 PID96474/96894/97300/97656四份模型均已ready，独立15SHA/fullBF16CUDA16381470720B/identity/mean-input-L2验证通过。GPU0目前2条新答，R/G原始RID/token SHA、36×8×130原位置、原KV gather/RoPE、保护行/实际19169280B核验通过，T尚在途；其余新worker尚待本attempt三臂新答，不能冒称全部正式三臂审计完成，后续按verify_worker.py继续。
一次完整UTF8 inspect：352/675唯一（新348+复用4），剩323，115complete/8running/102pending/0held，0重复/越界/新故障。源锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f与两参考快照完好。账本actual76537250/reserved51446/unknown58106，健康预留不手工结算，未算未完成准确率。
CPU资源观察：原GPU1四进程全部线程挤同一CPU70，各约25%CPU，允许8CPU且cgroup无节流。仅对已核验自有model-ready worker的线程在当时容器允许CPU集合内分开，不改线程数/科学参数；CPU集合会动态变化，不盲套过期核号。最终8进程亲和性分开后5次资源采样：GPU0 util100–100%、GPU1 99–100%，两卡P0/SM1410=max1410MHz；GPU0用70036/free11118MiB，GPU1用72658/free8496MiB，温度59/57C。保留生成余量，不提高功率或时钟，不把利用率/字节比解释为实测吞吐倍数，不据此宣称A100单流比4090快。
management/gpu01_expansion_verified.json、gpu0_takeover_verified.json、gpu01_management_lock.json、cpu_distribution_verified.json及旧管理/队列快照两地保存；完整旧manager日志保留。原40分钟监控已更新两卡8并发与新管理入口，旧run继续暂停。

2026-11-07T02:38Z巡检：一次完整UTF8 inspect私有落地，373/675唯一（新369+复用4），剩302；123complete/8running/94pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交91/87/84/93/6/1/3/4。实际RID预留均归属本run存活worker；账本76628796/reserved42714/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T02:38Z扩展核验：GPU0 PID96474/97300/97656分别6/3/4条本attempt新答，T/R/G三臂独立15SHA/BF16CUDA/norm、raw RID/token SHA、36×8×130原位置/实际KV与保护行核验通过；PID96894只有G一条，结构核验通过且R/T仍健康在途，不冒称该worker正式三臂完成。八个worker及manager96471健康，management模块/资源策略SHA与科学锁均核对通过；checkpoint两地保存。按授权保持安静，不做重复长轮询或未完成准确率。

2026-11-07T03:18Z巡检：一次完整UTF8 inspect私有落地，411/675唯一（新407+复用4），剩264；135complete/8running/82pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交94/94/89/99/13/4/8/6。实际RID预留均归属本run存活worker；账本76808889/reserved59703/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T03:18Z核验：GPU0最后待三臂worker PID96894已4条本attempt新答，T/R/G均独立核验通过（15文件SHA/BF16CUDA/identity/norm、raw RID/token SHA、36×8×130位置/实际KV/原KV gather/RoPE/保护行）。至此GPU0四份与原GPU1四份均有各自正式三臂证明；不以复用4条充当新提交。management模块/资源策略及科学锁SHA复核通过，证明两地保存。八worker和manager96471健康，不重复通知常规进展，监控40分钟继续。

2026-11-07T03:38Z用户询问速度：只读诊断，未改运行参数或模型。一次inspect437/675（新433+复用4），剩238，144complete/8running/73pending/0held，无新故障。最近30分钟样本36条完成答案，生成token均值3515/中位2682/最大8192；13条>=4096，2条触顶8192。带现有审计钩子的Judger均值389.7秒/中位279.8秒/最长1061.3秒，单流tokens/s中位9.41。Judger占重建单臂时间总和92.54%；共享prefix中位5.80秒，独立张量审计中位0.34秒。另一次相邻窗口按方法统计selector中位T33.98秒/R34.59秒/G3.68秒；完整三臂case实际队列墙钟中位1387.1秒（13题样本），不是三倍重复计入共享prefix。两GPU util99/100、SM1410=max1410MHz，活跃Judger持续forward，非卡死。CPU集合动态变化，本次只观察，不重复套过期affinity。诊断含并发/审计/调度开销，未隔离纯模型或硬件基准，不宣称A100相对4090速度倍数；未计算未完成准确率。按原协议继续，不中途改8192/prompt。完整数值诊断两地保存。

2026-11-07T03:58Z巡检：一次完整UTF8 inspect私有落地，458/675唯一（新454+复用4），剩217；151complete/8running/66pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交97/100/91/106/22/12/13/13。实际RID预留均归属本run存活worker；账本77017137/reserved59623/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T04:38Z巡检：一次完整UTF8 inspect私有落地，499/675唯一（新495+复用4），剩176；164complete/8running/53pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交101/109/94/111/27/14/17/22。实际RID预留均归属本run存活worker；账本77217020/reserved59871/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T05:18Z巡检：一次完整UTF8 inspect私有落地，547/675唯一（新543+复用4），剩128；180complete/8running/37pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交109/113/98/113/34/21/25/30。实际RID预留均归属本run存活worker；账本77428507/reserved59766/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T05:58Z巡检：一次完整UTF8 inspect私有落地，581/675唯一（新577+复用4），剩94；191complete/8running/26pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交113/116/102/116/37/28/31/34。实际RID预留均归属本run存活worker；账本77604566/reserved59787/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T06:38Z巡检：一次完整UTF8 inspect私有落地，612/675唯一（新608+复用4），剩63；202complete/8running/15pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交116/119/105/124/39/34/35/36。实际RID预留均归属本run存活worker；账本77798558/reserved59914/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T07:18Z巡检：一次完整UTF8 inspect私有落地，647/675唯一（新643+复用4），剩28；213complete/8running/4pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager96471与8个真实自有worker正常，各提交119/130/108/127/41/38/37/43。实际RID预留均归属本run存活worker；账本77993876/reserved68169/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07T07:58Z巡检：一次完整UTF8 inspect私有落地，675/675唯一（新671+复用4），剩0；225complete/0running/0pending/0held。源锁与快照SHA完好，无重复/越界/新故障，manager已退出与0个真实自有worker正常，各提交121/133/114/129/45/42/39/48。实际RID预留均归属本run存活worker；账本78143752/reserved0/unknown58106，健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。

2026-11-07完成：675精确键，全部进程正常退出、预留0；完整671新答结构审计/费用对账/本地归档五重验证通过。冻结程序T104/225、R114/225、G114/225；R对T精确p=.220328未显著。Diamond与非Diamond分开报告见final_report.md，不宣称等价或边界因果收益。中文记录和结果两地保存，监控gpqa-448已删除，无新增实验。
