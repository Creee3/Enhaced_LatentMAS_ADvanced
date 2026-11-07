# 记录

2026-11-07：GPU1查询P0/99–100%/SM1410=max1410MHz/显存1512=max1512MHz，温度56–57C，功耗约205–214W/300W，无当前功耗/热降频；CPU cgroup无节流。不能把100% GPU利用率等同于所有算术单元满载，也不能直接用不同并发/输出长度的时长宣称A100单token比4090快。

用户随后明确要求使用GPU0，fresh GPU0 util0%/free81153MiB/81920MiB；不触碰他人进程，不修改系统功率/时钟。资源例外仅用新management版本记录，科学锁仍fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f。

2026-11-07T02:13:40.442984+00:00：真人明确授权两张卡各4、总8并发。6项CPU准入边界测试通过。旧manager28169身份UID1000/start201557708/argv核对后计划SIGTERM退出；锁释放延迟时未绕过锁，复查实际释放后，新manager96471/UID1000/start205250690接管原GPU1 PID28172/28440/28695/28985，四模型未重启迁移。GPU0 slot1 PID96474正在加载。源锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f与原快照不改；管理模块/资源策略另行SHA锁，完整旧管理和队列快照两地保存，原日志保留。原40分钟监控已更新GPU0/1总8与新管理入口；健康在途不手工结算，未算未完成准确率。

发现并实测处理CPU资源瓶颈：原GPU1四worker及全部线程均限制CPU70，各约25%CPU，四进程合计约一个核；容器允许8个调度CPU，无cgroup CPU配额节流。仅对核对UID/start/argv/CVD的自有model-ready进程及其线程，在容器当时允许的CPU集合内分配不同核；不改线程数/科学参数/模型。CPU集合可能动态变化，证据保留当时集合，不盲套过期CPU编号。处理后GPU1即时util100%，GPU0三模型生成时util99%；这只是即时资源观察，不宣称同题吞吐倍数。第三份已ready，第四份PID97656加载中；前两份独立模型证明和前两条原始RID/token/选位/实际KV核验通过，尚未全部三臂提交，不冒称正式。

2026-11-07 GPU0扩展实际完成：真人明确授权GPU0/1各4、总8并发。新manager96471接管原GPU1驻留PID28172/28440/28695/28985，未重启迁移模型；GPU0 PID96474/96894/97300/97656四份模型均已ready，独立15SHA/fullBF16CUDA16381470720B/identity/mean-input-L2验证通过。GPU0目前2条新答，R/G原始RID/token SHA、36×8×130原位置、原KV gather/RoPE、保护行/实际19169280B核验通过，T尚在途；其余新worker尚待本attempt三臂新答，不能冒称全部正式三臂审计完成，后续按verify_worker.py继续。
一次完整UTF8 inspect：352/675唯一（新348+复用4），剩323，115complete/8running/102pending/0held，0重复/越界/新故障。源锁fa21c45d93c74ab80fbd3196acb5f389a679830600a731a86ca6d26144db040f与两参考快照完好。账本actual76537250/reserved51446/unknown58106，健康预留不手工结算，未算未完成准确率。
CPU资源观察：原GPU1四进程全部线程挤同一CPU70，各约25%CPU，允许8CPU且cgroup无节流。仅对已核验自有model-ready worker的线程在当时容器允许CPU集合内分开，不改线程数/科学参数；CPU集合会动态变化，不盲套过期核号。最终8进程亲和性分开后5次资源采样：GPU0 util100–100%、GPU1 99–100%，两卡P0/SM1410=max1410MHz；GPU0用70036/free11118MiB，GPU1用72658/free8496MiB，温度59/57C。保留生成余量，不提高功率或时钟，不把利用率/字节比解释为实测吞吐倍数，不据此宣称A100单流比4090快。
management/gpu01_expansion_verified.json、gpu0_takeover_verified.json、gpu01_management_lock.json、cpu_distribution_verified.json及旧管理/队列快照两地保存；完整旧manager日志保留。原40分钟监控已更新两卡8并发与新管理入口，旧run继续暂停。
