每30分钟检查并推进本聊天2026-11-03明确授权的注意力选位G实验，仅覆盖暂停时共同225题。重要：原T/R/R-shuffle main448 run仍为真人暂停状态，绝不恢复旧队列、重跑旧答案或启动其他实验。

代码：D:/liuxin_workspace/ICML2027/Enhaced_LatentMAS_ADvanced/research/attention_control225_20261103。唯一G私有包：D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_attention_control225_20261103。先读cloud_active.json、代码protocol.md/task_plan.md/progress.md和最新management。远端run以G active为准；SSH DevContainer，Python=/home/liuxin/.conda/envs/kv-graph/bin/python，root=/home/liuxin/ICML2027/experiments/latentmas_cumulative_history_20261025。

一次capture_current.py完整UTF8 inspect落本地，stdout仅计数/真实进程/异常摘要，不输出原答或全JSON。健康加载/生成/相同等待安静；仅新故障、实际恢复、首次正式接入、完成或需真人处理时通知一次。不做反复长轮询、子代理、AI语义评分或未经请求的未完成准确率。

固定G225唯一键(dataset,qid,steps10,methodG)，Diamond106、非Diamond119；旧七方法1575条只读参考，最终8方法1800键，Diamond848和非Diamond952分别报告，不能相加当独立重复。已看过阶段结果，机制探索而非新盲测。冻结seed42/latent10/Qwen3-8B BF16/norm_only、15权重SHA、identity与mean-input-L2 norm1.3758320808410645、作者prompt/解析器、cap8192/context32768、temperature.6/top_p.95/top_k20。前两receiver交接冻结，G每题核对同题T的prefix选位SHA，失配保留现场停止该worker。末次同64query、同<=192 bank，只按注意力质量选88自由prompt+8强制Refiner末prompt；全部30latent+4sink保护，36×8×130位置、实际19169280B。原K/V gather/原RoPE，无V拟合回填或角色目标；不调参或加题/seed/latent40。V只做选后诊断。

唯一入口manager_recovery.py委托manager.py。GPU0–7候选最多6自有worker，一卡完整模型，每10秒逐卡fresh util<=30%、free/total>=70%；健康不迁移重筛。每卡本run自动一次，失败不盲重载。GPU3/4/6旧OOM须独立实证clearance，GPU5须达到既有24081MiB改善证据，不凭相同快照放宽。只认真实容器PID/UID/start_ticks/argv/CVD和RID owner，不认NVML宿主PID、不碰他人。新worker加载不冒充正式提交；本attempt实际新G答案后运行verify_formal_worker.py --pid PID，独立15权重SHA/BF16CUDA16381470720B/norm、raw RID/token SHA、前缀同T位置SHA、原KV gather/36×8×130原位置及latent/sink保护验证通过后才称正式。证明本地和远端management保存。

共享root/cumulative_budget.json及budget.lock沿用：actual+reserved+独立unknown<=115000000，全部probe/失败收费，不清零迁移、加调用硬限或重复手工结算；启动准确边界见launch_preflight.json（0model forward）。健康在途预留正常。启动/捕获/完成核对源锁与两个参考快照SHA，不每题哈希大参考原答。故障只处理对应worker，完整停止现场、费用和归属保留；archive_failed_worker.py本地归档，核验远端流SHA/gzipCRC/tar精确集合/逐成员/安全全解包SHA后，有明确修复或实际资源改善才能只恢复真实缺失G键；已提交不重跑、加载失败0任务不造held。无pending正常tail退出非故障。实际ENOSPC才清理已备份自有停止文件，先证实路径/归属/无引用，不动当前/原暂停run/模型/锁/账本。

完成须G225精确唯一、无held/failed缺失/pending、参考1575原答和源锁完好、8方法1800键覆盖、自有worker/manager全退出、自有RID预留0。完整本地归档G run、来源引用、bundle37/模型SHA锁（不复制权重），校验远端流SHA/gzipCRC/tar精确集合/逐成员/安全全解包SHA。之后冻结程序重算225/106/119八方法准确率、实际末次KV保留/减少比、去重RID费用；主比较G对T配对2×2和双侧精确McNemar alpha.05。探索性，不显著不等效；时间辅助，不把字节比当通信/GPU/FLOP实测收益。中文UTF8同步本地及远端root/management/run_name和run/management，本地Git提交不push。全部G运行/归档/汇总完成后删除本监控，原main448仍暂停。真人明确暂停立即停止G自有运行和自动推进，明确恢复才继续。
