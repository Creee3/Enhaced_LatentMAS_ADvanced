# G同题注意力选位对照：锁定协议 v1

授权：2026-11-03用户“你先去启动G的实验吧”。仅新增G225题；原T/R/R-shuffle未完成队列保持暂停。所有已生成方法原答只读复用，不补跑原448缺口。

题集：暂停快照七方法共同225个(dataset, question_id, latent_steps=10)；Diamond106、非Diamond119。题单从原cohort按原顺序提取，来源锁与快照SHA存source_reference.json，七方法1575条参考原答另存只读baseline_results.jsonl。已看过阶段结果，属于探索性机制对照。

模型/生成：Qwen3-8B，15权重SHA，BF16完整CUDA参数16381470720B，identity/norm_only，mean-input-L2 target1.3758320808410645。seed42、latent10、冻结作者prompt/解析器、cap8192/context32768、temperature.6/top_p.95/top_k20均沿用原协议。

压缩：前两交接receiver完全冻结。末次同Judger探针、最后16query×4 GQA头，保留64个独立观测。各层各KV头对候选prompt累加注意力质量，沿用T的stable storage-order tie break与同<=192候选bank。保留注意力最高的88个自由prompt以及Refiner末8强制prompt；全部30latent+4sink保护，最终36×8×130位置和实际19169280B。保持原K/V gather及原RoPE；无V拟合、回填或角色目标。V仅用于选择后诊断，不影响排序。

差异隔离：新worker的stage/probe/judge/final_select/validate/timed/args_for函数AST与原worker完全一致；仅方法列表改G、替换selector import。旧科学文件/原run_lock不修改。若前缀压缩位置SHA与同题T不同，标为不可直接比较，保留现场、停止对应worker检查，不继续把该样本当匹配证据。

成本：所有真实forward/probe/失败均计入同root/cumulative_budget.json，actual+reserved+独立unknown<=115M。零model-forward的真实tokenizer逐题预检，绝不清零或重复结算。最终答案225条不等于总forward225次。两端CUDA同步，选择计时含诊断；仅报告末次KV保留比，不将字节比当端到端收益。

运行：manager_recovery.py为唯一显式启动入口，委托manager.py；UID优先且PermissionError返回不可认领。GPU0–7候选、最多6自有worker、每10秒逐卡fresh util<=30%、free/total>=70%，一卡完整模型；健康不迁移/重筛；每卡本run自动一次。原GPU3/4/6失败须单独实证clearance，GPU5满足既有24081MiB改善门槛。失败完整保留并归档验证后，有明确修复/资源改善才允许只补缺失键；不盲重载、不触碰他人。

比较：全部225个G唯一键完成且8方法精确1800键覆盖后评分，统一冻结解析器，报告225/106/119三个scope。主比较G对T的配对2×2与双侧精确McNemar alpha .05；分层探索性，不独立重复。不显著不等效，不据中途准确率调参或挑题。T/R/shuffle/旧四方法均不重跑。

完成：G225唯一、无重复/越界/缺失、参考快照完好、自有worker/manager全退出、自有RID预留0。完整本地流式归档并核验远端SHA/gzipCRC/tar精确集合/逐成员及安全全解包SHA，保留源锁/bundle37/模型SHA锁而不复制权重；之后汇总，中文UTF8两地保存，本地Git提交不push。真人暂停立即停止自有进程和自动推进。
