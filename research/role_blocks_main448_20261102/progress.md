# 进展

- 已确认用户本轮新授权及本地AGENTS规则；旧main448与两题试跑均完整退出、归档。
- 实时账本actual71705806、unknown33278、reserved0；当前仅GPU1达到准入条件。
- 正在准备冻结队列、源锁、六条复用与旧四方法快照。
- 已部署role_blocks_gpqa_main448_s42_l10_20261102T142146Z，run_lock SHA70828faf23244d950cc1a562af9253773a2848ef962f718ba6c40e5e92468de9。16小文件锁与2个大只读快照锁。
- 核心函数AST及selector字节一致；新CPU4项通过、7个Python文件逐一编译通过；旧30源SHA与3584指标验证通过。
- PowerShell不展开py_compile通配符，首次编译命令报Invalid argument，已改用pathlib逐文件编译通过，无云端影响。
- 真实tokenizer预检0model forward：新增最大位置上界12313990，max original context19770，budget计入后84053074<115M。
- manager650899、GPU1worker650904实际运行，UID1033/start_ticks183929670/CVD1，尚在加载。其余卡等待原准入条件；当前6/1344全部为复用，新增0，446pending，无held/重复/越界。
- 30分钟heartbeat gpqa-448已ACTIVE；正常等待安静，新故障/恢复/重要变化/完成通知。首次automation调用缺destination，已用destination=thread正确创建，无重复监控。
- 冻结reuse_index的parent_main_run字段最初取提取目录basename为run：它只是描述标签。parent_source_lock及archive SHA、1792逐键快照核对始终正确。未热改冻结索引；management/source_reference_clarification.json明确真实父run及所有来源路径，未来部署器标签取正确basename。

## 2026-11-02T14:55Z首次巡检：GPU1正式接入

- 一次capture：12/1344唯一（新增6、复用6），剩1332；queue 2 complete/1 running/443 pending/0held，无重复或越界；1792基线和6条父复用快照SHA均通过，manager650899不变。
- GPU1 PID650904/UID1033/start_ticks183929670/CVD1完成15文件SHA、全BF16CUDA16381470720B、identity norm及target norm1.3758320808410645。加载/哈希/初始化464.30秒，未重启模型。
- 独立审计随后已核验该PID本attempt 7条新答案，覆盖T/R/R-shuffle；raw输入/输出RID与generated_ids SHA、36×8×130位置npz SHA、原RoPE与全部latent/sink保护、实际19169280B及源KV gather运行时证明通过。证明formal_gpu1_pid650904_verified.json两地保存，父6不冒充新worker答案。
- 捕获时账本actual71748150/unknown33278/reserved8504，单个健康Judger RID预留正常；审计时仍1个健康在途RID，不手工结算，无重复费用结算和新unknown。
- 当前仅GPU1 worker接入，其余卡由既有manager按fresh原门槛继续检测；没有故障/held，不重载、重排或改变任何科学参数。首次正式接入仅通知一次，之后同样健康进展安静。

## 2026-11-02T15:55Z巡检：GPU3加载失败已归档

- 一次capture为39/1344唯一（新增33+复用6），剩1305；11case complete/1running/434pending/0held，无重复或越界、快照SHA完好。manager650899及健康GPU1 PID650904持续运行。
- GPU3 PID695578（UID1033/start_ticks184305622/CVD3）以fresh util22%、free22947/24564MiB准入后，在from_pretrained加载时CUDA OOM并退出。失败时仅38MiB剩余，未完成模型加载，未领题、0请求/0raw/0终态答案/0自有RID预留；不能把NVML报错中的宿主PID当容器PID。
- 没有手工结算、没有新增unknown（仍33278），不制造held或重排已完成答案。GPU3保持本run已尝试状态，不盲重载，不改变分配器或科学参数，不碰其他进程。
- 完整停止现场、日志、16锁文件、两份原答快照、当时队列/manager/workers/账本归属证明和bundle37已本地流式归档；无预训练权重。archive SHA66978e559a809f70d504814be668e2c65a98152abb699f20026a22c9211b45c8，22675639B/70成员，远端流SHA/gzipCRC/tar精确集合/逐成员与全安全解包SHA全部通过，failure_pid695578_verified.json已两地保存。
- 本次新增归档器archive_failed_worker.py仅归档停止的自有worker；capture摘要改为截短异常，不回显完整traceback，完整inspect仍私有落盘。源锁70828faf不变。新故障仅通知一次，后续健康进展或相同等待保持安静。

## 2026-11-03T00:25Z巡检：GPU0/2正式接入，三卡运行

- 一次capture为292/1344唯一（新增286、复用6），剩1052；94case complete/3running/349pending/0held，无重复/越界，两个原答快照SHA通过。manager650899、GPU1 PID650904及模型不变。
- GPU2 PID1039151/UID1033/start_ticks187320842/CVD2、GPU0 PID1039794/UID1033/start_ticks187326404/CVD0均完成模型加载和本attempt T/R/R-shuffle三臂新答案。独立审计分别9条、8条正式新提交，15文件SHA/fullBF16CUDA16381470720B/identity norm、raw RID/token SHA、实际36×8×130位置SHA与原KV gather/原RoPE/全部latent和sink保护通过；formal_gpu2_pid1039151_verified.json及formal_gpu0_pid1039794_verified.json两地保存。
- 三个健康Judger RID在途预留25497正常，账本捕获actual73025645/unknown33278，无新unknown、无手工或重复结算。没有重载、迁移或重新派发已完成任务。
- GPU3旧PID695578加载失败继续保留原档，不重试；其他候选仍由manager原门槛检测。首次正式三卡接入本次通知一次，之后相同健康进展安静。科学锁70828faf不变。

## 2026-11-03T02:25Z巡检：GPU6正式接入，GPU7首题运行

- 一次capture为416/1344唯一（新增410+复用6），剩928；135case complete/5running/306pending/0held，无重复/越界，源与两快照SHA通过，manager650899不变。
- GPU6 PID1132908完成三臂，独立核验本attempt 3条新答：15模型文件SHA/fullBF16CUDA16381470720B/identity norm、raw RID/token SHA、36×8×130原位置SHA、实际KV字节及原KV gather、全部latent/sink保护通过。formal_gpu6_pid1132908_verified.json两地保存。
- GPU7 PID1130369/UID1033/start_ticks188146611/CVD7以fresh util18%、free18376/24564MiB准入，已完成加载并提交首个T新答，正在首题R生成；尚未完成三臂独立核验，不称正式三臂worker。inspection_20261103T0225Z.json记录真实身份/原准入/持续native及raw进展。
- 当前0/1/2/6四张卡完成正式三臂核验，GPU7为第五个健康运行中的worker且待三臂验证。健康既有模型均未重启或迁移。五个健康在途Judger预留42646正常；捕获actual73592965/unknown33278，无新故障/unknown或手工结算。
- 旧GPU3加载失败仍保留完整归档不重试；其他候选继续按原规则。仅本次正式扩展通知一次；正常首题生成与后续同样健康进展安静。源锁70828faf不变。

## 2026-11-03T02:55Z巡检：GPU7独立三臂通过，五卡正式运行

- 一次capture为466/1344唯一（新增460+复用6），剩878；151case complete/5running/290pending/0held，源及两个原答快照SHA通过，无重复/越界，manager650899不变。
- GPU7 PID1130369/UID1033/start_ticks188146611/CVD7独立核验10条本attempt新答，T/R/R-shuffle均通过：15文件SHA/fullBF16CUDA16381470720B/identity norm、raw RID/token SHA、实际36×8×130位置SHA/实际KV字节及源KV gather、原RoPE和全部latent/sink保护通过。formal_gpu7_pid1130369_verified.json两地保存。
- 当前GPU0/1/2/6/7五卡正式健康，均未重启/迁移。五个在途Judger RID预留42385正常，捕获actual73829869/unknown33278，无新故障、unknown或手工/重复结算。旧GPU3失败继续归档保留、不重试。
- inspection_20261103T0255Z.json保存全部五worker真实身份及native/trace/raw进展，两地UTF8同步。首次GPU7正式接入本次通知一次，此后相同健康进展安静。源锁70828faf不变。

## 2026-11-03T03:25Z：GPU6运行OOM与管理器权限故障修复

- 一次capture为503/1344唯一（新增497+复用6），剩841；164case complete/4running/277pending/1held，无重复/越界，原答快照SHA完好。
- GPU6 PID1132908在gpqa_diamond_test_192的R-shuffle Judger阶段CUDA OOM退出，原11条正式答案完整保留；该题T/R已提交。失败RID1132908-1793674477464879616由Engine一次结算actual8063、unknown428（bound8491），36请求/72 raw记录完整，自有RID预留0；unknown从33278至33706，不再结算。
- manager650899因读取/proc/1132908/environ PermissionError退出。已完整停止归档该worker、原管理器日志/故障与状态、锁文件/快照/队列及bundle37，无权重：SHA6732574ec52d9c19d1f43b4269bf9d4617ec863e7c5efb71320fe703f091c780，23471472B/91成员，流SHA/gzipCRC/tar精确集合/逐成员及安全全解包SHA全通过，failure_pid1132908_verified.json两地保存。
- 明确管理修复：management_identity.py先核对UID，PermissionError/消失进程仅返回不可认领，不读取异UID环境；原PID/UID/start_ticks/argv/CVD判定保留。4项CPU测试通过。manager_recovery.py只在管理进程包装identity并委托原manager.py，原16锁文件/worker/算法/参数均不改。捕获器使用同一管理guard避免旧退出PID权限异常。
- 新manager1180054/UID1033/start_ticks188635966已实际启动并健康接管GPU0/1/2/7原驻留PID1039794/650904/1039151/1130369，无模型重启/迁移、无GPU6重载。完整故障归档后，四个健康驻留worker且其实际可用显存高于失败时5.88MiB，构成接续资源改善；仅一次将case_0151从held转高优先pending，methods仅R-shuffle、recovery_attempt=1，T/R不再生成。
- 复查new_unique502（加复用6为508），无重复；新管理器无故障，缺失R-shuffle尚pending，不能称恢复完成。新Manager及派发证明manager_recovery_20261103T0325Z.json、recovery_check_20261103T0325Z.json两地保存；原manager_failure已移到management保留原件，完整归档同时保留原路径快照。
- GPU3旧加载失败和GPU6本次运行失败均不盲重载，不修改allocator，不碰其他进程。旧未知费用一次结算已证明，健康四卡在途预留不手工处理；本次新故障及管理修复/缺失臂派发通知一次，后续实际补齐再记录，若再失败保留新现场不盲试。
