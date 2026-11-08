# MPS端到端前缀与原cap8192 smoke

短MPS同B1对照78.86总token/s，4题×2轮全部匹配已核对HF参考，相比此前同代码无MPS四进程45.46探索性提升73%。服务已退出。按用户条件授权，进一步验证而非立刻重跑225。

新私有GPU0 MPS命名空间，旧真实R packet只读。先使用冻结stage/probe/final_select函数在MPS下重建原顺序首4题prefix，逐元素比较每层每头K/V、position、RNG、prompt IDs和absolute end，norm目标需匹配。任何不匹配即BLOCKED，不生成长答案。

若全部匹配，固定首2题做原HF generate原cap8192、原temperature.6/top_p.95/top_k20，原observe/position fix、不强行EOS、不缩cap。两独立worker同屏障起跑；按原协议记录自然EOS/截断、原位置36层SDPA/实际KV、first128与已核对参考相同、原账本费用及进程归属。模型15SHA/BF16CUDA/实际embedding mean-input-L2独立核对。总保守上界50000positions，budget115M；首次util<=30/free>=70，后续free>=21GiB，加载串行，最多2模型。无sudo/compute mode/功率/其他卡修改，服务最后关闭。

本阶段不算准确率，不追加/覆盖旧结果。不创建自动监控。通过后只准备新实验启动草案，具体正式方法/题集范围仍须明确；四题前缀和两题长生成不能外推所有题都逐token等价。
