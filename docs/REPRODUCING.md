# 复现范围

## 可以直接离线核对

`python scripts/verify_snapshot.py`核对发布源的SHA和所有3584个唯一键；`python scripts/reproduce_results.py --check`从逐题正确标记和实际KV字节重新生成16格统计并与summary比较。这是统计一致性检查，不是独立重新生成答案。原答曾按冻结解析器在私有归档中重算，当前公开CSV不含原答，因此不能从本仓库独立重做答案解析。

`python -m unittest discover -s tests -v`检查逐题指标重复/缺失拒绝、压缩预算与历史latent/sink保护、逐层逐头不同选择、原K/V gather和RoPE位置映射。CPU张量测试采用轻量缓存容器，不证明真实GPU推理兼容性。

## 重新生成答案所需的外部材料

1. Linux、Python3.10.21及`requirements.txt`所列的原云端版本；安装匹配硬件的PyTorch CUDA构建。
2. 自行取得Qwen3-8B权重和获准访问的GPQA数据。`snapshot/bundle/runtime/model_lock.json`记录原15文件SHA；其中路径是历史路径，不是可直接使用的配置。
3. 原始冻结题单及选项顺序。公开`cohort_ids.json`只记录ID和题文哈希；历史复用题的选项顺序必须完全一致，仅重新按seed洗牌不足以保证同一输入。
4. HeadOBF外部依赖可通过`python scripts/fetch_obf.py`按固定URL下载并核对SHA。其源仓库在锁定revision未找到LICENSE，因此本仓库不重新分发这些源文件，也不授予其使用权。
5. 私有部署的`plan.json/questions.json/queue.json/reuse_index.json/reused_results.jsonl`及既有共享账本。历史`cloud_common.py`、manager和runtime有固定布局与历史路径，必须在隔离工作区配置并生成新的管理/源锁。不要修改或清零正在使用的生产账本，也不要直接运行历史manager去复活旧run。

本版交付精确历史源码与统计复现，**不承诺下载后可直接启动GPU**。将路径参数化并验证的通用启动器属于后续工程工作；不得把该适配路径未经验证地称为本轮已跑的代码。

## 原归档

新run及复用来源引用的完整归档保存在本地，未上传GitHub。SHA256：`11393167cf8d3789d7f6dd492027582800b223da2ed8f91fd9fa1ac4ae067eea`，195962206字节、6941成员。流SHA、gzipCRC、tar精确集合及安全全解包逐成员SHA通过。此摘要不是公开原始回答或独立复现证明。
