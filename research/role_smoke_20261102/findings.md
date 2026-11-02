# 实施边界

旧CPU role_contrast为lambda1中心化误差原型，不用于此次生成。新R直接匹配self/sink及三个来源块；T匹配总和。所有头长度一致，头内原始KV gather，不拟合V。Judger没有latent步骤。

前两交接复用冻结receiver选择器；第三交接从当前仍可用的全部prompt行中保留96行，另保护30latent及4sink，共130。每个KV头汇入四个GQA查询头的最后16查询；不先平均读出。

假角色只错移真实来源块的连续边界，保持每块行数，sink/self不改；不会使用简单角色名置换。工程试跑不作正确率优劣检验，也不改动历史答案。
