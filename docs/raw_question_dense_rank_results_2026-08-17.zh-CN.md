# Raw Question Dense Rank 结果（2026-08-17）

## 问题

此前 7 个 persistent retrieval miss 的 gold 文档在官方 `4096-token` 文档输入
recipe 下均保留答案，但 48 条 Agent 生成查询的 dense top-20 仍为 `0/7`。本轮只问
一个更小的问题：查询拆解是否丢掉了原始问题中的组合约束？

最简单对照是每题只编码一次已经冻结的原始完整问题，使用完全相同的
Qwen3-Embedding-0.6B、query prefix、dense index 和 top-1000 搜索深度。builder 在
读取 gold 与历史 rank 前先持久化 7 份完整 top-1000 slate；scorer 随后才打开 gold
和已冻结 generated-query 结果。

## 运行边界

- host：`A6000_wh`
- device：CPU，`CUDA_VISIBLE_DEVICES=""`，`gpu_used=false`
- runtime：PyTorch `2.7.0+cu126`、FAISS `1.11.0`、Transformers `4.53.2`、
  Tevatron `0.0.1`，与 generated-query dense baseline 的核心版本一致
- dense query encode：7
- provider、online search、Judge call：均为 0
- sealed holdout：未访问
- model load：125859 ms；7-query search：7722 ms

安装包带有 CUDA build tag 不等于本轮使用 GPU；GPU 对进程不可见，实际执行设备为
CPU。

## 结果

| Query | Raw-question rank | Generated-query best rank | 关系 |
| --- | ---: | ---: | --- |
| 875 | missing | 109 | loss |
| 754 | missing | 981 | loss |
| 898 | missing | missing | tie |
| 710 | missing | 551 | loss |
| 873 | 13 | 78 | win |
| 869 | missing | missing | tie |
| 805 | 255 | 211 | loss |

汇总：

| 指标 | Generated queries | Raw full question | 预注册门槛 |
| --- | ---: | ---: | --- |
| gold-hit cases @20 | 0/7 | 1/7 | raw >=4/7，失败 |
| gold-hit cases @100 | 1/7 | 1/7 | raw >=4/7，失败 |
| gold-hit cases @1000 | 5/7 | 2/7 | diagnostic only |
| raw rank wins | - | 1/7 | >=4/7，失败 |

decision：`freeze_raw_question_dense`。

## 结论

原始完整问题不是一个普遍更好的 dense anchor。它只明显帮助 q873，却让 4 题的
gold rank 变差；generated-query 分解在 top-1000 反而覆盖更多 gold 文档。因此不能
把失败单独归因于“Planner 丢失原问题”，也不能把“始终追加 raw question”包装成
创新点。

当前证据更像是：原始长问题可能稀释局部实体信号，而分解查询又可能遗漏跨实体的
关系桥。下一步不是调 query prefix、深度或在这 7 题上扩充 query，而是先审计已有
bridge/hypothesis probe。审计发现另一组 3 个已知失败上已连续拒绝 obligation
rewrite、typed contrastive、draft-blind、firewall、Pro substitution 与
corpus-grounded induction。它们不应换名后在当前 7 题重跑。当前 cluster 改为
regression-only；下一步是在完整 175 题 development 上冻结运行 v10，先得到更可靠的
错误分布，再由新的主失败类选择机制。

这仍是 outcome-selected failure cluster 上的 posthoc gold-aware rank 诊断，不是
端到端效果、官方 accuracy、leaderboard 结果或模型能力提升。

## 审计产物

- registration：`benchmarks/browsecomp_plus_v0/raw_question_dense_rank_v0.json`
- implementation：`src/deepresearch_harness/raw_question_dense_rank.py`
- runner：`scripts/run_raw_question_dense_rank.py`
- tests：`tests/test_raw_question_dense_rank.py`
- ignored slate：`runs/browsecomp_plus_v0/raw-question-dense-rank-v0-20260817/slate.json`
- ignored audit：`runs/browsecomp_plus_v0/raw-question-dense-rank-v0-20260817/audit.json`
- audit SHA256：`c1472ff7b281447b67195123754f3f364bdfc11ea5f8d545cbebba8b83f7a001`
