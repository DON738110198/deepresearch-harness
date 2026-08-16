# Passage Index Representation v0 结果

## 问题

三次有效 v10 baseline 中有 7/25 题持续没有召回 gold 文档。语料审计已证明：
18/18 gold 文档存在，7/7 题的正确答案逐字存在于这些文档。全文 BM25 上，相同
模型生成查询的最好结果为 top-20 `0/7`、top-100 `2/7`、top-1000 `6/7`。

因此本轮不改模型、query、Agent 数或搜索次数，只改一个变量：

```text
one full document per BM25 record
-> deterministic overlapping passage records
```

## 预注册

注册文件在提交 `987406f` 中先于实现和结果进入 `main`。冻结项为：

- 100,195 个 byte-verified source documents；
- `whitespace_v0`，每段 256 tokens，overlap 64 tokens；
- 每段重复最多 64 tokens 的 frontmatter；
- BM25 `k1=0.9`、`b=0.4`；
- 相同 trial-1 的 48 条成功生成查询；
- 每条查询内部最多取 200 个 passage hit，再折叠为最多 20 个唯一源文档；
- acceptance：7 个 outcome-selected case 中至少 4 个命中 gold document；
- provider、在线搜索、Judge call 全部为 0，sealed holdout 禁止。

这只是 gold-aware 离线筛选，不是 end-to-end 或 leaderboard 结果。

## 构建

| Item | Value |
| --- | ---: |
| Source documents | 100,195 |
| Source documents represented | 100,195 |
| Passage records | 2,715,518 |
| Corpus JSONL bytes | 4,586,083,582 |
| Lucene index bytes | 5,494,752,480 |
| Index build wall time | 592.96 s |
| Unindexable / empty / errors | 0 / 0 / 0 |

索引器已成功写完 2,715,518 条记录，但父 Python 进程读取 Windows 本地化日志时
发生 UTF-8 decode error。恢复流程没有重建：它先用 Lucene 打开 partial index，
验证 document count 和 sample document，再将其原地转正，并把 stdout、recovery
原因、index file hashes 和 `completion_mode=recovered_completed_partial` 写入 manifest。

## 结果

| Gate | Observed | Threshold | Pass |
| --- | ---: | ---: | --- |
| Full-document top-20 reproduction | 0/7 | 0/7 | yes |
| Passage gold-doc Recall@20 cases | 2/7 | >=4/7 | no |
| Source-document coverage | 100% | 100% | yes |
| 25题 gold-document coverage | 57/57 | 57/57 | yes |
| Passage index document count | 2,715,518 | 2,715,518 | yes |

命中的题为 `754` 和 `869`，passage wins 为 2、losses 为 0。48 条查询的本地累计
搜索延迟为全文 BM25 `2246.10 ms`、passage BM25 `2744.12 ms`。Provider、在线
search、Judge 均为 0。

结论是 `reject`：完整性门槛通过，Recall@20 主门槛失败。不能据此声称系统效果
提升，也不能在这 7 题上继续调整 passage size、overlap 或内部 top-k。

## 框架边界

- [DeerFlow](https://github.com/bytedance/deer-flow) 的动态 Sub-Agent 不能修复共享
  index 的低召回；本轮先测表示层是更小的干预。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  支持可配置 search backend，但 backend 可换不等于已经定位该换哪一层。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 可扩大 crawler 广度，
  但当前必须先验证既有 dense channel 的候选深度，而不是增加搜索角色。
- [MindSearch](https://github.com/InternLM/MindSearch) 的搜索图适合独立分支遗漏；
  当前失败仍是同一 query 下的 candidate visibility。

## 下一步

该 dense gold-rank audit 已完成并被拒绝：相同 48 条查询下，Qwen3-Embedding-0.6B
仅有 top-20 `0/7`、top-100 `1/7`，均低于预注册 `4/7`；top-1000 才达到
`5/7`。因此不进入 reranker 或 fresh paired comparison。完整结果见
`docs/persistent_miss_dense_rank_results_2026-08-16.zh-CN.md`。未运行的 Judge
accuracy、citation support、Token 和费用指标保持 `planned`。
