# Multi-Query RRF v0 结果

## 问题

7 个 persistent retrieval miss 的 frozen generated queries 在单条查询 top-20 上为
`0/7`，但在 top-1000 内能找到 `6/7`。passage、dense 和 visible-pivot 分支均未
达到各自门槛。在引入 LLM entity linker、reranker 或额外 Agent 前，本轮先补最简单
的遗漏 baseline：把同一题已有查询的排序结果做确定性融合。

## 预注册

注册提交 `72217c3` 先于实现提交 `67db0c9` 和结果。固定项为：

- 同一 7 个 outcome-selected persistent miss；
- trial-1 保存的 48 条成功查询，重复 query fail-closed；
- 同一 100,195 文档全文 BM25 index，`k1=0.9`、`b=0.4`；
- 每条查询只取 top-1000；
- uniform Reciprocal Rank Fusion，`k=60`；
- score 为各查询 `1 / (60 + one_based_rank)` 之和；
- tie 依次按 best individual rank、出现次数降序、docid 升序；
- 只评估 fused top-20 与 top-100；两个 gate 均要求至少 `4/7`；
- provider、在线搜索、Judge、GPU 和 sealed holdout 均为 0/禁止。

builder 只打开注册、保存的 run trace 和 index，不解析 lexical result、pivot result
或 gold。完整 48-query ranking 与 fused top-100 先原子写入 `slate.json`，scorer
之后才打开 gold。这是校准边界，不是 unseen 或端到端效果。

## 执行完整性

本机只剩约 0.66 GiB 可用内存，因此没有冒险加载索引。运行迁移到已有的高内存
CPU 环境，显式设置空 `CUDA_VISIBLE_DEVICES`。

第一次远端启动在检索前失败：系统 Java 11 缺少 Pyserini 需要的
`jdk.incubator.vector`。没有生成 slate/audit，也没有执行 BM25 query。恢复只改变
运行时：使用官方 Eclipse Temurin `21.0.11+10`，下载 archive SHA256 为
`4b2220e232a97997b436ca6ab15cbf70171ecff52958a46159dfa5a8c44ca4de`，校验通过。

第二次 Python 进程完成两个 artifact；外层日志管道因 PowerShell 提前展开远端
`PATH` 而找不到 `tee`，所以 wrapper 退出 1。该错误发生在 artifact 完成边界外，
没有重跑 48 次检索。恢复记录为：

- `execution.json` SHA256：
  `71e22ae28ae0726b7f24808b4f011298d7e5d8a3c357ef7a303b9627cff1ae74`
- `slate.json` SHA256：
  `46f0653c8745cfb11508d43233e37e6b56e2fc7212fca2baa392008081c5afc8`
- `audit.json` SHA256：
  `205892ab67eb815657e3e3b369ab0c9201cff179cda7e812159afcbd6c952ff7`

本地与远端 SHA256 一致，两个 artifact 均通过 Pydantic contract。实际执行 48 次
离线 BM25，记录 search latency 63.750 秒；provider、在线搜索、Judge 和 GPU 为 0。

## 结果

| Candidate view | Gold-hit cases | Threshold | Pass |
| --- | ---: | ---: | --- |
| Best single query top-20 | 0/7 | baseline | n/a |
| Fused top-20 | 0/7 | >=4/7 | no |
| Best single query top-100 | 2/7 | baseline | n/a |
| Fused top-100 | 0/7 | >=4/7 | no |

RRF 不仅没有把 gold 推入 top-20，在 top-100 还比 single-query oracle 少 2 个 case。
注册 decision 为 `freeze_multi_query_rrf`。

仅用于失败解释的 post-hoc full-fusion 检查显示，6 个可见 gold 的 fused rank 为
`215, 459, 1257, 1986, 405, 180`，另 1 题仍缺失。对应 gold 文档跨查询出现次数为
`1, 2, 3, 2, 4, 5`。也就是说，gold 并非完全没有跨查询重复；但常见噪声文档重复
得更多，uniform consensus 反而进一步压低了 gold。

## 框架对照

- [DeerFlow](https://github.com/bytedance/deer-flow) 用 Lead Agent 汇总隔离 Sub-Agent
  结果；本轮说明在候选本身不可见时，先增加并行上下文不是更简单的修复。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  的迭代搜索与 compression 能做更强聚合，但 RRF 是不增加模型调用的必要下界。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 会执行更多 research
  questions/crawlers；当前已有多查询共识仍为 0/7，不能把搜索广度直接当贡献。
- [MindSearch](https://github.com/InternLM/MindSearch) 聚合 WebPlanner/WebSearcher 图；
  本轮是它的无 Agent 排名聚合下界，失败后仍需先修候选表示。

## 决策

冻结 uniform RRF，不在相同 7 题调 `k`、tie-break 或融合权重，也不允许对一个
gold coverage 为 `0/7` 的 fused top-100 追加 reranker。typed entity/relation 仍是
候选方向，但在投入该机制或构建大规模 passage-dense index 前，先做更便宜的
**dense-head answer visibility** 诊断：检查 pinned 512-token 文档输入是否实际包含
18 个 gold 文档中的 literal answer span。该诊断只定位 full-document dense 的
截断边界，不能产生准确率或模型能力结论。fresh paired run、Judge、official 和
leaderboard 指标继续标为 `planned`。
