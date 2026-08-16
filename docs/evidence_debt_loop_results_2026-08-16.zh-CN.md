# Evidence Debt 研究循环结果（2026-08-16）

## 结论边界

本页记录冻结模型下的 harness、检索和证据治理实验。所有数字来自保存的本地
artifact；模型参数始终未训练、未更新。任何差异都不能表述为模型能力提升。

本轮使用 development 数据做 bad-case 定位。outcome-selected 校准、离线 oracle、
persistent Qwen3-32B Judge 都不是官方 BrowseComp-Plus 排名。sealed holdout、官方
提交和 leaderboard 指标均为 `planned`。

## 执行完整性

repeat trial-3 candidate 的 8769 服务已死亡，但旧 runner 把 200/200 个 search
transport failure 后生成的终稿记成 25 个成功题。该 arm 被永久标记为
`reject_execution`，没有送 Judge：

- 伪成功题：25
- 成功搜索：0/200
- 已记录但无效的 provider 费用：$0.0181841464
- artifact：`runs/browsecomp_plus_v0/obligation-span-repeats-v0-20260816/execution-audit.json`
- SHA256：`3f22902d85822109fb767787efbb51f48524ebe2dc111a35880bf78af3a4ec0a`

随后加入 exact retriever identity preflight、search/open fail-closed、batch abort 和
per-run invalid artifact。零 provider/Judge 验证通过 33 个 Python 测试、11 个 Node
contract 测试，并确认 8768/8769 search/open 可用。它只证明执行完整性。

## 已拒绝分支

| 分支 | 结果 | 结论 |
| --- | --- | --- |
| v11 Evidence-Debt Search Reserve | reject | 已知结果校准未过 selectivity、correction、preservation gate，不进 fresh slice |
| v12 Answer-First Selective Repair | reject | answer-first trigger 未达到进入 fresh slice 的门槛 |
| v13 Late-Draft Last-Mile Repair | reject | 保留结果，转向 repair-query 质量诊断 |
| v14 Obligation Span Opening known-5 | calibration pass | 0/5 -> 3/5 Judge，仅允许 fresh 验证 |
| v14 fresh-25 | reject | baseline 11/25 Judge，candidate 12/25；strict exact 7 -> 6，evidence recall 59.17% -> 55.83% |
| v14 两个完整 paired trial | freeze | Judge 22 -> 18，strict exact 14 -> 10，mean evidence recall 59.87% -> 50.95%；Token ratio 0.900696，记录费用 ratio 0.852072 |

仓库没有把一个不存在的 `v15` 运行写成结果。后续 mechanism probe 使用独立 loop
ID，不伪装成端到端 adapter 版本。

检索 query-generation 分支同样保留为负结果：

| Loop | Decision | 直接含义 |
| --- | --- | --- |
| `contrastive-bridge-hypothesis-v0` | reject | 强制对比假设没有达到检索门槛 |
| `draft-blind-counter-hypothesis-v0`（B4 probe） | reject | 隔离 prior answer 后仍未获得足够 gold hit |
| `counter-candidate-frontier-v0`（B5 replay） | reject | 搜完已生成候选仍没有隐藏的 gold-retrieving query |
| `hypothesis-firewall-slate-v0`（B6 probe） | reject | 三查询 firewall slate 不集成 |
| `bridge-generator-model-gate-v0` | reject | 换 Pro 生成 bridge 不能作为修复 |
| `atomic-clue-frontier-v0` | reject | 原子句直接搜索不足 |
| `corpus-grounded-bridge-induction-v0` | reject and freeze | 同一检索-query 层重复失败，按防抖规则冻结 |

B3 固定三 Researcher 没有实现或运行；不得把 B4 probe 写成完整多 Agent 系统。

## 三次 Baseline 诊断

在同一 25 题上审计三个有效 v10 baseline run：

| Failure category | Cases |
| --- | ---: |
| stable correct | 9 |
| persistent retrieval miss | 7 |
| gold doc present, answer span missing | 4 |
| unstable answer | 3 |
| answer span present, persistent wrong | 1 |
| persistent wrong, other | 1 |

这里不再依赖旧 `evidence_recall` 的答案原子重叠，因为常见词可能造成假阳性。
`answer span present` 要求 literal gold answer 出现在 retrieved gold-document preview。

## Evidence Visibility

4 个 gold-doc-present/span-missing 题为 `772, 579, 552, 679`。

| Mechanism | Head hit | Selected hit | Decision |
| --- | ---: | ---: | --- |
| same-size head preview | 0/4 | - | baseline |
| `answer_obligation_window_v0` | - | 3/4 | availability pass, q679 miss |
| answer-obligation compiler v1 | - | 3/4 | reject，正确识别 q679 obligation 但仍选错窗口 |
| section-aware selector v2 | - | 4/4 | calibration-only pass |

v2 只在 obligation 明确点名 section 时优先包含其规范化别名的窗口。例如 q679
从作者简介改选 `ACKNOWLEDGMENT` 附近，并露出 `ASCENT Project 10`。这 4 题已被
观察且使用 gold 文档，因此 4/4 不是泛化效果数字。

`obligation_channel_slate_v0` 不把 gold 输入 selector。它选择与 answer obligation
最匹配的两个已记录 search call，再分别取一个 BM25 anchor 和 dense lead：

- gold target coverage：4/4
- 总目标文档：15
- 每题最多目标：4
- gold 在 slate 中的位置：1、1、4、3
- 新搜索、provider、Judge、document open：均为 0

## Monotonic Overlay

overlay 不改变 baseline exploration，只打开上述 slate 的 v2 span。replacement 必须
满足：short answer 和 evidence quote 均逐字存在于每个 cited span，否则保留原答案。

| Variant | Supported replacement | Strict exact | Judge correct | Provider Token (in/out) | Cost | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v0 baseline-visible | 2/4 | 2/4 | 2/4 | 8293 / 143 | $0.00120106 | reject |
| v1 evidence-only | 0/4 | 0/4 | 0/4 | 7085 / 92 | $0.00101766 | reject and freeze prompt branch |

v0 正确修复 q579 `March 5, 2021` 和 q552 `Sarah Abbassalizadeh`；q772、q679
选择 keep。v1 隐藏 baseline 后 4 题全部 keep。两轮都是 0 parse failure、0 unsupported
replacement、0 new search。不能为了追阳性继续改 verifier wording。

## Retrieval Miss

7 个 persistent retrieval miss 为 `875, 754, 898, 710, 873, 869, 805`。

语料可回答性审计：

- 7/7 gold-document 集合包含 normalized literal answer
- 18/18 gold 文档存在并可打开
- missing gold document：0
- provider/search/Judge call：0

同一 BM25 full-document index 上，以 depth 1000 比较 raw full question 和 trial-1 的
全部 48 条模型生成查询，共执行 55 个离线 BM25 query：

| Source | top-5 | top-20 | top-100 | top-1000 |
| --- | ---: | ---: | ---: | ---: |
| raw full question | 0/7 | 0/7 | 0/7 | 1/7 |
| best generated query | 0/7 | 0/7 | 2/7 | 6/7 |

六个可见 gold 的 best generated-query rank 为 `47, 96, 102, 119, 413, 736`；另
一题未进入 top-1000。raw-question anchor 和小幅提高 top-5 pool 均被拒绝。当前
因果层定位为 full-document/index representation，而不是 corpus absence。

## 框架对照

- [DeerFlow](https://github.com/bytedance/deer-flow) 当前由 Lead Agent 动态派生隔离
  Sub-Agent。我们的 7 个主失败在共享检索层，增加 worker 只会重复低 rank。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  把较低性能的 supervisor-researcher multi-agent 放在 legacy；角色数不是质量保证。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 使用 planner、
  execution/crawler agents 和 publisher，适合扩大搜索；当前应先修索引表示。
- [MindSearch](https://github.com/InternLM/MindSearch) 的 WebPlanner/WebSearcher 图适合
  独立分支遗漏；当前 gold rank 结果尚未证明需要这个层级。

## 下一阶段

1. `completed/reject`：完整语料 passage-level BM25 将 100,195 个源文档切为
   2,715,518 个段落；相同查询的 collapsed gold-doc Recall@20 仅为 2/7，低于
   预注册 4/7。源文档和 25 题的 57 个 gold 文档均 100% 覆盖，因此 passage
   branch 按规则冻结，不再调 chunk size、overlap 或 top-k。
2. `planned`：固定相同 48 条生成查询，审计 dense gold-document rank，区分 lexical
   representation failure 与 semantic candidate-depth failure；provider、在线搜索和
   Judge call 均保持 0。
3. `planned`：只有新的检索候选通过离线门槛后才注册 fresh paired development test。锁定 DeepSeek 模型、
   prompt、语料、8 个 search call、总 Token，并分别报告 Token-matched 和
   cost-matched 结果。
4. `planned` metrics：Judge accuracy、strict exact、gold-doc recall、citation support、
   latency、Token、美元及人民币费用。未运行前不填数字。
5. 多 Agent 继续 `planned`。只有 passage retrieval 稳定后反复出现独立分支遗漏、
   上下文互扰、矛盾证据未核验或串行延迟，才比较 B2/B3/B4。
