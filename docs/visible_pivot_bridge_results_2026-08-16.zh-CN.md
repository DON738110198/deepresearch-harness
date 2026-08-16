# Visible-Pivot Bridge v0 结果

## 问题

全文 BM25、passage BM25 和 dense candidate-depth gate 都没有把相同 7 个
persistent miss 稳定带入 top-20。重新增加 query Agent、提高 candidate depth 或
加入 reranker 都违反前序 stopping rule。本轮只问：模型保存下来的非 gold snippet
中，是否已经出现过一个能通向 gold 文档、但没有进入后续 query 的词汇 pivot。

## 预注册

注册提交 `462f0a1` 先于实现与结果进入 `main`。冻结规则为：

- 相同 7 题、trial-1 的 48 条成功 query、755 条保存 snippet；
- 相同 100,195 文档全文 BM25，`k1=0.9`、`b=0.4`、top-k 20；
- pivot 必须同时出现在一个保存的非 gold snippet 与一个 gold 文档中；
- 排除 raw question、48 条 query 和 gold answer 的 analyzed vocabulary；
- 每题最多 64 个 pivot，按 corpus document frequency 升序后字典序固定；
- 每个 pivot 依次追加到冻结 query，命中后该题停止；
- acceptance 为至少 `4/7`，总离线 BM25 query 不超过 3,120；
- provider、在线搜索、Judge、GPU 均为 0；sealed holdout 禁止。

这是 gold-aware existence oracle，不是 gold-blind selector 或系统效果实验。

## 结果

| Gate | Observed | Threshold | Pass |
| --- | ---: | ---: | --- |
| Frozen BM25 baseline reproduction | 0/7 | 0/7 | yes |
| Visible-pivot gold-doc Recall@20 cases | 4/7 | >=4/7 | yes |
| Offline BM25 query budget | 1,088 | <=3,120 | yes |

所有 7 题都存在至少一个符合词表交集条件的 candidate。四个 rescue 为：

| Query ID | Analyzed pivot | Gold rank | Visible support | Gold support |
| --- | --- | ---: | --- | --- |
| 875 | `inlin` | 4 | 38321 | 49342, 60750 |
| 754 | `argentin` | 9 | 48829 | 64533 |
| 869 | `naep` | 6 | 60021 | 92067 |
| 805 | `obra` | 19 | 27532 | 51820, 54316 |

失败题为 `898, 710, 873`。基线执行 48 个 query，pivot 执行 1,040 个 query。
Provider、在线 search、Judge call 均为 0。结果 artifact：
`runs/browsecomp_plus_v0/visible-pivot-bridge-v0-20260816/audit.json`，SHA256
`bbf01d457b45001fd15409349bd0c59a4767446274cd317dd19b85f2b9279a05`。

## 关键反例

注册 gate 按规则 `accept`，但不能把 4/7 直接写成“语义 bridge 成功”。post-hoc
上下文检查发现 q875 的 `inlin` 是 analyzer 对 `inline` 的 stem；它在可见文档与
gold 文档中都来自 YAML 坐标字段，例如 `coordinates: ... inline`。这是模板格式
共现，不是题目语义。

其余三项至少有可解释的正文或标题语境：`argentin` 指向 Argentina，`naep` 指向
教育评估主题，`obra` 出现在作者作品语境。但它们仍由 gold overlap 选出，不能
据此声称一个无 gold 的系统会选中它们。

## 与成熟框架的差异

- [DeerFlow](https://github.com/bytedance/deer-flow) 可把不同研究范围派给隔离
  Sub-Agent；本轮先证明单个已见 snippet 是否含可复用边，再决定是否值得派生。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  的 search backend/compression 边界值得保留；这里新增的是可审计 pivot provenance，
  不是角色复制。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 通过 planner/crawler
  扩大覆盖；本项目下一步要求先在固定两次以内的 pivot search 中证明选择质量。
- [MindSearch](https://github.com/InternLM/MindSearch) 用搜索图扩展节点；Visible-Pivot
  把最小的一条边单独拿出来做可反驳 gate，不直接实现完整图或多 Agent。

## 下一步

`planned`：gold-blind Visible Pivot Slate。它必须删除 wrapper 和 frontmatter，只从
可见正文提取 capitalized entity/acronym 候选；每个候选记录 snippet docid、source
query 和正文位置；不读取 gold 文档或 answer；冻结 slate 大小、实际 pivot query
数、总 search calls 和 Token/费用。只有该 selector 先通过离线 gate，才允许注册
新的 fixed-budget development run。未运行的 selector recall、Judge accuracy、
citation support、Token、费用和多 Agent 指标全部保持 `planned`。
