# Gold-Blind Visible Pivot Slate v0 结果

## 问题

Visible-Pivot oracle 用 gold overlap 找词，`4/7` 只证明词汇桥存在，不证明运行时
能选中。q875 的成功词甚至来自 frontmatter 格式元数据。本轮将“选 pivot”和“用
gold 评分”拆成两个 artifact 阶段，测试一个最简单、真实受预算约束的无 gold
selector。

## 预注册

注册提交 `616e264` 先于实现和结果进入 `main`。固定规则为：

- selector 只读取保存的 `request.json` question、成功 query 和 result snippet；
- selector 不读取 gold 文档或 answer；
- 删除 leading wrapper 与 YAML frontmatter 后再提取正文候选；
- 候选必须是 ASCII capitalized token，且默认 analyzer 后只有一个 term；
- 排除 question 和既有 query term，corpus document frequency 不超过 10,000；
- 先按 document frequency 升序，再按跨可见文档支持数和首次出现排序；
- 每题固定两个 pivot，各自追加到首次暴露它的 source query；
- 最多 14 个离线 BM25 top-20 query；
- acceptance：总体至少 `2/7`，并保留至少 `2/4` oracle rescue；
- selection failure 与 frontmatter-only selected candidate 必须为 0；
- provider、在线搜索、Judge、GPU 为 0，sealed holdout 禁止。

## 执行完整性

第一次 build 在 slate 落盘前 fail-closed：Pyserini 对已 analyzed term 又套默认
analyzer，某个 token 被过滤为空后 Java term-count API 抛出越界。没有 slate、没有
score、没有外部调用。提交 `fc76258` 改为 exact analyzed-term document frequency，
不改变注册选择规则，真实 index integration check 通过后恢复。

恢复运行先写出：

- `slate.json` SHA256：
  `38afc27265a8a06a3f83f803821ca9acb2eefee637ea44ea0aa15aaf06d1f9d5`
- `audit.json` SHA256：
  `80950baaab11784a138b76d5b44e8df263ba2b92e131e47062edccf4478cad88`

测试还显式删除 gold 与 oracle 文件后单独运行 builder，证明 selection 阶段不打开
这两个 artifact；scorer 只在 slate 已持久化后读取 gold。

## 结果

| Gate | Observed | Threshold | Pass |
| --- | ---: | ---: | --- |
| Selector gold-doc Recall@20 cases | 0/7 | >=2/7 | no |
| Retained oracle rescues | 0/4 | >=2/4 | no |
| Selection failures | 0 | 0 | yes |
| Frontmatter-only selected candidates | 0 | 0 | yes |

Answer-string leak、gold-docid leak 均为 0。7 题各选两个候选并执行一次 provenance
query，总离线 BM25 query 为 14；provider、在线搜索、Judge call 均为 0。

| Query ID | Selected surfaces |
| --- | --- |
| 875 | `Qiblih`, `Cailor` |
| 754 | `Bourmont`, `Fathia` |
| 898 | `Foodlabs`, `Obremskey` |
| 710 | `Sipido`, `Zoobiquity` |
| 873 | `NaSHOF`, `ORCHESTA` |
| 869 | `CityNote`, `Cailor` |
| 805 | `Poopoksakul`, `Chindea` |

14 个候选的 corpus document frequency 都是 2。selector 没有“找不到实体”，而是
在每题 320 至 1,190 个正文 capitalized candidates 中，被 rarity-first 排序吸到
无关长尾专名。这把失败定位到 **candidate representation/ranking**，而不是 gold
leakage、frontmatter 或检索执行错误。

## 框架对照

- [DeerFlow](https://github.com/bytedance/deer-flow) 可让 Sub-Agent 独立追踪候选，
  但若候选表示先把随机 df=2 专名排在最前，多 worker 只会扩大噪声。
- [LangChain Open Deep Research](https://github.com/langchain-ai/open_deep_research)
  的 search/compression 分层提示我们把 pivot provenance 留在结构化 slate 中；本轮
  已做到可审计，但选择质量没有过线。
- [GPT Researcher](https://github.com/assafelovic/gpt-researcher) 可增加 research
  questions 与 crawler 广度；当前失败要求先提升 entity/relation 表示，而非加调用。
- [MindSearch](https://github.com/InternLM/MindSearch) 的图扩展需要可靠节点和边；
  `0/7` 说明当前 rare-token 节点不值得扩成多 Agent 搜索图。

## 决策

机器决策为 `freeze_gold_blind_pivot_branch`。不得在同一 7 题上扩大 slate、交换
排序键或改 top-k 追阳性。下一允许的问题是不同的 candidate representation，例如
typed entity/relation linking；它必须在新注册下保持 retriever、Agent 数和查询预算
不变，并首先优于本轮固定 `0/7`。fresh paired run、Judge accuracy、citation
support、Token、费用、sealed holdout 和多 Agent 全部保持 `planned`。
