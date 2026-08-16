# 受控 Deep Research 研究循环

## 目的

这个项目不以“功能越来越多”为进度，而以一条可反驳的实验链为进度：

```text
定位一个主失败簇
-> 检查更简单的 baseline
-> 对照现有 Deep Research 框架
-> 只冻结一个机制变化
-> 离线校准
-> 新开发集实跑
-> Judge + 资源门槛
-> 接受或保留负结果
-> 再进入下一轮
```

每次准备暂停、换方向或宣布阶段完成前，都必须生成并验证一个
`ResearchLoopCheckpoint`。只要 loop 没有 measured result、机器 decision，
或者仍有付费调用在途，检查器都会返回 `ready_to_pause=false` 和明确的下一动作。

## 每轮边界

1. 一轮只允许改变一个机制；prompt、检索器、模型或预算的联动修改必须拆开。
2. 必须保留最简单 baseline，并写清为什么当前机制比“多搜几次”或“多加一个
   Agent”更适合已观察失败。
3. 至少对照三个当前开源框架，而且只能引用官方仓库或官方文档。对照内容必须
   是机制和失败边界，不比较 stars，也不把功能清单当创新。
4. 外部机制只有映射到保存的 bad case 才能进入实现。仅达到 DeerFlow、Open
   Deep Research、GPT Researcher 或 MindSearch 的功能平价不计作贡献。
5. 每轮在注册文件中固定模型、工具、语料、问题切片、搜索数、Token、费用上限、
   Judge 和停止规则。`sealed holdout` 始终禁止，直到三次开发集稳定确认通过。
6. outcome-selected bad case 只能用于机制校准，不能用于效果数字；效果判断必须
   使用此前未评测的新 development slice。
7. 失败实验不删除、不调阈值、不挑子集。下一轮只能从新的失败定位开始。

## 防抖与执行完整性

1. 同一机制层连续三次注册实验被拒绝后，该层默认冻结；更严格的注册可以提前
   冻结某个窄分支。本轮 terminal-verifier prompt 分支在 v0、v1 两次拒绝后已按
   checkpoint 约定提前冻结。
2. 冻结不是“以后永远不能碰”，而是必须先出现新的 failure cluster 或新的离线
   证据，不能继续在原结果上换措辞、调阈值或扩大样本来追阳性。
3. 搜索服务必须在每次 provider 子进程前通过 health 与精确 `retriever_id`
   检查。任一 search/open transport error 都使该题失败并中止批次；有流畅终稿
   也不能把工具失败改写成成功。
4. 失败进程和部分结果必须保存。恢复只读取已落盘的 per-case artifact；付费结果
   不得因进程中断而重复调用。
5. 离线 oracle、smoke 与静态测试只证明可用性或执行完整性，不得写成效果提升。

## 外部对照的作用

- DeerFlow 2.0 的动态 Sub-Agent、隔离上下文和终止条件适合真正的并行分支问题；
  当前错误不是这个类型，因此暂缓多 Agent。
- LangChain Open Deep Research 已把早期 supervisor-researcher 多 Agent 版本列为
  低于当前实现的 legacy。它提醒我们角色数量不是质量保证，但其压缩阶段和
  Deep Research Bench 导出边界值得后续对照。
- GPT Researcher 的 Planner、execution/crawler agents 和 publisher 强在搜索
  广度与报告生产；本项目要验证的是更小的答案级证据债务控制能否用更少资源
  达到更可靠的停止。
- MindSearch 用 WebPlanner 动态扩展搜索图并派发 WebSearcher，适合独立子问题
  漏检。只有我们的 bad case 反复出现独立分支遗漏时，才进入该层对比。

## 当前结论与方向

项目不会声称在通用产品能力上超过这些成熟框架。已经跑出的结论是：

- 全局暴露 span-open 工具会扰动搜索轨迹，两个完整 paired trial 合计少 4 个
  Judge-correct、少 4 个 strict exact，mean evidence recall 下降 8.92pp，因此冻结。
- 三次 v10 baseline 把 25 题分成 9 个 stable-correct、7 个 persistent retrieval
  miss、4 个 gold-doc-present/span-missing，以及 5 个其他合成或不稳定案例。
- 对 4 个 span-missing 案例，section-aware selector 的 gold-aware 离线校准达到
  4/4；no-gold target slate 也能在最多 4 个候选中达到 4/4。这只是可用性门槛。
- bounded post-run overlay v0 仅修复 2/4；隐藏 baseline 的 v1 为 0/4。两轮均被
  注册门槛拒绝，terminal-verifier prompt 分支冻结。
- 7 个 persistent retrieval miss 的 18 个 gold 文档全部在索引中，且 7/7 全文
  含 literal answer。raw full question 的 BM25 top-5 为 0/7；模型生成查询最好也
  只有 0/7 top-20、2/7 top-100、6/7 top-1000。
- 完整 passage index 将 100,195 个源文档确定性切成 2,715,518 个段落，源文档与
  25 题的 57 个 gold 文档覆盖均为 100%；但相同 48 条生成查询只把 2/7 拉入
  collapsed document top-20，未达到预注册 4/7，passage 分支冻结。
- 相同 48 条查询的 Qwen3-Embedding-0.6B dense rank 为 top-20 `0/7`、top-100
  `1/7`、top-1000 `5/7`；top-20 和 top-100 两个预注册 `4/7` gate 均失败，
  dense candidate-depth 与 bounded reranker 分支冻结。
- Visible-Pivot lexical oracle 在保存的非 gold snippet 与 gold 文档共有词中排除
  question/query/answer vocabulary，再追加一个 token；它以 `4/7` 刚好过线，
  但 q875 的 `inlin` 来自坐标 frontmatter 的 `inline` 格式词。该结果只证明一跳
  lexical sufficiency，不能当作 semantic selector 效果。
- gold-blind body-only selector 先独立落盘两个 pivot/case，再打开 gold 评分；它无
  answer、gold docid 或 frontmatter leakage，但 14 次 provenance-bound search 为
  `0/7`，oracle retention `0/4`。被选 pivot 全是 df=2 长尾词，说明纯 rarity 不能
  把可见实体转成有效研究边。
- uniform RRF 在 gold 访问前持久化 48 条 frozen query 的完整 BM25 top-1000，
  fused top-20 与 top-100 均为 `0/7`；best-single-query baseline 分别为 `0/7`、
  `2/7`。跨 query 的通用噪声共识压过 gold，fusion 分支冻结。

因此当前差异点不再表述为泛化的 `repair_search`，而是 **Evidence Debt 驱动的
检索表示诊断**：先证明 obligation、目标文档、答案 span 分别在哪一层丢失，再
决定是否改变 index、candidate pool 或验证器。passage 与 dense gate 均已以负结果
关闭；Visible-Pivot Bridge Sufficiency oracle 已过最小存在性门槛。下一候选不是
再扩 oracle。gold-blind 小 pivot slate 与 uniform RRF 均已被拒绝并冻结；不能在
同一结果上改 rarity order、RRF `k`、tie-break 或扩大 slate 追阳性。typed
entity/relation linking 仍是候选，但源码核对纠正了此前的输入假设：官方复现命令是
query `512`、document `4096`，预构建向量仓库本身却没有绑定历史 preprocessing
metadata。因此在投入 typed linking 或 passage-dense build 前，先验证官方 `4096-token`
document recipe 是否实际看得到 answer span，并把 provenance 缺口单独记录。oracle
本身仍不是可部署 selector。该预注册诊断现已完成：case visibility 为 `7/7`，
document visibility 为 `17/18`，所以 `4096-token` head truncation 不是主要解释，
passage-dense 不获准。随后 raw full question 的 dense rank 为 top-20 `1/7`、
top-100 `1/7`、top-1000 `2/7`，对比 frozen generated-query 的 `0/7`、`1/7`、
`5/7`；逐题 rank 也只赢 `1/7`，三个预注册 `4/7` gate 全部失败。raw-question
general anchor 因此冻结，不能在同一 7 题调 prefix、depth 或 query 混合。下一轮先
审计仓库中已有 bridge/hypothesis probe。审计发现另一组 3 个已知失败上已经连续拒绝
obligation rewrite、typed contrastive bridge、draft-blind counter-hypothesis、
hypothesis firewall、Pro generator substitution 和 corpus-grounded induction。
因此不再把 prompt-only typed bridge 换名后重跑；当前 7 题进入 regression-only，
下一轮改为冻结 v10 在完整 175 题 development 上做一次规模化错误分布采集，再由新
bad-case 主类选择机制。
多 Agent、fresh paired run、sealed holdout、official Judge 和 leaderboard submission
都保持 `planned`。

## 操作命令

```powershell
python scripts/check_research_loop_checkpoint.py `
  experiments/evidence_debt_search_reserve_v0/checkpoint.json
```

日常检查允许 active loop 返回非暂停就绪；真正结束阶段时加
`--require-pause-ready`，未闭环会以非零状态退出。

同一 failure cluster 达到拒绝上限后，另一个机器检查负责阻止继续挑机制：

```powershell
python scripts/check_failure_cluster_route.py `
  benchmarks/browsecomp_plus_v0/persistent_miss_cluster_route_v0.json
```

当前 route 绑定 5 个同 cluster 负实验和 6 个历史 bridge analogue，输出
`selection_allowed=false`。此后这 7 题只能做 regression replay；下一动作必须是
`broader_development_profile`，不能再以新的名称调同一批结果。
