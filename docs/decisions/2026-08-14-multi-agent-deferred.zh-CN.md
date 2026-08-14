# ADR：暂缓多 Agent，先完成检索层确认

- 日期：2026-08-14
- 状态：已采纳
- 当前实施：单 Agent B2；B3/B4 均为 `planned`，未实现、未运行

## 问题

“Planner + Researcher + Writer”式角色拆分已经广泛存在于 DeerFlow、MindSearch、GPT Researcher、Spring AI Alibaba DeepResearch、Skywork DeepResearchAgent 等项目中。LangChain Open Deep Research 也保留过 supervisor-researcher 方案，但当前实现并不以旧多 Agent 方案为主线。因此，增加角色数量本身既不是本项目的创新，也不自动保证效果。

本项目当前的实跑证据指向更靠前的失败层。冻结的 25 题 baseline profile 中，校准 Judge 判错的 17 题全部同时满足 `evidence_recall = 0` 和 `gold_recall = 0`；判对的 8 题都至少召回了一份相关文档。该诊断及其输入哈希已冻结在 `benchmarks/browsecomp_plus_v0/dense_confirmation_v1.json`。这说明当前最小因果问题是 first-pass retrieval miss，而不是角色不足、写作批评不足或缺少 Research DAG。

多个 Agent 若共享同一个弱检索器，会重复零召回，增加 Token、费用和协调开销，却没有直接修复当前坏例。

## 决策

1. 不立即实现 B3/B4，也不把多 Agent 写成项目创新点。
2. 保持已预注册顺序：先完成冻结 fresh-25、三次重复的 BM25 vs Dense Retrieval gate，再用已校准且版本锁定的 Judge 评分，最后生成 hash-bound decision artifact。
3. 不因中间 trial 结果修改候选检索器、阈值、问题集合或停止规则；sealed holdout 继续禁止访问。
4. 若 Dense gate 未通过，保留负结果并继续定位检索层原因，不用多 Agent 掩盖 retrieval miss。

## 后续候选

以下设计只是预留命名，不构成已经实现或已有收益：

| 版本 | 定义 | 当前状态 |
| --- | --- | --- |
| B2 | 当前单 Agent Evidence Debt | 已实现；效果仍需按注册门槛判断 |
| B3 | 固定派生 3 个并行 Researcher | `planned` |
| B4 | Evidence-Debt-Guided 动态派生 Sub-Agent | `planned` |

B4 只有在坏例证明必要后才允许立项。候选边界为：Lead 根据未解决 obligation 分簇；Sub-Agent 只返回结构化 `EvidencePacket`；共享 Claim-Evidence Ledger 是唯一事实源；只有冲突或低置信证据触发反证 Agent；Writer 只能使用已验证 Ledger；调度与停止按预计 `Evidence-Debt reduction / Token` 决定。

## 启动条件

检索层稳定后，只有 bad-case 队列反复出现下列至少一种单 Agent 边界，才注册 B3/B4 实验：

- 独立研究分支持续遗漏；
- 单一上下文发生可复现的相互干扰；
- 矛盾证据已经召回但未被核验；
- 串行研究的墙钟延迟成为主要失败项。

“召回为空”不属于这些条件，仍应先修检索。

## 公平比较合同

B2/B3/B4 比较必须锁定生成模型、thinking 配置、搜索 provider、语料快照、问题集合、总 query calls、总 Token 和总人民币费用；报告 evidence recall、校准 Judge accuracy、citation support、冲突发现率、重复搜索率、synthesis loss、墙钟时间和费用。Token-matched 与 cost-matched 必须分别报告，所有未跑指标标为 `planned`。

任何未来差异只能表述为 harness/orchestration effect，不能表述为冻结模型能力提升。

## 检索确认后的更新

fresh-25 Dense top-5 确认没有通过 `+10 pp` evidence recall 门槛。随后
Evidence Bandwidth top-20 在另一组 fresh-25 三轮实验中把 recall 提高
`10.193333 pp`，但校准 Judge accuracy 下降 `8 pp`，搜索、Token 和费用
比例也超过预注册上限。错误结构从 BM25 的 24 个 no-evidence / 22 个
relevant-but-wrong，变为候选的 19 / 35。

这使下一问题从“是否需要更多搜索角色”进一步收窄为“如何让更宽的候选池
只把经过选择的证据送入上下文，并把召回增益转成正确答案”。单上下文干扰
目前只是候选解释，还没有达到 B3/B4 的启动证据。先测试 progressive
disclosure、Evidence Packet 和受限 `open_evidence`，仍然比增加 Agent 数量
更简单、因果问题更清楚。

## 2026-08-15 更新

Progressive Disclosure 的第一版因 65 次搜索和 `3.131361x` Token 被拒绝；
加入 8 次搜索上限后，资源问题消失，但 fresh-5 的 Judge 仍为 0/5。保存
trace 随后发现 Dense lead 只有 frontmatter 开头，四个相关 lead 均没有可供
Agent 选择的标题值或段落。query-aware preview 的零 provider-call 校准把
可选择相关 lead 从 0/4 提高到 4/4，fresh-5 才获准进入 25 题确认。

fresh-25 的机器总决策因 baseline schema completeness `92% < 96%` 保持
`reject`，不能事后删除该 gate。但其余质量与资源 gate 均通过：校准 Judge
`16% -> 20%`，evidence recall `36.03% -> 44.79%`，搜索/Token/费用比为
`0.488764 / 0.251863 / 0.483486`。候选错误中 zero-recall 从 baseline 的
11 个降至 7 个，而 evidence-present-but-wrong 从 10 个增至 13 个。

因此当前最小问题是“未解决的答案义务没有在最终编译前触发候选核验”，而
不是 Researcher 数量不足。下一候选是单 Agent 的 Evidence-Debt Search
Reserve：在同一 8 次搜索和 10k 输出预算内，为最多两条 debt-guided repair
query 预留资源。B3/B4 继续保持 `planned`，不得抢跑。
