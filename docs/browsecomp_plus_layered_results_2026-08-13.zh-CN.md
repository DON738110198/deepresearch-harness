# BrowseComp-Plus 分层创新实验记录（2026-08-13）

## 结论边界

当前结果来自 5 道 development 题，只用于机制诊断，不是官方准确率，也不能用于声称排行榜名次。模型参数始终冻结；变化来自检索器、推理阶段调度、提示和预算控制，不能表述为模型能力提升。

官方 Qwen3-32B Judge 尚未运行，状态为 `planned_not_run`。本页的 normalized exact match 是刻意严格的字符串指标，会把带单位、解释或等价改写的答案判错。

## 先定位问题

严格预算运行暴露了四个不同层次的问题：

1. **协议层**：Pi 0.84.1 为 DeepSeek 自动选择了 `max_completion_tokens`，但 DeepSeek Chat Completions 文档使用 `max_tokens`。前者被 API 接受却没有约束观察到的推理输出，因此早期运行不能作为 Token-matched 证据。
2. **成文层**：标准 high-thinking loop 经常把 10,000 输出 Token 全部花在探索上，最后没有 `Exact Answer` 和 `Confidence`。
3. **检索层**：BM25 即使进行了多轮搜索，也会在长线索改写题上得到零相关文档。
4. **控制层**：个别题在第一次工具调用前就耗尽探索预算；单纯提醒模型搜索并不能可靠改变行为。

## 已运行实验

所有价格、Token 和延迟均来自保存的 provider trace。每题最多 10,000 输出 Token，overshoot 均为 0；下表中的 exact 和 recall 不是官方 Judge 分数。

| 变体 | 检索 | 题数 | 格式完整 | strict exact | evidence recall | gold recall | 搜索次数 | 输出 Token | 费用（USD） | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| standard high-thinking | BM25 | 5 | 0/5 | 未评分 | 未评分 | 未评分 | 80 | 50,000 | 0.04845 | 严格基线，全部耗尽预算 |
| 8k explore + 2k high compile | BM25 | 5 | 2/5 | 未评分 | 未评分 | 未评分 | 66 | 49,583 | 0.06668 | 未通过 4/5 格式门槛 |
| 6k explore + 4k high compile | BM25 | 5 | 0/5 | 未评分 | 未评分 | 未评分 | 48 | 49,998 | 0.05323 | “多给成文 Token”假设被否定 |
| 8k high explore + 2k non-thinking compile | BM25 | 5 | 4/5 | 1/5 | 40.00% | 40.00% | 70 | 43,120 | 0.06867 | 通过格式门槛，形成当前控制基线 |
| 同一控制策略 | Qwen3-Embedding-0.6B | 5 | 5/5 | 1/5 | 54.92% | 60.00% | 31 | 28,854 | 0.02690 | 修正 snippet 合同后的可比运行 |

稠密检索输出中，另外两题与参考答案语义一致，但分别多了括号说明或单位，因此当前运行按语义推断约为 3/5；在 Qwen3-32B 实际给出判定前，这不能写成官方 60% 准确率。

更早一次 dense 运行得到 2/5 strict exact、77.14% evidence recall，并可能有 4/5 语义正确，但当时短 snippet 会被 tokenizer 重新 decode，而 BM25 短 snippet 保留原文。这个差异很小，却破坏了“完全相同 snippet 合同”，所以该运行保留为诊断、从主表排除。修正后重跑的差异同时提醒我们：DeepSeek thinking mode 不支持 temperature/top-p 控制，接口也没有 seed，单次运行不能代表可靠性。

## Counterfactual Replay

为避免先付费重跑 Agent，再猜测提升来自哪里，我们固定了 BM25 运行已经生成的 70 条搜索 query，只替换检索器：

- BM25 evidence recall：40.00%；
- Qwen3-Embedding-0.6B evidence recall：55.95%；
- 变化：+15.95 percentage points；
- BM25+dense RRF evidence recall：54.37%。

这说明稠密检索本身值得进入端到端实验；RRF 在这个 5 题切片上没有胜过纯 dense，因此暂不增加融合复杂度。replay 不调用 DeepSeek，也不改变 query 数量。

## 控制层负结果

题目 `1005` 在 BM25 和 dense 主运行中都没有发出搜索调用。我们依次测试了三个更小的干预：

| 控制策略 | 搜索次数 | 格式完整 | evidence recall | strict exact | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 首轮 512 Token deadline + 固定提醒 | 0 | 是 | 0% | 错 | prompt nudge 无法约束长推理 |
| 512 Token non-thinking 工具启动 | 1 | 是 | 0% | 错 | 修复“未搜索”，但 query 过于泛化 |
| 1,024 Token 三路稀有锚点 portfolio | 3 | 是 | 0% | 错 | non-thinking query compiler 仍丢失关键约束 |

离线手写的稀有锚点组合可以命中相关文档，说明失败位于 query 编译，而不是 dense index 完全不可用。但继续针对单题改 prompt 会造成 development overfitting。当前 `rare_anchor_portfolio_v0` 因而保留为负结果，不晋升为候选策略。

## 分层创新主线

### Layer 0：协议完整性守卫

问题：provider 接受参数不等于预算真的生效。

实现：强制 `max_tokens`，逐请求记录 requested/applied limit、thinking type、sampling、全局/阶段剩余 Token，并拒绝不符合版本契约的 trace。thinking 阶段不发送官方文档声明无效的 sampling 参数，non-thinking 阶段固定 `temperature=0`。

### Layer 1：Phase-Adaptive Reasoning

问题：high-thinking 同时用于探索和短答案编译，推理 Token 吞掉最终答案。

实现：探索阶段 high-thinking，编译阶段关闭 thinking，并在同一 10k 总预算内预留 2k。它改善的是系统交付行为，不是模型参数或固有能力。

### Layer 2：Retriever Counterfactual Replay

问题：端到端分数无法区分 query 质量和 retriever 质量。

实现：保存每次 query 和 BM25 top-5，对完全相同 query 离线重放 pinned dense index，再决定是否付费跑端到端。

### Layer 3：Constraint Portfolio Search

问题：长题被压成泛化词袋，稀有时间、教育、比赛和地理约束丢失。

状态：`v0` 已被 1005 bad case 否定。下一版必须在至少 10 个 retrieval bad cases 上先做 query-only replay，不能继续围绕一个答案调 prompt。

### Layer 4：Evidence-Debt Graph

问题：检索到文档不等于所有回答义务得到支持。

状态：项目 B2 已有 obligation -> evidence -> claim 的数据契约；BrowseComp-Plus 集成与消融仍为 `planned`，不得把已有代码当作 benchmark 收益。

### Layer 5：Marginal-Value Controller

问题：重复 query 消耗搜索、上下文和费用，但不减少未解决约束。

计划：按 evidence-debt reduction / Token / search call 选择下一动作；只有在较大开发集确认重复搜索是主失败簇后实现。

## 审计锚点

- strict standard summary SHA-256：`8cb00ab455ee1913ad4a57fed7114c26b8c7223b0158a658f070aab86a585c31`
- BM25 phase-adaptive summary SHA-256：`ff062778cf0563da76c52e200b150a302694bc8f16f64df939bde4af35332517`
- retrieval replay SHA-256：`0a8ac390cd6ed57b63be61ca67f2f857f28759a873b8fe73676351283dc55e76`
- corrected dense end-to-end summary SHA-256：`c913eb7e7e61e44747bac52db806368b900099ad451c28f3dfec3cbbc29c25ae`
- corrected dense diagnostic SHA-256：`e54a04aefaea715139ce74dc21a580f84533725655632e2b31893c1986d9dbd9`
- corrected official-input manifest SHA-256：`d23bbefa9bf20947096d2d36131132c9a6af2b3fd9525571a0dd3e4a7fc2c2bf`
- dense model revision：`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- index dataset revision：`b3f37f70c33829eb09d04784a54277a31871fd63`

早期使用 `max_completion_tokens` 的运行即使表面格式较好，也因输出超预算而被排除，不能与以上严格预算结果混用。

## 下一阶段与验收标准

1. **先跑官方 Judge**：使用 pinned Qwen3-32B revision 和固定 decoding，给 BM25 与 dense 两组 5 题预测打分。验收：零 parse failure，并报告逐题 Judge 原始结果；5 题只作管线门槛。
2. **先做重复运行**：adapter v6 固定可控 sampling 边界；thinking mode 仍无法设 seed。BM25 与 dense 每题至少独立运行 3 次，报告均值、标准差和 paired win/loss，禁止挑最好的一次。
3. **冻结 25 题开发切片**：不再针对 `1005` 调 prompt。运行相同 DeepSeek V4 Flash、phase policy 和 10k 预算的 BM25/dense paired ablation。验收：官方 accuracy、recall、search calls、Token、费用和延迟全部齐全。
4. **检索层晋升门槛**：dense 相对 BM25 的 evidence recall 至少 +10 pp，且官方 accuracy 不下降；否则回退，不堆 reranker。
5. **再诊断 query 编译**：只有 25 题中至少 10 个失败属于“相关文档未召回”，才开发 Constraint Portfolio v1。先 query-only replay，证据召回提升后再付费端到端。
6. **175 题 development**：冻结候选后运行完整开发集和官方 Judge，形成可信区间与 bad-case 分簇。
7. **655 题 sealed holdout**：只在机制、阈值和预算全部预注册后打开一次。DeepSeek V4 Pro 作为独立模型轨道，不与 Flash 混报。
8. **排行榜目标**：先越过冻结快照的 top-20 门槛 63.86%，再挑战 top-10 门槛 78.41%。达到阈值仍需同时公开同模型基线、费用、延迟和失败样本。

当前最近的阻塞不是实现，而是官方 Qwen3-32B Judge 所需 GPU；本机 4 GiB GPU 无法承载，已检查的远端 GPU 均在被其他任务占用，因此没有抢占或停止任何他人进程。
