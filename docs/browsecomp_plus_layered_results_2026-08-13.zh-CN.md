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

## adapter v6 三轮配对结果

为避免挑选最好的一次，我们在同一 5 题 development 切片上运行了 3 轮
BM25/dense 配对实验，共 30 个独立 provider run。固定项包括 DeepSeek V4
Flash、空 system prompt、Pi 0.84.1、adapter v6、8k high-thinking explore +
2k non-thinking compile、top-5、512 Qwen-token snippet、10k 输出上限和题目
顺序。执行顺序为 `BM25 -> dense`、`dense -> BM25`、`BM25 -> dense`。

| trial-level 指标 | BM25 mean +/- sample std | Dense mean +/- sample std | 方向 |
| --- | ---: | ---: | --- |
| 格式完整率 | 86.67% +/- 11.55 | 100.00% +/- 0.00 | dense 更高 |
| strict exact | 6.67% +/- 11.55 | 46.67% +/- 11.55 | dense 更高；仍非官方分数 |
| evidence recall | 17.36% +/- 2.88 | 60.00% +/- 14.84 | +42.64 pp |
| gold recall | 20.00% +/- 0.00 | 66.67% +/- 11.55 | +46.67 pp |
| 每题搜索次数 | 11.27 +/- 2.52 | 6.40 +/- 2.80 | dense 更少 |
| 每题输出 Token | 8,097.73 +/- 876.58 | 5,848.93 +/- 814.29 | dense 更少 |
| 每题总 Token | 208,353.07 +/- 63,467.92 | 81,843.93 +/- 56,046.36 | dense 更少 |
| 每题估算费用（USD） | 0.011472 +/- 0.002356 | 0.005662 +/- 0.002053 | dense 更低 |
| 每题延迟（ms） | 88,876.60 +/- 9,222.08 | 59,770.87 +/- 7,169.28 | dense 更低 |

15 个 query-trial 配对观察中，dense 的 strict exact 为 6 胜 / 0 负 / 9
平，evidence recall 为 8 胜 / 2 负 / 5 平，费用为 13 胜 / 2 负 / 0 平。
这些计数重复使用同一组 5 题，不能当成 15 道独立 benchmark 样本。
三轮合计估算 API 费用为 0.25701789 USD；所有运行均为 5/5 succeeded，
且输出预算 overshoot 为 0。

费用只包含 DeepSeek provider trace；本地 dense index 的 GPU/机器成本没有
折算成美元。两组固定的是相同 10k 最大输出 allowance，而不是事后强行让
实际总 Token 或实际费用相等。因此这轮用于检验“替换检索器后系统行为如何
变化”，不能称为完整的 cost-matched 或 total-token-matched 胜利。

这次自动化第一次启动时，在 trial 1 BM25 完成后因评测 Python 缺少
`duckdb` 中断。随后我们打开了该轮 development gold，修复的只有依赖
preflight、恢复与聚合代码，没有改动生成 prompt、adapter、模型、预算或
检索器。尽管如此，它的 manifest 必须诚实标为
`reconstructed_after_interruption`，因此本结果仍是探索性诊断，不是预注册
确认性实验。下一次 25 题运行必须在第一次 API 调用前写好完整 manifest。
官方 Qwen3-32B Judge 仍未运行，46.67% 不能写成官方 accuracy。

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
- v6 repeat experiment raw SHA-256：`0b610af6a5640e82c47bbe969ff3e917b54ba28b4967c58b20b3c9ea04071c9e`
- v6 repeat comparison raw SHA-256：`6b649973f3e952e098cc0dcf6f46e958e054f9861fc0c2e4c8d92eed960ab0b4`
- official development ground-truth JSONL SHA-256：`9a975130c225bc66fa5a1fa206098bb2458ca782150e86339a72b63417c7d259`
- official Judge 资产 manifest SHA-256：`0308e17cc6d113fb1871ea6c4c35590575d4ebc924abb80df0a6a443b79bb9a3`
- official Judge 资产校验结果：24/24 通过；审计文件 SHA-256 为 `f919e78fe5e8346aa84432616cbbec8bc32588387eebd18ce7d893c919215719`
- upstream evaluator SHA-256：`1a21233937c377ab6323c98ff9af67742756a57fbacab4ebf9bc30852eae530a`
- official Judge `uv.lock` SHA-256：`45d3e6d00719dbf732160b25e3419ed4599121e5d832723357ff2fea01477c43`
- dense model revision：`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- index dataset revision：`b3f37f70c33829eb09d04784a54277a31871fd63`

早期使用 `max_completion_tokens` 的运行即使表面格式较好，也因输出超预算而被排除，不能与以上严格预算结果混用。

## 下一阶段与验收标准

1. **先跑官方 Judge**：使用 pinned Qwen3-32B revision 和固定 decoding，给三轮共 6 组冻结预测打分。验收：30/30 有逐题原始结果、零 parse failure，并聚合 official accuracy 的均值、标准差和 paired win/loss；5 题仍只作管线门槛。
2. **冻结 25 题开发切片**：不再针对 `1005` 调 prompt。第一次生成前写入 `pre_generation` manifest，运行相同 DeepSeek V4 Flash、phase policy 和 10k 预算的 BM25/dense paired ablation。验收：官方 accuracy、recall、search calls、Token、费用和延迟全部齐全。
3. **检索层晋升门槛**：dense 相对 BM25 的 evidence recall 至少 +10 pp，且官方 accuracy 不下降；否则回退，不堆 reranker。
4. **再诊断 query 编译**：只有 25 题中至少 10 个失败属于“相关文档未召回”，才开发 Constraint Portfolio v1。先 query-only replay，证据召回提升后再付费端到端。
5. **175 题 development**：冻结候选后运行完整开发集和官方 Judge，形成可信区间与 bad-case 分簇。
6. **655 题 sealed holdout**：只在机制、阈值和预算全部预注册后打开一次。DeepSeek V4 Pro 作为独立模型轨道，不与 Flash 混报。
7. **排行榜目标**：先越过冻结快照的 top-20 门槛 63.86%，再挑战 top-10 门槛 78.41%。达到阈值仍需同时公开同模型基线、费用、延迟和失败样本。

截至 2026-08-14，官方仓库、锁定依赖、Qwen3-32B 权重、30 份冻结输入和
development ground truth 已在远端准备完成；通过镜像取得的 17 个权重分片及
7 个配置/索引/tokenizer 文件已经逐字节匹配 pinned Hugging Face revision。
当前最近的阻塞只剩官方 Judge 所需 GPU：它需要两张空闲 48 GiB GPU，而最新
检查时远端 8 张卡都有其他任务，因此没有抢占或停止任何他人进程。官方
accuracy 仍为 `planned_not_run`。
