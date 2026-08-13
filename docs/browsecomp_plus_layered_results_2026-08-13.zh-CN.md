# BrowseComp-Plus 分层创新实验记录（2026-08-13）

## 结论边界

当前结果来自 5 道 development 题，只用于机制门槛，不能代表 175 题 development、655 题 sealed holdout 或排行榜名次。模型参数始终冻结；变化来自检索器、推理阶段调度、提示和预算控制，不能表述为模型能力提升。

固定 revision 的官方 Qwen3-32B evaluator 已对三轮共 30 份冻结输出完成评分，零 parse failure。本页会把这个结果明确称为“5 题 official-evaluator development slice”，而不是完整 benchmark accuracy。normalized exact match 仍是刻意严格的字符串诊断，会把带单位、解释或等价改写的答案判错。

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

稠密检索输出中，另外两题与参考答案语义一致但分别多了括号说明或单位。后续官方 evaluator 的三轮 dense 分数为 40%、60%、80%，说明单次“约 3/5”的人工推断不能替代重复评测。

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
46.67% 仍只是 strict exact；官方 evaluator 的对应重复结果单列如下，二者不能混报。

## 官方 evaluator 五题门槛

冻结 Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`、
upstream evaluator、ground truth、30 份预测和 decoding 参数后，三轮结果为：

| trial | 执行顺序 | BM25 | Dense |
| --- | --- | ---: | ---: |
| trial-01 | baseline first | 20% | 40% |
| trial-02 | candidate first | 20% | 60% |
| trial-03 | baseline first | 20% | 80% |
| mean +/- sample std |  | 20.00% +/- 0.00 | 60.00% +/- 20.00 |

30 个 query-trial 判断全部解析成功；15 个配对观察中 dense 为 7 胜、1 负、
7 平。execution registration、execution result 和本地 comparison 的 SHA-256
分别为 `ef65a0f8687f3f54bb81d9061da4dd525a41c438c907ea237493f5ec50d87f1a`、
`c211d2f6eea0f14506c4184d154d17d3e1171f20eb8ade3cb486b525a0238388` 和
`af281142b924778851d17ede5eae18d7fd4f49b26885077b1cdcc4da7308e7ef`。
逐题运行产物继续保存在 ignored `runs/`，仓库文档只记录稳定哈希和聚合结论，
不提交 benchmark 派生 payload。

第一次执行在 NCCL communicator 初始化处约 10 分钟无进展，未进入权重加载、
未生成任何判断，随后只终止本次登记的进程组并保留失败目录。重试显式记录
`NCCL_P2P_DISABLE=1`，通过 shared-memory transport 完成；这只改变运行时通信
路径，不改变模型、预测、官方脚本或 decoding。成功执行耗时约 13.5 分钟，
其中 471.9 秒用于从共享盘加载权重。该失败与修复属于基础设施诊断，不是可调
分数的 harness 创新。

这仍然只重复使用 5 道固定题，不能把 30 个判断当成 30 道独立题，也不能据此
声称 60% 的完整 benchmark accuracy 或排行榜位置。它完成的是进入 25 题扩展
实验所需的门槛。

把同一批 repeat、diagnostic 与官方 Judge 结果送入可审计的 layer-promotion gate
后，evidence recall 差为 +42.64 pp、官方 accuracy 差为 +40.00 pp，两个机制门槛
均通过；但 5 题小于预注册的 25 题下限，且旧 repeat manifest 是
`reconstructed_after_interruption`，因此最终机器决策是
`insufficient_scope`，不是 `promote`。这阻止我们拿漂亮的 5 题结果提前宣布
dense 已成为最终方案。

候选的 15 个官方判断中有 6 个错误：5 个属于“没有召回任何 gold/evidence
文档”，1 个属于“已召回相关文档但答案仍错”，0 个属于格式失败；按题统计为
1 个三轮持续失败、2 个不稳定、2 个三轮稳定成功。这个分层诊断说明当前下一步
仍应完成 25 题检索层门槛，而不是立刻添加 Critic 或更多 Agent。机器决策文件
保存在 ignored `runs/`，SHA-256 为
`0226e3a842862ec5ebc87354865136a27c9218df4597ee2291e52992fae5cfa4`。
机器文件是在部分运行后补成，但阈值并非看过分数后选择：`+10 pp recall`
和“官方 accuracy 不下降”已经存在于运行前提交
`dd25e78b90f6f83a85009461585e2d75604c004c`。因此它记录为
`formalized_after_generation_from_precommitted_thresholds`，不伪装成提前冻结的
文件哈希。

## 25 题配对扩展：余额中断，不报中间分数

`pre_generation` 的三轮 BM25/dense 交替网格共有 150 个 query-variant
观测。当前五个 variant 已完整结束，最后一个 dense variant 完成 20/25 后，
DeepSeek 返回 `402 Insufficient Balance`：一条失败记录保留了中断前的部分
Token/费用，另外四条在调用开始即失败。当前合计 145 succeeded、5 failed，
2,259 次搜索、1,132,443 output Token、46,839,467 total Token，已记录的
DeepSeek API 费用为 2.098859336 USD。

这些只是运行进度和资源账本，不是 25 题效果结果。六个 variant 尚未全部冻结，
所以不计算、不挑选也不报告任何中间 accuracy、recall 或胜负。中断 summary 的
SHA-256 为 `6c3d88b1673a26bf29423916f7a4485a0048a3cd721f0183734a3162af8208f2`。

余额恢复后，同一个 manifest 的 `-Resume` 只会重试这五条 `failed` 记录：
先校验当前 request/run/prediction 与既有 attempt 链，再归档被替代的失败尝试和
source summary。失败尝试消耗的 Token、费用、搜索和延迟继续计入累计预算；
已成功的 145 条不会重跑。这个恢复规则是在观察到 402 后新增的运维修正，因此
最终结果必须标注为“预注册生成网格 + 事后披露的失败恢复规则”，不能把它写成
完全未受中断影响的确认性实验。

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
- official evaluator manifest SHA-256：`6a011f90894c1222de81654ffafe16cf266d52f74a9f93e619f40580afa8faf1`
- self-contained Judge batch manifest SHA-256：`2e924c1322a33225cee59d194c0515ada41e8db0ada5c6997700d11db373220c`
- 30 份 Judge 输入的 canonical set SHA-256：`7607a6fe4548c9e4420ca0270ee4ecb3d4437d050ecc773e043a154e009b92f0`
- 25 题 question-only artifact 内容 SHA-256：`e62a37c48e1482fb39771d9366e3c6b1cefb33624e1955dc014855f49c9669ed`
- 25 题 artifact normalized file SHA-256：`d3925f67fcce34b6e9bb3fec86ff6560afdfedf82ebcf2b12480436e69d3b923`
- 25 题 corrected `pre_generation` manifest raw SHA-256：`b9833b69cd19672b83ad0a172d2382efd5a0981e207412494ec25a889f998fa2`
- 25 题 partial final-variant summary SHA-256：`6c3d88b1673a26bf29423916f7a4485a0048a3cd721f0183734a3162af8208f2`
- layer-promotion gate manifest SHA-256：`dbf69941239f3cea0ac9ef84874805b0a79a5f73a3c05d3d859fd1bdefafa99e`
- 5 题 layer decision SHA-256：`0226e3a842862ec5ebc87354865136a27c9218df4597ee2291e52992fae5cfa4`
- dense model revision：`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`
- index dataset revision：`b3f37f70c33829eb09d04784a54277a31871fd63`

早期使用 `max_completion_tokens` 的运行即使表面格式较好，也因输出超预算而被排除，不能与以上严格预算结果混用。

## 下一阶段与验收标准

1. **已通过官方 Judge 门槛**：pinned Qwen3-32B 对 30/30 份冻结输出完成评分，零 parse failure；结果只标记为 5 题 development slice。
2. **25 题开发切片等待余额恢复**：不再针对 `1005` 调 prompt。第一次生成前已写入 `pre_generation` manifest；当前五个 variant 完成，最后一个为 20 succeeded/5 failed。恢复后只补失败项。验收：六个 variant 全部成功、官方 accuracy、recall、search calls、累计 Token、累计费用和延迟齐全，且披露事后恢复规则。
3. **检索层晋升门槛**：dense 相对 BM25 的 evidence recall 至少 +10 pp，且官方 accuracy 不下降；否则回退，不堆 reranker。
4. **再诊断 query 编译**：只有 25 题中至少 10 个失败属于“相关文档未召回”，才开发 Constraint Portfolio v1。先 query-only replay，证据召回提升后再付费端到端。
5. **175 题 development**：冻结候选后运行完整开发集和官方 Judge，形成可信区间与 bad-case 分簇。
6. **655 题 sealed holdout**：只在机制、阈值和预算全部预注册后打开一次。DeepSeek V4 Pro 作为独立模型轨道，不与 Flash 混报。
7. **排行榜目标**：先越过冻结快照的 top-20 门槛 63.86%，再挑战 top-10 门槛 78.41%。达到阈值仍需同时公开同模型基线、费用、延迟和失败样本。

第 2 步的 25 题问题文件和三轮 BM25/dense 交替执行 manifest 已于
2026-08-14 用 `-RegisterOnly` 冻结，状态为 `pre_generation`。第一次草稿只绑定了文件路径，我们在启动前识别出
“同路径替换题目文件”这一协议漏洞，因此将其以 0 次调用状态 supersede；
corrected v2 同时绑定 normalized file SHA-256。相同参数 resume 保持 manifest
哈希不变，改用 5 题文件则在 provider call 前被拒绝。5 题官方 Judge 门槛通过
后，25 题付费运行已经按原 manifest 启动。运行在第六个 variant 的 20/25 处因
API 余额不足暂停；完成并重新聚合前不报告中间分数。

截至 2026-08-14，官方仓库、锁定依赖、Qwen3-32B 权重、30 份冻结输入和
development ground truth 已在远端准备完成；通过镜像取得的 17 个权重分片及
7 个配置/索引/tokenizer 文件已经逐字节匹配 pinned Hugging Face revision。
两张 48 GiB GPU 空闲后，启动器在 GPU 4/5 上完成三次连续空闲检查并运行成功；
没有抢占或停止任何他人进程。当前可报告的是 5 题 official-evaluator slice，
不是完整 benchmark accuracy。

Judge 启动边界已不再依赖手工拼命令：自包含 batch 会绑定每个
`trial -> variant -> query` 的预测、ground truth、repeat/evaluator/权重清单；
启动器只接受 clean upstream commit、固定 lockfile、24/24 资产校验和三次连续
空闲检查均通过的两张 GPU。我们已在真实远端对 GPU 0/1 做过占用失败路径验证：
检测到占用后在创建 execution 目录前退出，因此没有残留能被误认为评分结果的
半成品。GPU 4/5 上的第一次正式执行又暴露 NCCL P2P 初始化挂起，审计后的
transport-only override 修复了运行时问题；第二次执行完整产出 30 个结果并通过
本地哈希聚合。
