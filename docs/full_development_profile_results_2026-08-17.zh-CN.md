# 完整 Development Failure Profile 结果（2026-08-17）

## 为什么跑这 175 题

此前连续机制选择集中在 7 个 retrieval miss 上，已经拒绝 passage BM25、dense
depth、raw-question anchor、rarity pivot、uniform RRF 和多种 prompt-only bridge。
继续在同一小簇上换机制会过拟合。因此本轮不引入新 Agent 或新检索器，只把已经
冻结的 Pi v10 Query-Aware Progressive Disclosure 单 Agent 策略运行在全部 175 道
development 问题上，目标是得到更可靠的失败分布。

模型参数保持冻结。本轮测量的是 harness 策略，不是模型能力提升。

## 执行完整性

- 模型：`deepseek-v4-flash`，high thinking，空 system prompt
- 策略：Pi v10，最多 8 次 search、8 次 open、10,000 output Tokens
- 检索器：BM25 5 个 anchor + Qwen3-Embedding-0.6B 15 个 query-aware lead
- 语料与 175 题 development partition 均由哈希固定
- sealed holdout：未访问
- 第一次执行：106 成功，随后 DeepSeek 返回 `402 Insufficient Balance`
- 恢复：余额与 artifact 哈希审计后仅重试 69 个失败 ID，106 个成功结果保持不变
- 最终：175 成功，0 失败，0 budget exhausted，1 次 failed-only resume
- provider/runtime patch 只保留真实错误并 fail closed，不改变研究行为

恢复前 summary SHA256 为
`f8a0fc04d7dd21ae68b927d85a939a7042d3a99b5242a6d92dfa657c39f5599a`；
最终 summary SHA256 为
`0799a3da6182a7fa5298f3e0ce0390899e5ed6902e1e3d5180612232d5780a2d`。

## 运行结果

| 指标 | 结果 | 状态 |
| --- | ---: | --- |
| 成功题数 | 175/175 | 通过 |
| schema complete | 172/175 | 通过预注册下限 168 |
| search calls | 1,177 | 记录值 |
| evidence open calls | 160 | 记录值 |
| evidence ingress Tokens | 2,735,917 | 记录值 |
| total Tokens | 13,651,521 | 记录值 |
| output Tokens | 890,197 | 记录值 |
| output budget overshoot | 0 | 通过 |
| provider cost | $1.1793086032 | 低于 $2.00 上限 |
| cumulative per-query latency | 10,193,225 ms | 记录值，不等于端到端墙钟 |

gold 仅在 175 份 prediction 冻结后读取。确定性 development diagnostics 为：

| 指标 | 结果 |
| --- | ---: |
| normalized exact | 42/175，24.00% |
| evidence recall | 52.51% |
| gold-document recall | 54.28% |

持久化 Qwen3-32B BF16 vLLM Judge 已通过既有 calibration，本轮 175 个请求全部成功，
0 parse failure、0 request failure，判对 65/175，即 37.14%。该数值的名称是
`calibrated_development_diagnostic_not_official`，不能替代 upstream official
evaluator。

## 预注册失败分布

| 类别 | 全部题数 | Judge-wrong 内占比 |
| --- | ---: | ---: |
| Judge correct | 65 | - |
| reference document retrieved, answer wrong | 67 | 60.91% |
| reference document not retrieved | 40 | 36.36% |
| answer contract failure | 3 | 2.73% |

共 110 个 Judge-wrong case。预注册路由规则规定：若“reference document retrieved,
answer wrong”达到 wrong cases 的 60%，下一层必须是 evidence selection、opening 或
synthesis。实际 67/110 刚好越过该门槛，因此机器决策为
`evidence_selection_opening_or_synthesis`。

这也修正了此前“小样本错误主要是 retrieval miss”的判断。小簇诊断本身没有错，
但它不能代表完整 development 分布；扩大覆盖后，主失败类已经转移到证据到达后的
处理链路。

## 下一步

先做零 provider call 的 Evidence Reachability Funnel，不立即修改 Agent：

1. reference document 是否进入任一 search result；
2. gold answer 的 literal atoms 是否实际出现在暴露给 Agent 的 snippet/open content；
3. answer-bearing reference 是否被最终回答引用；
4. 已见且已引用 answer-bearing evidence 时，最终答案是否仍被 Judge 判错。

只有这四层的分布冻结后，才能选择最小机制：span opening、evidence selection、
citation-grounded extraction 或 synthesis verifier。40 个 retrieval miss 进入另一条
分层队列，不能和本轮一起改。多 Agent 继续延后；多个 Researcher 共用同一证据处理
缺陷不会自动修复 67 个 downstream case。

## 产物

- registration：`benchmarks/browsecomp_plus_v0/full_development_profile_v0.json`
- taxonomy：`benchmarks/browsecomp_plus_v0/full_development_failure_taxonomy_v0.json`
- final summary：`runs/browsecomp_plus_v0/full-development-profile-v0-20260817/v10/summary.json`
- development gold：`runs/browsecomp_plus_v0/full-development-profile-v0-20260817/development_gold_175.json`
- deterministic diagnostic：`runs/browsecomp_plus_v0/full-development-profile-v0-20260817/gold_diagnostic.json`
- calibrated Judge：`runs/browsecomp_plus_v0/full-development-profile-v0-20260817/persistent-judge/evaluation/development_judge_result.json`
- failure profile：`runs/browsecomp_plus_v0/full-development-profile-v0-20260817/failure_profile.json`

这些结果来自已用于策略选择的 development partition。它们不是 fresh same-model
对照、upstream-official BrowseComp-Plus accuracy、sealed-holdout 结果、leaderboard
名次、跨框架优越性或模型能力提升。
