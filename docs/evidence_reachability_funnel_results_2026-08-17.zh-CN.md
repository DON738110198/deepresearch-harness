# Evidence Reachability Funnel v0 结果（2026-08-17）

## 问题

完整 175 题 profile 中，110 个 Judge-wrong case 有 67 个被初始 taxonomy 标为
`reference_document_retrieved_answer_wrong`。这个标签仍然太粗：它没有区分答案 span
未暴露、可见证据未选择，以及已有直接证据但最终合成错误。

本轮在读取逐题 trace 前固定了一个零调用 funnel：reference docid 定义为
`gold_docs ∪ evidence_docs`，可见内容只包括保存的 reference snippet 和成功 open
内容；随后用相同 answer-atom coverage 检查全部可见 reference 内容和最终回答实际
引用的 reference 内容。全过程 provider、online search、Judge 和 sealed holdout
访问均为 0。

## 注册结果

| 阶段 | Case 数 |
| --- | ---: |
| reference 到达 | 67/67 |
| 至少一个 reference 可 open | 51/67 |
| 实际显式 open reference | 9/67 |
| literal answer 完整可见 | 20/67 |
| literal answer 未完整可见 | 47/67 |
| answer 可见但 reference 未提供完整 cited coverage | 11/67 |
| cited reference 已含完整 answer 但 Judge 仍错 | 9/67 |

`47/67 = 70.15%`，超过预注册的 60% 门槛，因此机器输出的 `next_layer` 是
`evidence_exposure_or_opening`。在 20 个 answer-visible case 内，uncited 与 cited-wrong
分别为 55% 和 45%，没有任何一个达到 60%。

## 为什么不直接实现 opening

v0 的 reference policy 是 `gold_docs ∪ evidence_docs`。这与最初 failure taxonomy
保持一致，但也暴露了 taxonomy 本身的混淆：只召回 supporting evidence doc 就能进入
这 67 题，即使最终 gold document 尚未到达。此时 literal answer 不可见可能仍是多跳
retrieval incomplete，而不是 span-opening failure。

因此保留 `47/67` 和注册路由，但拒绝把它直接当作 runtime mechanism selector。这个
决定也避免重复此前已被 paired trial 拒绝的全局 obligation-span opening。下一步是
一个明确标记为 selected-after-v0 的纠偏审计，将全部 110 个 wrong case 划为互斥类：

1. answer contract failure；
2. 无 reference 到达；
3. 仅 supporting evidence 到达且 answer hidden；
4. gold document 到达但 answer hidden；
5. answer visible but uncited；
6. answer-bearing reference cited but wrong。

该纠偏仍只读保存的 trace，不设置 promotion threshold，也不调用模型。只有纠偏分布
冻结后，才分别为 retrieval、exposure、selection、synthesis 建立互不污染的队列。

## 审计产物

- registration：`benchmarks/browsecomp_plus_v0/evidence_reachability_funnel_v0.json`
- implementation：`src/deepresearch_harness/evidence_reachability_funnel.py`
- runner：`scripts/run_evidence_reachability_funnel.py`
- tests：`tests/test_evidence_reachability_funnel.py`
- ignored result：`runs/browsecomp_plus_v0/evidence-reachability-funnel-v0-20260817/funnel.json`
- result SHA256：`7aee84b3d8528c13b730c218f197ec198afeed05989d15c2382fddf5fe4ed92a`

本轮是 gold-aware、posthoc development localization。它不是效果提升、官方 accuracy、
sealed-holdout 结果、leaderboard 名次、框架优越性或模型能力提升。
