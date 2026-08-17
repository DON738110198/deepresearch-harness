# Gold/Evidence Causal Funnel v1 结果（2026-08-17）

## 为什么要纠偏

v0 的 `reference = gold_docs ∪ evidence_docs` 适合复现最初 taxonomy，但不适合选择
runtime 机制：只要召回 supporting evidence，就可能进入“reference 已到达”的队列，哪怕
答案所在的 gold document 从未出现。直接据此加入 span opening 或多 Agent，会把检索遗漏
误判成证据暴露问题。

因此 v1 只重读 175 题完整 development profile 中 110 个 calibrated-Judge-wrong trace，
将 gold docids 与 supporting-evidence docids 分开，并同时记录四种 literal-answer
coverage：gold-only、supporting-only、两者并集、最终 bracket-cited 文档。全程没有
provider、online search、Judge、GPU 或 sealed-holdout 调用。

## 结果

| 互斥类别 | Case 数 | 对应的后续队列 |
| --- | ---: | --- |
| answer contract failure | 3 | answer contract |
| 无 gold 或 supporting evidence 到达 | 40 | retrieval / evidence frontier |
| 仅 supporting evidence 到达 | 21 | retrieval / evidence frontier |
| gold 已到达但 gold span 未完整可见 | 31 | evidence exposure |
| gold span 可见但最终未引用完整答案证据 | 8 | evidence selection |
| 已引用含完整答案的证据但 Judge 仍错 | 7 | synthesis verification |

合并后的候选队列是：retrieval/evidence frontier `61`，evidence exposure `31`，evidence
selection `8`，synthesis verification `7`，answer contract `3`。

这里的 `supporting-only=21` 不能被写成“答案 hidden”：其中 `3` 个 trace 的 supporting
evidence 已能拼出 literal answer；同样，`31` 个 gold-span-incomplete trace 中有 `2` 个在
与其他文档合并后能看到答案。故 v1 保存了正交 coverage 字段，而不是把这些情况压成一个
“hidden”标签。

## 决策

纠偏审计被接受为后续 queue map，但**不**是 runtime 机制的 promotion：110 个 case 是
在 Judge 结果和 v0 观察之后选出的，不能拿它们报告效果。下一次实际的机制测试必须：

1. 只针对一个队列，保留 no-reference 与 supporting-only 的分层；
2. 使用此前未用于机制选择的 fresh development slice；
3. 固定模型、语料、retriever、search calls、Token、费用和 Judge；
4. 不重复已拒绝的全局 span opening；
5. 多 Agent 继续 deferred，除非 trace 新出现独立分支遗漏、单上下文干扰、未核验矛盾证据
   或串行研究延迟。

## 审计产物

- registration：`benchmarks/browsecomp_plus_v0/gold_evidence_causal_funnel_v1.json`
- implementation：`src/deepresearch_harness/gold_evidence_causal_funnel.py`
- runner：`scripts/run_gold_evidence_causal_funnel.py`
- tests：`tests/test_gold_evidence_causal_funnel.py`
- result：`runs/browsecomp_plus_v0/gold-evidence-causal-funnel-v1-20260817/funnel.json`
- result SHA256：`cb2390cf0b7b6a785b1dfcd25f76cf0f08ff8ff31df4b45b5345de2c3530f2f5`
- closed checkpoint：`experiments/gold_evidence_causal_funnel_v1/checkpoint.json`

本轮是 selected-after-v0 的 gold-aware posthoc development localization，不是 harness
效果提升、官方 accuracy、sealed-holdout 结果、leaderboard 名次、框架优越性或模型能力提升。
