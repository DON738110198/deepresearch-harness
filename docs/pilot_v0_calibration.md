# Pilot v0 AI-Assisted Calibration

## Status and claim boundary

This is a rubric calibration and bad-case audit, not the registered human evaluation. The reviewer was AI-assisted, the submission is labeled `calibration_only`, and no result below should be presented as a human metric or a model capability improvement.

Post-run audit found that the B0/B1 batch passed `task.question` to the runtime but omitted the suite's separate `decision_context`. The table remains useful for diagnosing the observed artifacts, but it is not accepted as the final B0/B1 baseline. Future matched manifests must use the corrected shared input for every variant.

The run used the frozen token-matched B0/B1 manifest at implementation revision `2781fc9389d450c465a4ea19d76e5f21c4290406`. Both variants used the same `deepseek-v4-flash` endpoint, local corpus snapshot, top-k, task suite, and per-task token cap.

## Locked review provenance

- Batch: `20260812T081209Z`
- Review packet: `pilot-v0-token-20260812-v2`
- Candidates reviewed before unblinding: 20/20
- Annotation SHA-256: `be24156c4a38cbaab188392c7e0285abcf933f8542bcc81922cc0c9bcc26886e`
- Annotation type: `ai_assisted`
- Result status: `calibration_only`

## Observed results

| Metric | B0 Search-Write | B1 Plan-Search-Ledger-Write |
| --- | ---: | ---: |
| Completed tasks | 10/10 | 10/10 |
| Total tokens | 11,794 | 21,349 |
| Estimated API cost | $0.00215726 | $0.00365946 |
| Automatic evidence-ID recall | 0.6000 | 0.8000 |
| Automatic evidence-ID precision | 0.3000 | 0.4167 |
| Structural citation integrity | 1.0000 | 1.0000 |
| AI-assisted obligation coverage | 0.6000 | 0.7667 |
| AI-assisted citation support rate | 1.0000 | 1.0000 |
| AI-assisted unsupported-claim rate | 0.0000 | 0.0000 |
| AI-assisted irrelevant-claim rate | 0.3417 | 0.0500 |
| AI-assisted conflict-handling rate | 0.5000 | 1.0000 |

The calibration suggests that B1's clearest effect is filtering decision-irrelevant claims, not changing the frozen model. B1 used 81.0% more tokens and an estimated 69.6% more API cost in this run. A registered human review is still required before making a semantic performance claim.

## Saved B1 bad cases

B1 left seven required obligations uncovered across six tasks:

| Task | Missing obligation(s) | Failure layer |
| --- | --- | --- |
| vector index | retrieval recall | search did not collect the recall record |
| reranking | first-stage candidate recall | evidence was collected but omitted from the ledger/report |
| agent memory | relevance and freshness | plan/search returned only the privacy record plus a routing distractor |
| multi-agent | coordination cost | plan and search focused on dependency only |
| context compaction | token-saving need | plan/search covered loss and durable retention only |
| evaluation allocation | judge reliability | plan/search covered task coverage and cost allocation only |

These failures share one causal pattern after accounting for the missing-context flaw: the plan stores free-form steps and an unlinked query list, while the ledger has no auditable mapping from answer obligation to query, evidence, and claim. Increasing top-k is not selected because it would add distractors and cannot repair the reranking case, where the needed evidence was already present. A critic loop or Research DAG is also premature because the missing coverage can be tested without adding an LLM call.

## Next hypothesis

**Problem -> evidence -> hypothesis -> change -> metric -> acceptance**

- Problem: B1 often produces well-supported claims but silently omits one decision criterion.
- Evidence: seven uncovered obligations; one survives retrieval but is dropped by the ledger.
- Hypothesis: an explicit obligation-to-query-to-evidence-to-claim contract will expose evidence debt and reduce silent omissions.
- Change: add B2 with an obligation plan and persisted evidence-debt status, keeping the same three provider calls, collector, top-k, model, and budgets.
- Metric: blind obligation coverage is primary; citation support, completion rate, tokens, cost, and irrelevant-claim rate are guardrails.
- Acceptance: the exact threshold remains `planned` until the B1/B2 experiment manifest is registered before execution.
