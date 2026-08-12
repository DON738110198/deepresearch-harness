# B1 vs B2 Token-Matched v2

## Status

The pre-registered automatic improvement gate **failed**. B2 v2 repaired the aggregate retrieval regression from v1 but did not exceed B1 by the required `0.10`. Cost-matched and human evaluation remain unrun.

## Provenance

- Manifest: `experiments/pilot_v0/b1_b2v2_token_matched.json`
- Implementation revision: `96679621e4c97b8a478802714ec5f82a44793141`
- Manifest registration commit: `33a18a1`
- Batch: `20260812T101100Z`
- Provider/model: OpenAI-compatible `deepseek-v4-flash`, thinking disabled
- Shared controls: question plus decision context, corpus snapshot, lexical collector, top-k 6, three LLM calls, 8,000 observed total-token cap per task

## Automatic results

| Metric | B1 | B2 v2 |
| --- | ---: | ---: |
| Completed | 10/10 | 10/10 |
| Evidence-ID recall | 0.8333 | 0.8333 |
| Evidence-obligation recall | 0.8333 | 0.8333 |
| Evidence-ID precision | 0.4333 | 0.4650 |
| Structural citation integrity | 1.0000 | 1.0000 |
| Total tokens | 21,987 | 26,032 |
| Estimated API cost | $0.00369490 | $0.00479458 |
| Median traced latency per task | 5,481.5 ms | 7,376.0 ms |

B2 v2 tied B1 on retrieval obligation recall, improved retrieval precision by `0.0317`, and cost 18.4% more tokens, 29.8% more estimated API cost, and 34.6% more median traced latency. This does not justify a positive result claim.

## What changed from v1

The domain-specific answer-contract instruction recovered the vector-index recall criterion and the model-routing quality criterion. It no longer showed v1's aggregate `-0.10` recall regression. Remaining B2 misses included first-stage recall for reranking and checkpoint storage/overhead evidence.

## Search-layer diagnosis

The B2 planner explicitly generated queries for reranker application stage and checkpoint frequency/overhead, but the collector merged all query terms into one global score. This allowed several high-scoring records for one obligation to consume the entire top-k and left other obligation queries without a guaranteed evidence slot.

An offline counterfactual replay used the saved plans and corpus, made no provider calls, and changed only selection to round-robin across per-query lexical rankings. It produced the following diagnostic automatic recall:

| Saved-plan replay | Current global merge | Query-balanced counterfactual |
| --- | ---: | ---: |
| B1 evidence-obligation recall | 0.8333 | 0.7667 |
| B2 evidence-obligation recall | 0.8333 | 0.9000 |

These counterfactual values are diagnostic, not experiment results: the ledger and report were not rerun. They justify implementing a shared query-balanced collector policy and then registering a new full experiment. The collector change must apply equally to B1 and B2.
