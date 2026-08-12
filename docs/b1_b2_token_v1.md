# B1 vs B2 Token-Matched v1

## Status

The pre-registered automatic gate **failed**. This negative result is retained as evidence for the next design decision. No human semantic result is claimed, and the cost-matched batch was not launched after the token-matched primary gate failed.

## Provenance

- Manifest: `experiments/pilot_v0/b1_b2_token_matched.json`
- Implementation revision: `6e559874b2745a7b01d53e201656628b9ea97514`
- Manifest registration commit: `76304ad`
- Provider/model: OpenAI-compatible `deepseek-v4-flash`, thinking disabled
- Batch: `20260812T100350Z`
- Runtime input: task question plus decision context
- Shared controls: corpus snapshot, lexical collector, top-k 6, three LLM calls, 8,000 observed total-token cap per task

## Automatic results

| Metric | B1 | B2 Evidence Debt |
| --- | ---: | ---: |
| Completed | 10/10 | 10/10 |
| Evidence-ID recall | 0.8667 | 0.7667 |
| Evidence-obligation recall | 0.8667 | 0.7667 |
| Evidence-ID precision | 0.4667 | 0.4100 |
| Structural citation integrity | 1.0000 | 1.0000 |
| Total tokens | 21,263 | 26,053 |
| Estimated API cost | $0.00359534 | $0.00460048 |
| Median traced latency per task | 5,078.5 ms | 6,546.0 ms |

The registered retrieval gate required B2 to exceed B1 evidence-obligation recall by at least `0.10`. Observed B2 was instead `0.10` lower. B2 also used 22.5% more tokens, 28.0% more estimated API cost, and 28.9% more median traced latency.

## Bad-case localization

B2 lost one required evidence obligation relative to B1 on each of three tasks:

| Task | Missed evidence | Observed cause |
| --- | --- | --- |
| reranking | first-stage candidate recall | generic quality/latency obligations omitted the first-stage retrieval criterion |
| checkpointing | storage/overhead side of frequency policy | generic overhead wording retrieved failure-frequency and idempotency records instead |
| model routing | quality-based routing condition | generic benefit/difficulty queries retrieved latency/privacy records but not the quality gate |

The v1 prompt explicitly encouraged generic `benefit/risk/constraint/trade-off` slots. Those slots were auditable, but they were not reliably aligned with the domain-specific external rubric. Evidence Debt can expose omissions relative to its own answer contract; it is not an independent verifier of contract completeness.

## Next smallest test

- Keep the same B2 contracts, provider-call count, collector, top-k, corpus, model, and budgets.
- Change only the obligation-planning instruction: require domain-specific workflow stages, comparison axes, failure modes, and hard constraints; explicitly forbid generic benefit/risk/constraint/trade-off slots unless the input asks for them.
- Re-register a new implementation revision before running.
- Retain the same automatic gate. Do not launch cost-matched or human evaluation unless token-matched passes.
