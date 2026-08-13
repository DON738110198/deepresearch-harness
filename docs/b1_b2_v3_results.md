# B1 vs B2 v3 Results

## Claim boundary

B2 v3 is a training-free harness variant. The model and weights were fixed. The results below are automatic retrieval/structure measurements on a ten-task synthetic pilot, not a model capability gain and not a human semantic evaluation.

## Frozen setup

- Implementation revision: `4049793cce0df6ca2ca9530c8a00742edb9d50ff`
- Provider/model: OpenAI-compatible `deepseek-v4-flash`, thinking disabled
- Shared runtime input: question plus decision context
- Shared tools: query-balanced lexical collector, same corpus snapshot and top-k 6
- Shared call allowance: three LLM calls per task
- B2-only harness logic: domain-specific answer obligations plus persisted obligation -> query -> evidence -> claim Evidence Debt

## Token-matched

- Manifest: `experiments/pilot_v0/b1_b2v3_token_matched.json`
- Batch: `20260812T101940Z`
- Cap: 8,000 observed input plus output tokens per task

| Automatic metric | B1 | B2 v3 |
| --- | ---: | ---: |
| Completed | 10/10 | 10/10 |
| Evidence-ID recall | 0.7333 | 0.8333 |
| Evidence-obligation recall | 0.7333 | 0.8333 |
| Evidence-ID precision | 0.4000 | 0.4433 |
| Structural citation integrity | 1.0000 | 1.0000 |
| Total tokens | 20,999 | 26,792 |
| Estimated API cost | $0.00336764 | $0.00459161 |
| Median traced latency per task | 5,401.5 ms | 7,317.0 ms |

The token-matched automatic gate passed exactly: B2 exceeded B1 evidence-obligation recall by `0.10`. B2 used 27.6% more tokens, 36.3% more estimated API cost, and 35.5% more median traced latency.

## Cost-matched

- Manifest: `experiments/pilot_v0/b1_b2v3_cost_matched.json`
- Batch: `20260812T102309Z`
- Cap: $0.002 observed provider cost per task

| Automatic metric | B1 | B2 v3 |
| --- | ---: | ---: |
| Completed | 10/10 | 10/10 |
| Evidence-ID recall | 0.8333 | 0.8667 |
| Evidence-obligation recall | 0.8333 | 0.8667 |
| Evidence-ID precision | 0.4167 | 0.4667 |
| Structural citation integrity | 1.0000 | 1.0000 |
| Total tokens | 21,511 | 25,037 |
| Estimated API cost | $0.00359578 | $0.00419066 |
| Median traced latency per task | 5,436.5 ms | 6,998.5 ms |

The cost-matched automatic gate failed: the recall difference was `+0.0333`, below the registered `+0.10`. The fee cap did not bind; the largest observed per-task fee was `$0.00042462` for B1 and `$0.00051004` for B2. B2 used 16.4% more tokens, 16.5% more estimated API cost, and 28.7% more median traced latency.

## Decision

The current evidence does not establish a stable B2 improvement across budget regimes. The query-balanced collector and Evidence Debt remain useful auditable mechanisms, but the semantic human gate remains `planned`. No further prompt tuning is selected from these automatic results alone.

Two blinded 20-candidate packets are ready:

- Token packet SHA-256: `bf3b1df4c65d6681c14fdbb7afeca22f9031d64e0443f6cb126f1dd9d44ead0c`
- Cost packet SHA-256: `7573797756a18bcb13000459528706f86aaeceb4e9286eee6c1d4ed2f1668f09`

Both packets were checked for variant-name and retrieval-query leakage. The next admissible evidence is completed blind human annotation, not another architecture feature.

## Chinese reviewer workspaces

On 2026-08-13, `deepseek-v4-flash` generated packet-bound Simplified Chinese reading aids for both blinded workspaces. These calls are reviewer support tooling and are excluded from both variants' experiment budgets and performance results.

| Workspace | Entries | Calls | Tokens | Estimated cost | Traced latency | Bundle SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Token-matched | 156 | 6 | 17,826 | $0.00373646 | 55,343 ms | `bb0ba0d9cbae5b887571d1538e228de538d70072d5a20b05bf0f7a6e86c79c80` |
| Cost-matched | 159 | 6 | 17,519 | $0.00365582 | 55,697 ms | `1b000f361687ae73171b7058b1a1fc4a521df2d5ec00b7b0194a5fe25aabe93e` |

Every translated entry contained Chinese text, and citation marker preservation checks reported zero mismatches. The workspaces retain a one-click English-original view; the English text controls any ambiguous judgment.
