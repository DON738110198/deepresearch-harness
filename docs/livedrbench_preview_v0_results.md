# LiveDRBench Preview v0 Compatibility Result

## Why this benchmark

LiveDRBench models Deep Research as structured claim discovery rather than only long-form writing. Each task has a public question plus structured reference claims, and the official project reports precision, recall, and F1. That makes it a closer first external test for this harness's Claim-Evidence Ledger than a benchmark that scores only prose style.

- Official repository: https://github.com/microsoft/LiveDRBench
- Repository commit inspected: `6ff85b67b35fa303907f6f275417622338acd1f6`
- Dataset: https://huggingface.co/datasets/microsoft/LiveDRBench
- Dataset revision: `cad63dd0403b11e8a80001e2339f64ac93acec5a`
- Dataset license: CDLA-Permissive-2.0; repository code license: MIT.

## Claim boundary

This is a five-task `compatibility pilot`, not an official LiveDRBench or leaderboard score. The official evaluator uses model-based equivalence and task-specific tolerances. This project ran only:

1. deterministic prediction coverage;
2. official outer length/type compatibility;
3. normalized exact matching on designated main-claim fields.

The official evaluator status is `planned_not_run`. Exact matching is narrower and may reject semantically equivalent strings, but a zero result still exposes whether the current retrieval path finds the named targets at all.

## Frozen contract

- Manifest: `benchmarks/livedrbench_preview_v0/manifest.json`.
- Fixed keys selected before generation: `4, 31, 40, 55, 76`.
- Categories: entities, dataset discovery, flight incidents, geospatial paper discovery, and materials paper discovery.
- Model: `deepseek-v4-flash`, thinking disabled, frozen parameters.
- Variant: one-pass `b1_benchmark_structured`; Plan -> Search/Fetch -> Claim Ledger -> task-shaped JSON.
- Per-task guards: 3 model calls, 8,000 observed total tokens, `$0.002` estimated provider fee, 6 evidence items.
- Search: the existing GitHub repository search plus DuckDuckGo Lite/Bing RSS fallback. Every general query used Bing RSS in the final run.

## Final run

Run directory: `runs/public_benchmarks/livedrbench-preview-v0/20260813T074725Z`.

| Metric | Observed |
| --- | ---: |
| Completed tasks | 5/5 |
| Structured answer present | 5/5 |
| Official shape compatible | 4/5 |
| Macro exact main-claim precision | 0.0000 |
| Macro exact main-claim recall | 0.0000 |
| Macro exact main-claim F1 | 0.0000 |
| Total tokens | 19,324 |
| Estimated provider cost | `$0.00233460` |
| End-to-end wall time | 134.05 s |

Per-task exact matches were `0` for every task. Four tasks returned empty supported subsets. The materials task identified generic ZnO properties from Wikipedia and emitted two main claims, but neither matched the target paper title/material representation under exact scoring.

## Failure diagnosis

The plans were substantively aligned with the questions: they searched IMO Korea results, scene-level video datasets, repeated airline go-arounds, a five-dataset paper combination, and ZnO measured properties. The dominant failure occurred after planning:

- All general searches fell back to Bing RSS.
- `IMO` resolved to a messaging app or the word `international`; Korean language/truck pages filled the evidence set.
- Dataset discovery returned YouTube, Public.com, generic Kaggle, and punctuation pages.
- The flight query returned anime, games, hotels, and travel pages.
- The geospatial query resolved `Harvey` and `paper` as unrelated products.
- Only the ZnO task retrieved partially relevant evidence, but not the target paper.

The ledger produced zero claims for the four fully irrelevant evidence sets rather than inventing answers. This is desirable refusal behavior, but the end-to-end benchmark result remains zero.

## Retained implementation failures

Two earlier diagnostic runs are retained under the same ignored run root:

| Run | Completed | State-derived tokens | State-derived cost | Failure |
| --- | ---: | ---: | ---: | --- |
| `20260813T074130Z` | 3/5 | 12,831 | `$0.00187166` | Plan parser required a narrative `steps` field. |
| `20260813T074421Z` | 4/5 | 16,291 | `$0.00220320` | Empty relevant evidence led the ledger response to omit `claims`. |

Their original summaries undercounted failed-task usage. The runner now reloads persisted failed `state.json` files so paid plan/ledger calls remain in aggregate cost. Across the two diagnostics and final run, state-derived development usage was 48,446 tokens and `$0.00640946`.

## Decision

Do not add re-planning yet. Repeating queries against the same weak fallback is unlikely to repair entity disambiguation and would add cost.

The next minimal experiment is a stable general Search API adapter behind the existing `EvidenceCollector` boundary. Freeze a separate holdout before implementation, then compare search providers with the model, query policy, evidence cap, and per-task budget fixed. Acceptance requires improved first-pass main-claim recall without a material increase in irrelevant evidence or failed fetches. Any effectiveness numbers remain **planned** until that controlled run.
