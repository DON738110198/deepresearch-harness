# Training-Free Deep Research Harness

A small, auditable baseline for improving end-to-end Deep Research task execution **without changing model weights**. It calls an OpenAI-compatible chat-completions API and evaluates harness behavior, not intrinsic model capability.

> **Current direction (2026-08-15):** full manual blind annotation is paused
> because it imposed more reviewer work than the synthetic pilot could justify.
> The workspaces remain optional audit artifacts. BrowseComp-Plus plus its
> version-pinned official judge is now the primary experimental path; the live
> Chinese Search/Fetch flow remains a transfer test. See
> [`docs/market_research_and_pivot_2026-08.md`](docs/market_research_and_pivot_2026-08.md).

> **Active research target:** use frozen DeepSeek V4 Flash/Pro to test a coherent,
> failure-driven stack across retrieval, evidence control, phase-adaptive
> reasoning, and answer compilation. BrowseComp-Plus is the primary benchmark,
> with frozen leaderboard thresholds, leakage controls, and matched-budget gates.
> A fresh paired-25 development run now exists for Query-Aware Progressive
> Disclosure. Its calibrated Judge and evidence-recall gates improved, but the
> registered overall decision is `reject` because the baseline missed the
> per-variant schema gate. The 175-query run, sealed holdout, and every rank
> claim remain `planned`. See
> [`docs/research_goal_and_benchmark_strategy.md`](docs/research_goal_and_benchmark_strategy.md)
> and [`docs/query_aware_progressive_disclosure_results_2026-08-15.zh-CN.md`](docs/query_aware_progressive_disclosure_results_2026-08-15.zh-CN.md).

The first MVP is deliberately narrow:

```text
Question -> Plan -> Search / collect evidence -> Claim-Evidence Ledger -> cited report
```

Every run persists its normalized input, plan, evidence, claims, citations, status transitions, and per-stage token/cost/latency trace. The default `fake` provider and fixture corpus are deterministic, so the project is runnable offline with no API key.

## Quick start

```powershell
cd G:\Obsidian\Study\AI_infra_interm\training-free-deepresearch-harness
python -m pip install -e ".[dev]"
python -m deepresearch_harness.cli demo --output-dir runs\smoke
python -m deepresearch_harness.cli validate-pilot --suite benchmarks\pilot_v0\tasks.json
python -m deepresearch_harness.cli validate-experiment --manifest experiments\pilot_v0\token_matched.json
python -m pytest
```

The demo writes `runs/smoke/<run-id>/state.json` and `report.md`. `state.json` is the audit artifact; it includes prompts only as stage metadata, not credentials.

## BrowseComp-Plus research path

The active benchmark path pins the repository, datasets, BM25 and dense index
candidates, Qwen snippet tokenizer, query prompt, empty system-prompt policy,
DeepSeek model tracks, and deterministic 20/80 query-ID split. Validate the
public contract:

```powershell
python -m pip install -e ".[dev,browsecomp-plus]"
python -m deepresearch_harness.cli validate-browsecomp-plus-target `
  --manifest benchmarks\browsecomp_plus_v0\target_manifest.json
```

The same validator binds the documented DeepSeek served versions and dated
price table. Cost-matched experiments must refresh that snapshot when provider
pricing changes.

`benchmarks/browsecomp_plus_v0/query_partitions.json` contains query IDs and
partition labels only. Plaintext benchmark questions and all model traces stay
under ignored `runs/`. The thin Pi adapter is in `integrations/pi-browsecomp`;
Pi provides the tool loop, not the project's research contribution. Adapter
v1+ forces DeepSeek's documented `max_tokens` field after an earlier protocol
diagnostic showed that Pi's auto-selected `max_completion_tokens` did not cap
observed reasoning. Those earlier traces are retained but excluded.

The strict five-query diagnostic found two promising harness effects. First,
high-thinking exploration plus non-thinking answer compilation made 4/5
outputs schema-complete under the same 10k output allowance. Second, three
adapter-v6 paired trials found that replacing BM25 with pinned
Qwen3-Embedding-0.6B raised mean diagnostic evidence recall from
17.36% +/- 2.88 to 60.00% +/- 14.84 and strict exact match from
6.67% +/- 11.55 to 46.67% +/- 11.55. Dense won 8/15 query-trial evidence-recall
pairs and lost 2, while using fewer mean search calls, tokens, dollars, and
latency. Dollar values cover the traced DeepSeek API only; local dense-index
hosting is not monetized, so this is neither a full-system cost comparison nor
a total-token-matched experiment. These are five-query development diagnostics,
not official accuracy.
The first automation attempt was interrupted after its first BM25 run, so the
repeat manifest is honestly marked `reconstructed_after_interruption`; future
confirmatory runs must be registered before generation. Only the pinned
Qwen3-32B judge can establish official accuracy.

The official-judge preflight is now reproducible on the configured A6000 host:
the exact upstream repository and lockfile environment are installed, and all
24 required Qwen3-32B files (17 weight shards plus model/tokenizer metadata)
matched the pinned revision byte for byte. The 30 frozen repeat inputs and
prediction-bound development ground truth were staged and evaluated. Across the
three five-query trials, the official evaluator scored BM25 at 20.00% +/- 0.00
and dense at 60.00% +/- 20.00; dense won/lost/tied 7/1/7 paired judgments.
All 30 raw judgments were retained with zero parse failures. This is an official
evaluator result on five fixed development questions, not a 30-question sample,
full-benchmark accuracy, submission, or leaderboard result. Runtime artifacts
remain below ignored `runs/`; the stable hashes are recorded in the experiment
log without publishing benchmark-derived payloads.

Run an alternating-order, resumable three-trial comparison with:

```powershell
& .\scripts\run_browsecomp_plus_repeats.ps1 `
  -Trials 3 `
  -RunLabel <new-run-label>
```

The script writes the experiment manifest before generation, refuses mismatched
artifacts on resume, exports every frozen answer to the official input shape,
and aggregates trial means, sample standard deviations, and paired query
outcomes. `-Resume` reuses successful queries and retries only records whose
latest status is `failed`. Before retrying, it hash-validates the live request,
run, prediction, controls, and any earlier attempt chain. Superseded attempts
and their source summary are archived; their Token, cost, search, and latency
remain in cumulative experiment totals rather than disappearing.
Use `-RegisterOnly` to freeze a larger-slice manifest without making a provider
call; later execution must pass `-Resume` with the same model, query artifact,
trial count, and run label or the script rejects the change.
New manifests use repeat-contract v1 and preregister a three-resume ceiling,
failed-only eligibility, immutable completed records, artifact preservation,
and cumulative usage accounting. The already-running 25-query v0 manifest is
not rewritten; its post-failure recovery remains explicitly qualified.
The next 25-query Flash comparison was registered this way before generation.
Its paid execution began only after the five-query official-Judge gate passed.
Five of six variants completed; the final dense variant retained 20 successful
queries and five `402 Insufficient Balance` failures. Across the partial run,
145/150 query executions succeeded and traced DeepSeek cost is USD 2.098859336.
No 25-query score is reported from this incomplete grid. After provider balance
is restored, the exact `-Resume` command retries only those five failures under
the frozen generation controls and then regenerates hash-bound diagnostics.
The resume can be checked first without an API key, retriever server, file
write, or provider call. This validates the frozen summary, every live and
archived attempt, request controls, query ordering, and retriever hashes:

```powershell
python -m deepresearch_harness.cli audit-pi-browsecomp-resume `
  --manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --partitions benchmarks\browsecomp_plus_v0\query_partitions.json `
  --queries runs\browsecomp_plus_v0\dev_queries_25.json `
  --output-dir runs\browsecomp_plus_v0\repeats\flash-v6-dense25-paired3-preregistered-v2-20260814\trial-03-candidate `
  --search-url http://127.0.0.1:8766/search `
  --model deepseek-v4-flash `
  --control-policy answer_reserve_nonthinking_v0 `
  --retriever-id qwen3-embedding-0.6b `
  --retriever-manifest benchmarks\browsecomp_plus_v0\retriever_candidates.json
```

The current audit returns 20 immutable queries and five retry-eligible queries,
with `provider_calls=0` and `gold_accessed=false`.

```powershell
& .\scripts\run_browsecomp_plus_repeats.ps1 `
  -Trials 3 `
  -RunLabel flash-v6-dense25-paired3-preregistered-v2-20260814 `
  -Queries runs\browsecomp_plus_v0\dev_queries_25.json `
  -Model deepseek-v4-flash `
  -Resume
```

After the ignored Java, Python, and retrieval artifacts are present,
`scripts/run_browsecomp_plus_smoke.ps1` starts and stops the loopback search
server around one traceable run. It never accepts an API key argument.

Dense replay holds all frozen agent search queries fixed before paying for an
end-to-end rerun:

```powershell
python -m pip install -e ".[dev,browsecomp-plus-dense]"
python -m deepresearch_harness.cli replay-browsecomp-plus-retrieval `
  --manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --retriever-manifest benchmarks\browsecomp_plus_v0\retriever_candidates.json `
  --source-dir runs\browsecomp_plus_v0\pi_flash_answer_reserve_nonthinking_v0_20260813 `
  --gold-slice runs\browsecomp_plus_v0\dev_gold_answer_reserve_nonthinking_v0_5.json `
  --candidate-id qwen3-embedding-0.6b `
  --model-dir runs\_external\models\Qwen3-Embedding-0.6B `
  --index-root runs\_external\browsecomp-plus-indexes-direct `
  --output runs\browsecomp_plus_v0\retrieval_replay.json
```

After predictions are frozen, convert them without reading gold into the exact
outer shape consumed by the official evaluator:

```powershell
python -m deepresearch_harness.cli export-pi-browsecomp-runs `
  --source-dir runs\browsecomp_plus_v0\pi_flash_dense_nonthinking_v1_20260813 `
  --output-dir runs\browsecomp_plus_v0\pi_flash_dense_nonthinking_v1_official_input_20260813
```

The export manifest binds every evaluator file to its source run and prediction
hash. `benchmarks/browsecomp_plus_v0/official_evaluator.json` pins the judge
weights and inference settings. The exporter does not run or substitute for the
official Qwen3-32B judge.

When the judge weights arrive through a mirror, verify that every inference
asset is byte-equivalent to the pinned Hugging Face revision before loading it:

```powershell
python scripts\verify_browsecomp_plus_judge_assets.py `
  --manifest benchmarks\browsecomp_plus_v0\official_judge_assets.json `
  --model-dir <Qwen3-32B-directory> `
  --output runs\browsecomp_plus_v0\judge_asset_verification.json
```

The verifier checks all 17 weight shards plus config, index, and tokenizer
files using the upstream LFS SHA-256 or Git blob SHA-1. A mirror label alone is
not accepted as evidence that the official judge revision was reproduced.

Build a self-contained official-judge batch only from the already validated
repeat experiment:

```powershell
python -m deepresearch_harness.cli prepare-browsecomp-plus-official-judge-batch `
  --repeat-experiment <repeat-experiment.json> `
  --repeat-comparison <repeat-comparison.json> `
  --target-manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --official-evaluator benchmarks\browsecomp_plus_v0\official_evaluator.json `
  --output-dir runs\browsecomp_plus_v0\official-judge-batch
```

`scripts/run_browsecomp_plus_official_judge.py` is a fail-closed launcher for
the upstream evaluator. Before creating an execution directory, it checks the
batch hashes, clean upstream commit, evaluator and lockfile hashes, verified
write-protected model files, runtime versions, and three consecutive idle
snapshots for two distinct GPUs. It never preempts a process or accepts an
occupied device. On success it freezes a pre-inference registration, logs, and
every evaluator output hash. `--disable-nccl-p2p` is an explicit, audited
transport-only workaround for hosts where NCCL peer initialization hangs; it
does not change model weights, predictions, evaluator code, or decoding values.
The matching aggregation command rejects missing,
extra, reparsed, or prediction-mismatched per-query results:

```powershell
python -m deepresearch_harness.cli aggregate-browsecomp-plus-official-judge `
  --batch-manifest <batch_manifest.json> `
  --execution-registration <execution_registration.json> `
  --execution-result <execution_result.json> `
  --output runs\browsecomp_plus_v0\official-judge-comparison.json
```

An official evaluator score on a development slice remains distinct from a
full 830-query leaderboard submission.

When only one 48 GB A6000 is available, the project has a deliberately
non-official screening path. It starts a persistent loopback vLLM server with
the pinned Qwen3-32B-AWQ snapshot, evaluates the existing 150-item calibration
batch concurrently, and then compares its labels with the retained BF16 labels:

```bash
python scripts/run_browsecomp_plus_screening_judge.py \
  --screening-manifest benchmarks/browsecomp_plus_v0/screening_judge_v0.json \
  --batch-manifest <batch_manifest.json> \
  --python <official-venv-python> \
  --model-dir <huggingface-cache>/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8 \
  --output-dir <new-screening-output> \
  --gpu-id 5

python -m deepresearch_harness.cli calibrate-browsecomp-plus-screening-judge \
  --screening-manifest benchmarks/browsecomp_plus_v0/screening_judge_v0.json \
  --screening-result <new-screening-output>/evaluation/screening_result.json \
  --official-comparison <official-comparison.json> \
  --output <screening-calibration.json>
```

The AWQ path is useful only if every registered calibration gate passes. Its
scores remain screening diagnostics; the two-GPU BF16 evaluator remains the
official contract.

An accepted persistent BF16 calibration can score a complete frozen repeat
grid without reloading the model between variants. The wrapper validates the
running service and all source hashes, then writes one result per variant and a
paired comparison explicitly labeled as a non-official development diagnostic:

```powershell
python scripts\run_browsecomp_plus_repeat_development_judge.py `
  --repeat-experiment <repeat_experiment.json> `
  --repeat-comparison <repeat_comparison.json> `
  --target-manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --judge-manifest benchmarks\browsecomp_plus_v0\persistent_bf16_judge_v0.json `
  --calibration-result <accepted_calibration.json> `
  --reference-screening-result <reference_service_result.json> `
  --official-comparison <official_reference_comparison.json> `
  --service-registration <service_registration.json> `
  --asset-verification <asset_verification.json> `
  --output-dir <new_repeat_judge_output> `
  --base-url http://127.0.0.1:18015/v1
```

`decide_browsecomp_plus_dense_confirmation.py` then revalidates that result and
applies the original retrieval, Judge, structure, failure, and cost gates. It
does not choose a favorable replicate and does not access the sealed holdout.

The subsequent Evidence Bandwidth experiment tested the failure layer exposed
by that rejection. Zero-provider-call depth probing found useful dense evidence
below rank 5, while three fixed top-5 selectors failed the +10 point gate. A
fresh pre-registered 25-query, three-trial run therefore compared BM25 top-5
with dense top-20 under a fixed 1792-token search-payload budget. Evidence recall
improved from 39.68% to 49.87%, but calibrated Judge accuracy fell from 34.67%
to 26.67%; search calls, total tokens, and provider cost also exceeded their
matched-budget gates. The candidate is `reject`, not a promoted result. See
`docs/evidence_bandwidth_confirmation_v0_results_2026-08-14.zh-CN.md`.
The registered zero-provider-call selectivity probe then showed a two-sided
failure: dense found relevant evidence for 100% of candidate improvements but
only 50% of candidate regressions, where BM25 found it for 100%. Repeated query
strings were absent, while repeated document slots remained high. This led to
BM25 anchors plus deduplicated Dense leads, explicit evidence opens, an
eight-search governor, and query-aware lead passages. On a fresh paired-25,
calibrated Judge accuracy moved from 16% to 20% and evidence recall from 36.03%
to 44.79%, while search/Token/provider-cost ratios were 0.488764/0.251863/
0.483486. The machine decision is still `reject` because baseline schema
completeness was 92% against the frozen 96% gate. See
`docs/query_aware_progressive_disclosure_results_2026-08-15.zh-CN.md`.

The hash-bound decision can be reproduced locally after the ignored run
artifacts are present:

```powershell
python scripts\decide_browsecomp_plus_evidence_bandwidth.py `
  --preregistration benchmarks\browsecomp_plus_v0\evidence_bandwidth_confirmation_v0.json `
  --candidate-manifest benchmarks\browsecomp_plus_v0\evidence_bandwidth_candidate_v0.json `
  --repeat-experiment <repeat_experiment.json> `
  --repeat-comparison <repeat_comparison.json> `
  --target-manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --judge-manifest benchmarks\browsecomp_plus_v0\persistent_bf16_judge_v0.json `
  --judge-calibration <accepted_calibration.json> `
  --judge-execution-registration <execution_registration.json> `
  --judge-execution-result <execution_result.json> `
  --judge-comparison <repeat_development_judge_comparison.json> `
  --output <evidence_bandwidth_decision.json>
```

Layer selection is also a tracked, machine-readable decision rather than a
post-hoc reading of the best metric. `promotion_gates.json` requires at least
three trials and 25 development questions, evidence-recall delta >= 10 points,
non-negative official-accuracy delta, pre-generation registration, and zero
judge parse failures. The decision command revalidates every upstream artifact,
reports resource deltas, and classifies candidate failures before naming the
next experiment:

```powershell
python -m deepresearch_harness.cli decide-browsecomp-plus-layer `
  --repeat-experiment <repeat_experiment.json> `
  --repeat-comparison <repeat_comparison.json> `
  --target-manifest benchmarks\browsecomp_plus_v0\target_manifest.json `
  --promotion-gates benchmarks\browsecomp_plus_v0\promotion_gates.json `
  --judge-batch-manifest <batch_manifest.json> `
  --judge-execution-registration <execution_registration.json> `
  --judge-execution-result <execution_result.json> `
  --judge-comparison <official-judge-comparison.json> `
  --output runs\browsecomp_plus_v0\layer-decision.json
```

Applied to the existing five-question slice, the mechanism gates pass
(+42.64 evidence-recall points and +40.00 official-accuracy points), but the
decision is correctly `insufficient_scope` because the query-count and clean
registration gates fail. It does not promote the layer or claim a benchmark
result.

The machine-readable file was formalized after the partial 25-query execution,
so it says so explicitly; its `+10 pp recall / no accuracy decline` thresholds
are provenance-bound to commit `dd25e78`, where they were documented before
the first 25-query provider call.

## Live Chinese research

The live command uses the same frozen-model pipeline with public-web Search/Fetch, a persisted decision context, and deterministic clickable citations:

```powershell
python -m deepresearch_harness.cli research `
  --question "请对比 GPT Researcher、LangChain Open Deep Research 和 DeerFlow，并给出一个最小改进建议。" `
  --context-file docs\current_project_context.zh-CN.md `
  --max-evidence 3 `
  --config config.local.json `
  --output-dir runs\live
```

Repository-shaped queries first use GitHub's public repository search and fetch the repository README. Other queries fall back to DuckDuckGo Lite and Bing RSS without another API key. These no-key search paths are best-effort and rate-limited, not a production search SLA. The saved trace records every query, backend, fetched URL, fetch error, token count, estimated fee, and latency.

## Public benchmark pilot

The first external slice pins five tasks from Microsoft's LiveDRBench preview. It runs the same one-pass B1 search policy with a task-shaped JSON answer adapter:

```powershell
python -m deepresearch_harness.cli validate-public-benchmark `
  --manifest benchmarks\livedrbench_preview_v0\manifest.json

python -m deepresearch_harness.cli run-public-benchmark `
  --manifest benchmarks\livedrbench_preview_v0\manifest.json `
  --config config.local.json `
  --output-dir runs\public_benchmarks
```

The deterministic compatibility scorer checks prediction coverage, official outer shape/type compatibility, and normalized exact match on designated main claims without another model call. It is not the official LiveDRBench evaluator and must not be reported as a leaderboard score. The recorded baseline completed 5/5 tasks but obtained `0.0` macro exact main-claim F1 because the no-key general search fallback returned mostly irrelevant sources. See `docs/livedrbench_preview_v0_results.md`.

## Real API run

Copy `config.example.json` to a local config file, set only the named environment variable, then run:

```powershell
$env:OPENAI_API_KEY = "..."
python -m deepresearch_harness.cli run `
  --variant b1 `
  --question "What evidence supports a phased rollout?" `
  --corpus examples/offline_corpus.json `
  --config config.local.json `
  --output-dir runs
```

Use `--variant b0` for direct Search-Write, `--variant b1` for Plan-Search-Ledger-Write, and `--variant b2` for obligation-linked Evidence Debt. All persist the same `RunState` contract and compile citation IDs and markers deterministically in the harness. B2 retains B1's three LLM calls while recording whether every planned answer obligation is resolved or open.

The frozen ten-task diagnostic is launched with:

```powershell
python -m deepresearch_harness.cli run-experiment `
  --manifest experiments\pilot_v0\token_matched.json `
  --config config.local.json `
  --output-dir runs\experiments
```

Experiment output is ignored by Git and includes one state/report/automatic-score bundle per task and variant plus `summary.json`. Automatic scores cover retrieval and citation structure only; semantic support still requires human annotation.

Create a variant-blinded semantic review packet from a completed batch with:

```powershell
python -m deepresearch_harness.cli prepare-review `
  --summary runs\experiments\<experiment>\<timestamp>\summary.json `
  --suite benchmarks\pilot_v0\tasks.json `
  --output-dir runs\reviews\<review-id>
```

The reviewer packet contains candidates `A/B` without variant labels or model-generated retrieval queries. Keep the separately generated `answer_key.json` hidden until annotation is complete. The annotation file is rejected unless every candidate, claim, cited claim, obligation, and conflict label is internally complete and references known IDs.

Build a standalone review workspace in a directory that does not contain the answer key:

```powershell
python -m deepresearch_harness.cli render-review `
  --packet runs\reviews\<review-id>\review_packet.json `
  --output runs\reviewer_workspaces\<review-id>\index.html
```

Open `index.html` directly. The workspace shows candidates `A/B`, saves progress in browser local storage, imports/exports JSON drafts, and enables final export only after every claim and citation is classified. It embeds the blinded packet but never reads the answer key.

For a Chinese reading-aid workspace, first translate only the blinded packet, then render it with the validated bundle:

```powershell
python -m deepresearch_harness.cli translate-review `
  --packet runs\reviews\<review-id>\review_packet.json `
  --config config.local.json `
  --output runs\reviewer_workspaces\<review-id>\translations.zh-CN.json

python -m deepresearch_harness.cli render-review `
  --packet runs\reviews\<review-id>\review_packet.json `
  --locale zh-CN `
  --translations runs\reviewer_workspaces\<review-id>\translations.zh-CN.json `
  --output runs\reviewer_workspaces\<review-id>\index.zh-CN.html
```

The Chinese view keeps a one-click English-original toggle. Translation provenance, usage, cost, latency, and the source packet hash are stored in the bundle. Translation is a reading aid, not a second judge or an experiment metric; when wording is ambiguous, score against the English original.

Before unblinding, lock the exported submission and record its hash:

```powershell
python -m deepresearch_harness.cli validate-review `
  --packet runs\reviews\<review-id>\review_packet.json `
  --annotations <exported-annotations.json>
```

This command validates all 20 candidates and prints `reviewer_type` plus the annotation SHA-256 without opening `answer_key.json`.

After annotation is locked, validate it and aggregate semantic metrics with:

```powershell
python -m deepresearch_harness.cli score-review `
  --packet runs\reviews\<review-id>\review_packet.json `
  --annotations runs\reviews\<review-id>\annotations.json `
  --answer-key runs\reviews\<review-id>\answer_key.json `
  --output runs\reviews\<review-id>\scores.semantic.json
```

Set `reviewer_type` to `human` for the registered human evaluation. An `ai_assisted` submission is explicitly emitted as `calibration_only`; it is useful for rubric debugging and bad-case selection, but is not a human metric.

No API key is accepted through CLI flags or configuration values. `api_key_env` names the environment variable to read at runtime.

## Scope and limits

- This is a single-agent Plan-Search-Write baseline, not a multi-agent research system.
- B0, B1, and B2 are harness variants, not changes to model capability.
- Offline tests use a supplied JSON evidence corpus; the live collector adds bounded web search and readable-page extraction but does not claim comprehensive coverage or freshness.
- The fake provider is for deterministic pipeline tests, not quality evaluation.
- Report citations point to collected evidence IDs and deterministic source links; source quality and claim entailment still require evaluation.
- The recorded B0/B1 semantic pass is explicitly AI-assisted calibration; registered human semantic metrics remain **planned**.
- GitHub repository search is public and unauthenticated; DuckDuckGo Lite/Bing RSS are unofficial best-effort fallbacks. Dynamic pages, PDFs, and anti-bot pages are not handled reliably.
- The LiveDRBench preview score is a five-task compatibility diagnostic. The official judge was not run, and exact normalized matching is deliberately stricter and narrower than the official evaluator.
- BrowseComp-Plus exact/recall values are development diagnostics. Fresh 25-query three-trial retrieval experiments and a calibrated persistent Qwen3-32B Judge have run, but this service is not the upstream official evaluator. Sealed-holdout accuracy, submission, and every leaderboard claim remain `planned`.

## Repository map

- `src/deepresearch_harness/contracts.py`: Pydantic run-state contracts.
- `src/deepresearch_harness/providers.py`: fake and OpenAI-compatible providers.
- `src/deepresearch_harness/pipeline.py`: baseline orchestration and persistence.
- `src/deepresearch_harness/web_research.py`: no-key repository/web search, bounded fetch, text extraction, and request trace.
- `src/deepresearch_harness/public_benchmark.py`: pinned LiveDRBench loading, structured predictions, exact compatibility scoring, and batch audit summary.
- `src/deepresearch_harness/browsecomp_plus.py`: strict benchmark pins, leakage-safe query split/extraction, and Pi run contracts.
- `src/deepresearch_harness/bm25_server.py`: loopback-only pinned BM25/top-5/Qwen-tokenizer search service.
- `src/deepresearch_harness/dense_server.py`: loopback-only pinned dense/top-5 service backed by the same document store and snippet contract.
- `src/deepresearch_harness/dense_depth_probe.py`: zero-provider-call rank-depth localization over saved agent queries.
- `src/deepresearch_harness/wide_selector_probe.py`: fixed top-5 selector ablations over the saved dense top-20 pool.
- `src/deepresearch_harness/evidence_bandwidth.py`: deterministic fixed-budget snippet allocation for widened retrieval.
- `src/deepresearch_harness/evidence_bandwidth_server.py`: loopback-only dense top-20 service with aggregate snippet-token accounting.
- `src/deepresearch_harness/evidence_bandwidth_decision.py`: hash-bound Evidence Bandwidth gates and synthesis-loss routing.
- `src/deepresearch_harness/evidence_selectivity_probe.py`: zero-provider-call paired trace analysis for evidence discovery, duplication, and synthesis routing.
- `src/deepresearch_harness/progressive_disclosure.py`: deterministic anchor/lead/open evidence contracts and ingress budgets.
- `src/deepresearch_harness/progressive_disclosure_server.py`: run-scoped dual-channel retrieval and evidence-open service.
- `src/deepresearch_harness/evidence_preview.py`: query-aware, bounded Dense-lead passage selection.
- `src/deepresearch_harness/tool_loop_governor_decision.py`: fresh-five governor/preview gates and routing.
- `src/deepresearch_harness/query_aware_confirmation_decision.py`: hash-bound paired-25 confirmation and per-query trace decision.
- `src/deepresearch_harness/retrieval_replay.py`: hash-bound dense and RRF counterfactual replay over frozen agent queries.
- `src/deepresearch_harness/multi_query_rrf.py`: two-phase, gold-blind multi-query RRF slate construction and post-persistence scoring.
- `src/deepresearch_harness/browsecomp_evaluation.py`: post-prediction development-gold boundary and explicitly non-official diagnostics.
- `src/deepresearch_harness/browsecomp_repeats.py`: strict paired-repeat validation, artifact binding, distributions, and query-level win/loss aggregation.
- `src/deepresearch_harness/browsecomp_judge.py`: self-contained official-judge batches, execution contracts, result validation, and paired score aggregation.
- `src/deepresearch_harness/browsecomp_decision.py`: frozen promotion gates, resource deltas, and official-judge bad-case routing.
- `src/deepresearch_harness/pi_browsecomp.py`: auditable smoke orchestration, read-only resume audit, aggregate usage trace, and hash-bound official-run export.
- `src/deepresearch_harness/research_loop.py`: one-variable checkpoints, framework comparisons, pause audits, and hash-bound failure-cluster stop routes.
- `integrations/pi-browsecomp/`: pinned Pi/DeepSeek tool-loop adapter with no coding-agent prompt or ambient context.
- `src/deepresearch_harness/benchmark.py`: pilot contracts, asset validation, and scoring boundaries.
- `src/deepresearch_harness/batch.py`: frozen two-variant batch execution and automatic aggregation.
- `src/deepresearch_harness/review.py`: deterministic variant-blind review, validation, and aggregation.
- `src/deepresearch_harness/review_translation.py`: packet-bound Chinese reading-aid translations with provider trace.
- `src/deepresearch_harness/review_workspace.py`: standalone browser workspace for blind human annotation.
- `experiments/pilot_v0/`: token-matched and cost-matched manifests with pinned corpus hashes and pricing.
- `benchmarks/pilot_v0/`: ten controlled diagnostic tasks and a synthetic corpus.
- `benchmarks/livedrbench_preview_v0/`: frozen five-task external compatibility manifest.
- `docs/problem_statement.md`: problem-first design position and falsifiable hypotheses.
- `docs/pilot_design.md`: B0/B1/B2 comparison and stage gates.
- `docs/architecture.md`: current boundaries and next-stage extension points.
- `docs/experiment_protocol.md`: fair-comparison and bad-case evaluation protocol.
- `docs/browsecomp_plus_layered_results_2026-08-13.zh-CN.md`: Chinese experiment results, failed controls, hashes, and next gates.
- `docs/query_aware_progressive_disclosure_results_2026-08-15.zh-CN.md`: Progressive Disclosure, governor, preview, paired-25 decision, and bad-case route.
- `docs/continuous_research_loop.zh-CN.md`: controlled failure-driven loop, anti-thrashing boundary, and multi-Agent entry gate.
- `docs/evidence_debt_loop_results_2026-08-16.zh-CN.md`: execution audit, rejected repair branches, span/target calibration, overlay negatives, and retrieval-rank diagnosis.
- `docs/passage_index_representation_results_2026-08-16.zh-CN.md`: preregistered full-corpus passage-index build, transport recovery, negative Recall@20 gate, and frozen next boundary.
- `docs/persistent_miss_dense_rank_results_2026-08-16.zh-CN.md`: CPU-only dense top-20/100/1000 rank audit, retained local crash, negative gates, and the next query-bridge boundary.
- `docs/visible_pivot_bridge_results_2026-08-16.zh-CN.md`: gold-aware one-hop bridge sufficiency, exact 4/7 gate, metadata-artifact caveat, and the gold-blind selector boundary.
- `docs/gold_blind_visible_pivot_slate_results_2026-08-16.zh-CN.md`: no-gold two-pivot slate, persisted selection boundary, 0/7 negative result, and frozen rarity-first selector.
- `docs/multi_query_rrf_results_2026-08-16.zh-CN.md`: preregistered zero-provider rank fusion, 0/7 top-20/top-100 negative result, and the corrected dense-document visibility diagnostic.
- `docs/dense_document_visibility_results_2026-08-16.zh-CN.md`: source-verified 512-query/4096-document contract, prebuilt-index provenance gap, 7/7 visibility result, and the next raw-question dense-rank gate.
- `docs/raw_question_dense_rank_results_2026-08-17.zh-CN.md`: two-phase CPU-only raw-question rank audit, failed 4/7 gates, and the frozen raw-anchor boundary.
- `docs/full_development_profile_results_2026-08-17.zh-CN.md`: failed-only-resumed 175-question profile, calibrated Judge diagnostics, preregistered failure distribution, and the downstream-evidence route.
- `docs/evidence_reachability_funnel_results_2026-08-17.zh-CN.md`: zero-call arrival-to-synthesis funnel, its 47/67 hidden-answer result, and the retained gold-versus-evidence reference-policy confound.
- `docs/b1_b2_token_v1.md`: retained negative B2 v1 result and causal follow-up.
- `docs/b1_b2_token_v2.md`: retained B2 v2 tie and search-layer diagnosis.
- `docs/b1_b2_v3_results.md`: token/cost automatic results and current claim boundary.
- `docs/live_web_smoke_2026-08-13.md`: real DeepSeek live-search failures, final smoke evidence, and next gate.
- `docs/livedrbench_preview_v0_results.md`: first external benchmark result and search-layer diagnosis.

## Next implementation order

1. **Completed:** real Search/Fetch -> Chinese cited report with query/URL/provider/token/cost/latency trace and environment-only credentials.
2. **Completed:** pinned five-task LiveDRBench preview baseline with structured predictions, exact compatibility metrics, and retained parser/cost-audit failures.
3. **Completed:** bounded BM25-anchor/Dense-lead Progressive Disclosure, eight-search governor, query-aware preview calibration, and one fresh paired-25 development confirmation.
4. **Completed with retained negatives:** structured Evidence Debt, selective repair, span opening, and fail-closed execution loops.
5. **Completed with retained negatives:** full-corpus passage BM25 and fixed-query dense rank gates; neither cleared 4/7, so candidate-depth and reranker tuning are frozen on this cluster.
6. **Completed with a strict claim boundary:** zero-provider Visible-Pivot sufficiency reached 4/7, but one rescue was frontmatter metadata; it is an oracle, not a selector result.
7. **Completed with a retained negative:** the gold-blind two-pivot slate had zero leakage but recovered 0/7; freeze rarity-first token selection.
8. **Completed with a retained negative:** uniform RRF over all 48 frozen BM25 top-1000 lists recovered 0/7 at fused top-20 and top-100, versus 0/7 and 2/7 for the best-single-query baselines. Freeze fusion tuning. Source verification corrected an earlier assumption: the official BrowseComp-Plus recipe uses 512 tokens for queries but 4096 for documents. The next zero-provider diagnostic measures answer visibility under that 4096-token recipe and records that the downloaded vector shards do not bind their historical preprocessing metadata. Keep B3/B4 multi-Agent, fresh/official execution, sealed holdout, and leaderboard submission `planned` until their entry conditions are met.
9. **Completed diagnostic:** under the official 4096-token reproduction recipe, at least one gold document retained the literal answer for 7/7 persistent misses and 17/18 gold documents overall. Reject head truncation as the dominant cause and keep passage-dense frozen. The next bounded test is raw-question dense rank versus the already frozen generated-query ranks; this is still a posthoc retrieval diagnosis, not an effectiveness result.
10. **Completed with a retained negative:** raw full-question dense search reached 1/7 at top-20, 1/7 at top-100, and 2/7 at top-1000, versus 0/7, 1/7, and 5/7 for the frozen generated-query baseline. Raw question won only 1/7 per-case rank comparisons, so all three registered 4/7 gates failed. Six earlier prompt-only bridge/hypothesis variants on another saved cluster were also rejected. Freeze this seven-case cluster for regression replay instead of renaming another micro-intervention; the next registered step profiles frozen v10 over all 175 development questions. Multi-Agent, sealed holdout, and leaderboard claims remain `planned`.
11. **Completed development profile:** the frozen v10 policy produced 175/175 valid predictions after preserving 106 successes and retrying only 69 balance-failed IDs. Normalized exact was 42/175, evidence recall 52.51%, and the calibrated persistent Judge marked 65/175 correct. Of 110 Judge-wrong cases, 67 had some reference-document retrieval signal, 40 had none, and 3 failed the answer contract. The preregistered 60% route therefore selects a zero-provider evidence-reachability funnel before any new mechanism. These are development diagnostics, not official accuracy or model-capability improvement.
12. **Completed with a retained diagnostic limitation:** the registered reachability funnel found full literal answer coverage hidden in 47/67 downstream cases, while 11/67 were visible-but-uncited and 9/67 were cited-but-wrong. The 70.15% hidden share triggers the frozen exposure/opening route, but the union-of-gold-and-evidence reference policy can admit supporting-evidence-only arrival. Reject the route as a sufficient runtime-mechanism selector and formalize a zero-call mutually exclusive correction before changing the Agent.
