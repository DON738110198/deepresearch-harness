# Training-Free Deep Research Harness

A small, auditable baseline for improving end-to-end Deep Research task execution **without changing model weights**. It calls an OpenAI-compatible chat-completions API and evaluates harness behavior, not intrinsic model capability.

> **Current direction (2026-08-13):** full manual blind annotation is paused
> because it imposed more reviewer work than the synthetic pilot could justify.
> The workspaces remain optional audit artifacts. BrowseComp-Plus plus its
> version-pinned official judge is now the primary experimental path; the live
> Chinese Search/Fetch flow remains a transfer test. See
> [`docs/market_research_and_pivot_2026-08.md`](docs/market_research_and_pivot_2026-08.md).

> **Active research target:** use frozen DeepSeek V4 Flash/Pro to test a coherent,
> failure-driven stack across retrieval, evidence control, phase-adaptive
> reasoning, and answer compilation. BrowseComp-Plus is the primary benchmark,
> with frozen leaderboard thresholds, leakage controls, and matched-budget gates.
> A strict five-query diagnostic now exists, but official accuracy and every
> rank claim remain `planned`. See
> [`docs/research_goal_and_benchmark_strategy.md`](docs/research_goal_and_benchmark_strategy.md)
> and [`docs/browsecomp_plus_layered_results_2026-08-13.zh-CN.md`](docs/browsecomp_plus_layered_results_2026-08-13.zh-CN.md).

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
prediction-bound development ground truth are staged. Judge inference is still
`planned_not_run` because no two 48 GiB GPUs are currently idle; no other
user's process was preempted.

Run an alternating-order, resumable three-trial comparison with:

```powershell
& .\scripts\run_browsecomp_plus_repeats.ps1 `
  -Trials 3 `
  -RunLabel <new-run-label>
```

The script writes the experiment manifest before generation, refuses partial
or mismatched artifacts on resume, exports every frozen answer to the official
input shape, and aggregates trial means, sample standard deviations, and paired
query outcomes. `-Resume` reuses only validated frozen summaries and never
silently restarts an incomplete variant.

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
- BrowseComp-Plus exact/recall values are development diagnostics. Official Qwen3-32B accuracy and every leaderboard claim remain `planned_not_run`.

## Repository map

- `src/deepresearch_harness/contracts.py`: Pydantic run-state contracts.
- `src/deepresearch_harness/providers.py`: fake and OpenAI-compatible providers.
- `src/deepresearch_harness/pipeline.py`: baseline orchestration and persistence.
- `src/deepresearch_harness/web_research.py`: no-key repository/web search, bounded fetch, text extraction, and request trace.
- `src/deepresearch_harness/public_benchmark.py`: pinned LiveDRBench loading, structured predictions, exact compatibility scoring, and batch audit summary.
- `src/deepresearch_harness/browsecomp_plus.py`: strict benchmark pins, leakage-safe query split/extraction, and Pi run contracts.
- `src/deepresearch_harness/bm25_server.py`: loopback-only pinned BM25/top-5/Qwen-tokenizer search service.
- `src/deepresearch_harness/dense_server.py`: loopback-only pinned dense/top-5 service backed by the same document store and snippet contract.
- `src/deepresearch_harness/retrieval_replay.py`: hash-bound dense and RRF counterfactual replay over frozen agent queries.
- `src/deepresearch_harness/browsecomp_evaluation.py`: post-prediction development-gold boundary and explicitly non-official diagnostics.
- `src/deepresearch_harness/browsecomp_repeats.py`: strict paired-repeat validation, artifact binding, distributions, and query-level win/loss aggregation.
- `src/deepresearch_harness/pi_browsecomp.py`: auditable smoke orchestration, aggregate usage trace, and hash-bound official-run export.
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
- `docs/b1_b2_token_v1.md`: retained negative B2 v1 result and causal follow-up.
- `docs/b1_b2_token_v2.md`: retained B2 v2 tie and search-layer diagnosis.
- `docs/b1_b2_v3_results.md`: token/cost automatic results and current claim boundary.
- `docs/live_web_smoke_2026-08-13.md`: real DeepSeek live-search failures, final smoke evidence, and next gate.
- `docs/livedrbench_preview_v0_results.md`: first external benchmark result and search-layer diagnosis.

## Next implementation order

1. **Completed:** real Search/Fetch -> Chinese cited report with query/URL/provider/token/cost/latency trace and environment-only credentials.
2. **Completed:** pinned five-task LiveDRBench preview baseline with structured predictions, exact compatibility metrics, and retained parser/cost-audit failures.
3. Add one stable general Search API adapter behind the existing collector interface and keep its key environment-only. Do not add re-planning while first-pass retrieval is still broken.
4. Freeze a separate public holdout slice, then compare the current no-key fallback with the stable search adapter under the same model, query policy, evidence cap, and fee ceiling.
5. Only after first-pass retrieval is credible, test one bounded evidence-gap requery round; add a DAG or subagents only if repeated independent-branch failures justify them.
