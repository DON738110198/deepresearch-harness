# Training-Free Deep Research Harness

A small, auditable baseline for improving end-to-end Deep Research task execution **without changing model weights**. It calls an OpenAI-compatible chat-completions API and evaluates harness behavior, not intrinsic model capability.

> **Current direction (2026-08-13):** full manual blind annotation is paused because it imposed more reviewer work than the current synthetic pilot could justify. The workspaces remain optional audit artifacts, not the next-stage gate. The project is moving first toward a usable live Search/Fetch -> Chinese cited report loop, followed by deterministic checks and a versioned LLM Judge. See [`docs/market_research_and_pivot_2026-08.md`](docs/market_research_and_pivot_2026-08.md).

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

## Repository map

- `src/deepresearch_harness/contracts.py`: Pydantic run-state contracts.
- `src/deepresearch_harness/providers.py`: fake and OpenAI-compatible providers.
- `src/deepresearch_harness/pipeline.py`: baseline orchestration and persistence.
- `src/deepresearch_harness/web_research.py`: no-key repository/web search, bounded fetch, text extraction, and request trace.
- `src/deepresearch_harness/benchmark.py`: pilot contracts, asset validation, and scoring boundaries.
- `src/deepresearch_harness/batch.py`: frozen two-variant batch execution and automatic aggregation.
- `src/deepresearch_harness/review.py`: deterministic variant-blind review, validation, and aggregation.
- `src/deepresearch_harness/review_translation.py`: packet-bound Chinese reading-aid translations with provider trace.
- `src/deepresearch_harness/review_workspace.py`: standalone browser workspace for blind human annotation.
- `experiments/pilot_v0/`: token-matched and cost-matched manifests with pinned corpus hashes and pricing.
- `benchmarks/pilot_v0/`: ten controlled diagnostic tasks and a synthetic corpus.
- `docs/problem_statement.md`: problem-first design position and falsifiable hypotheses.
- `docs/pilot_design.md`: B0/B1/B2 comparison and stage gates.
- `docs/architecture.md`: current boundaries and next-stage extension points.
- `docs/experiment_protocol.md`: fair-comparison and bad-case evaluation protocol.
- `docs/b1_b2_token_v1.md`: retained negative B2 v1 result and causal follow-up.
- `docs/b1_b2_token_v2.md`: retained B2 v2 tie and search-layer diagnosis.
- `docs/b1_b2_v3_results.md`: token/cost automatic results and current claim boundary.
- `docs/live_web_smoke_2026-08-13.md`: real DeepSeek live-search failures, final smoke evidence, and next gate.

## Next implementation order

1. **Completed:** real Search/Fetch -> Chinese cited report with query/URL/provider/token/cost/latency trace and environment-only credentials.
2. Implement one bounded single-Agent evidence-gap requery round, driven by the saved missing-workflow/evaluation bad case; do not add a DAG or subagents.
3. Add a simple plan-confirmation and report view that exposes research progress without evaluation-internal IDs.
4. Add deterministic citation/link checks plus a versioned rubric Judge that returns `pass/fail`, reason, and report evidence; full manual annotation is not required.
5. Run a 5-10 task public benchmark slice under fixed model/tool and separately token- or cost-matched budgets before claiming an effect.
