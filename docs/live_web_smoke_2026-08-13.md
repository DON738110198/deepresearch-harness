# Live Web Smoke: 2026-08-13

## Problem and simplest test

The project had an auditable offline pipeline but no usable real-web loop. Human blind review was therefore premature: it asked a reviewer to score synthetic reports before the product could reliably collect current sources.

The selected minimum change was one single-Agent pass:

```text
Chinese question -> primary-source plan -> Search/Fetch -> claim ledger -> Chinese cited report
```

No subagents, Research DAG, Critic-Repair, or automatic re-planning were added. Repository queries use GitHub public repository search; general no-key search remains a best-effort fallback.

## Frozen run contract

- Model: `deepseek-v4-flash`, `thinking_mode=disabled`; model parameters were not changed.
- LLM calls: at most 3; output limit 1,200 tokens per call.
- Observed total-token guard: 8,000; estimated provider-fee guard: `$0.002` per task.
- Evidence cap for the final smoke: 3.
- Decision context: `docs/current_project_context.zh-CN.md`.
- Question: compare GPT Researcher, LangChain Open Deep Research, and DeerFlow workflows/evaluation, then choose one smallest next improvement for this project's single-Agent MVP.

## Saved bad cases

1. DuckDuckGo's non-Lite endpoint returned an anti-bot page and produced no evidence. Change: use DuckDuckGo Lite with Bing RSS fallback. This remains best-effort.
2. Bing split named projects into common words and returned unrelated pages. Change: repository-shaped queries first use GitHub public repository search.
3. A run completed with an empty report because a strict Evidence-Debt obligation combined workflow and evaluation into an all-or-nothing slot. Change: the user-facing live command uses the simpler B1 ledger; B2 remains an experiment variant.
4. A cited report treated observability as evaluation and recommended tracing that this project already had. Change: persist explicit decision context and define evaluation separately from observability.
5. `benchmark` was parsed as part of the repository name, and README boilerplate outranked the Evaluation section. Change: extract the entity name before the repository marker and rank document blocks by query-term rarity.

These failures are retained as diagnostic evidence. They do not establish that the fixes improve general Deep Research quality.

## Final smoke result

Run ID: `3799cac7102743b6b896429806cf4e98`.

- Status: succeeded.
- Search calls: 3, all through `github_repository_api`.
- Fetched sources: 3 official repositories: `assafelovic/gpt-researcher`, `langchain-ai/open_deep_research`, and `bytedance/deer-flow`.
- LLM usage: 7,716 input plus output tokens reported across 3 calls.
- Estimated provider cost: `$0.00082366` under the configured cache-aware price table.
- Report: Simplified Chinese with 4 deterministic claim/citation links and a generated source list.
- Audit checks: decision context persisted; run status succeeded; every citation marker appeared in the report; search/fetch backend, URL, latency, and outcome were present in the trace.

A focused inspection found that the final report distinguishes observability from evaluation, states missing GPT Researcher/DeerFlow evaluation evidence instead of filling the gap, and labels the proposed requery effect as planned. This is a smoke-level inspection, not registered human evaluation or a semantic metric.

## Current limits

- GitHub's unauthenticated search is rate-limited; DuckDuckGo Lite and Bing RSS are unofficial best-effort fallbacks.
- HTML/Markdown extraction is shallow. Dynamic pages, PDFs, authentication, paywalls, and anti-bot flows are not robustly handled.
- One 1,200-character excerpt per source can miss relevant sections.
- The pipeline performs one search round and cannot yet recover a missing answer axis.
- Citation markers and URLs are deterministic, but exact claim entailment and source quality are not automatically judged.
- No live benchmark slice, matched-budget comparison, or general quality metric has been run. All such results are **planned**.

## Next bad-case-driven gate

Implement exactly one bounded evidence-gap requery round inside the single-Agent pipeline.

Acceptance criteria:

1. A persisted gap record names the requested axis, existing evidence IDs, supplemental query, and stop reason.
2. At most one extra search round runs; it obeys the same model/tool/context contract and the existing Token/cost guards.
3. The saved comparison task retrieves a workflow/evaluation source only when the first pass lacks that axis; no subagent or DAG is introduced.
4. Deterministic tests cover `gap -> requery -> merged ledger` and `no gap -> no extra search`.
5. A fixed 5-10 task slice is registered before measuring effect. Compare against one-pass B1 with the same model and search implementation under separately token-matched and cost-matched budgets.
6. Report coverage, unsupported-claim rate, completion, tokens, fee, and latency are recorded. Until run, every effectiveness value remains **planned**.
