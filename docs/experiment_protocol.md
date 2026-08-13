# Experiment Protocol

## Claim boundary

The comparison target is harness-level end-to-end task performance. Keep model weights frozen and do not describe harness gains as a base-model capability gain.

## Required controls

For a baseline versus a harness variant, fix:

- model identifier, provider endpoint, decoding parameters, and system prompt policy;
- tool implementations, tool versions, corpus snapshot or web-search date window;
- question set, evidence eligibility rules, scoring rubric, and retry policy;
- decision context and other task metadata exposed to the runtime;
- concurrency, timeout policy, and random seed where supported.

Run each configuration under two separately reported budgets:

1. **Token-matched:** same maximum input plus output token budget per task.
2. **Cost-matched:** same maximum provider/tool monetary budget per task, with the price table version recorded.

For DeepSeek tracks, bind the API model alias to the documented served model
version and dated price table in
`benchmarks/browsecomp_plus_v0/deepseek_provider_snapshot.json`. A cost-matched
run is valid only while that snapshot's price window applies; after a provider
price or served-version change, refresh the snapshot and start a new comparison
block rather than mixing estimates across regimes.

The harness sends the remaining output allowance with `max_tokens` on every
provider request. This is not assumed to be an exact hard cap: a provider can
report more generated tokens than requested, especially when reasoning tokens
are involved. Total billed tokens and estimated fee therefore use
provider-returned usage and are enforced between calls. A response that crosses
the global cap is still billable. Matching the reference loop, a terminal answer
from that response remains scoreable while its overshoot is recorded; a response
that still asks for tools terminates with `status=budget_exhausted`. Reports must
preserve and quantify either case.

Do not substitute a stronger model, fresher search index, or different tool allowance in one condition. If any control differs, label the result as an ablation or exploratory result rather than a fair head-to-head comparison.

## Per-run artifacts

Persist the question ID, config digest, corpus/tool version, model and provider, run state, all trace events, total tokens, estimated cost, latency, selected evidence, ledger, report, and scoring output. Redact credentials and sensitive raw data before sharing.

## Evaluation status

The B0/B1 token-matched provider run and an AI-assisted blind calibration are recorded in `pilot_v0_calibration.md`. The registered human semantic evaluation remains **planned**. The calibration is evidence for rubric debugging and mechanism selection only.

As of 2026-08-13, full manual blind annotation is paused and is no longer an implementation gate. The prepared workspaces remain optional audit artifacts. Near-term comparisons use deterministic checks plus a fixed, versioned LLM Judge; only Judge disagreements, source-fetch failures, and selected bad cases are manually inspected. This changes the workflow, not the status of any earlier metric: unrun human results remain `planned`.

The live web run in `live_web_smoke_2026-08-13.md` is an exploratory end-to-end smoke, not a baseline comparison or benchmark result. Its observed Token/cost/latency values describe that run only. Report-quality, coverage, and factuality improvements remain **planned** until a fixed task slice and evaluation contract are run.

The five-task LiveDRBench preview run in `livedrbench_preview_v0_results.md` is a pinned compatibility pilot. `compatibility_exact_main_claim_v1` performs normalized exact matching on selected main-claim fields and checks the official outer shape/type contract. It does not perform the official LLM equivalence judgments, tolerances, or leaderboard aggregation. Therefore its scores must be labeled compatibility metrics, not official LiveDRBench results.

BrowseComp-Plus is the active primary benchmark target described in
`research_goal_and_benchmark_strategy.md`. Before its answers or relevance
labels are accessed, pin the benchmark artifacts and create a deterministic
development/holdout split from query IDs. Gold data belongs behind an
evaluator-only boundary. Freeze and hash predictions before holdout scoring.
Report the sealed-holdout result separately from the official full-set
leaderboard submission. Until the official tool/index/evaluator contract is
reproduced, every BrowseComp-Plus score and rank remains `planned`.

Before scoring, export frozen traces through the hash-verifying official-run
adapter. The exporter may map a terminal Pi answer to `completed` and an
unfinished tool loop to `incomplete`, but it must not rewrite response text,
drop failed queries, read gold, or call a judge.

The query-ID split is frozen at 175 development and 655 sealed-holdout
questions. Development-question extraction projects only the encrypted query
column and writes plaintext only below ignored `runs/`; it does not select or
decrypt answers, gold documents, negatives, or evidence documents. Predictions
and per-run hashes are frozen before a development-only scorer may project the
question, answer, and gold/evidence docids. It never persists gold document
text, URLs, or negatives. Holdout gold remains sealed.

The true strict five-query standard run used the documented `max_tokens` field,
generated exactly 50,000 output tokens in total, recorded zero overshoot, and
produced 0/5 schema-complete answers. Earlier runs using Pi's auto-selected
`max_completion_tokens` are protocol diagnostics only: observed provider output
could exceed the requested cap, so they are excluded from every matched-budget
comparison even when their answers look better.

The phase-adaptive BM25 candidate keeps the same 10,000-token allowance but
uses high thinking for at most 8,000 exploration tokens and disables thinking
for a fixed answer-compilation turn of at most 2,000. It reached 4/5 schema
completion; this is a harness reasoning policy ablation, not a model capability
claim. Its strict exact and retrieval diagnostics are explicitly non-official.

DeepSeek documents temperature and top-p as unsupported in thinking mode and
does not expose a seed in the pinned Chat Completions contract. Therefore a
single run is not a reliability estimate. Adapter v6 omits unsupported sampling
fields during thinking, pins non-thinking phases to `temperature=0`, and records
the effective request policy. Registered effectiveness comparisons use at
least three independent runs per development query and report mean, standard
deviation, and paired win/loss counts. The best replicate may not be selected
as the headline result.

Repeat manifests must normally be written before generation and must alternate
execution order (`baseline_first`, `candidate_first`, ...). Each pair must bind
the same query set, prompts, model, control policy, output allowance, adapter
version, and retriever identities. Aggregation requires unique run IDs and
hash-verifies every run, summary, diagnostic, prediction, and official-export
artifact. Report standard deviation across trial-level aggregates and describe
query-trial win/loss counts as repeated observations of the same fixed queries,
not as additional independent benchmark questions.

The first three-trial adapter-v6 automation is explicitly exploratory. Its
initial process stopped after trial 1 BM25 because the selected Python lacked
`duckdb`; development gold was then opened before the remaining frozen-policy
runs completed, and the experiment manifest was reconstructed during resume.
No generation prompt, adapter, model, budget, or retriever was changed after
that interruption, but the artifact is still labeled
`reconstructed_after_interruption`, not preregistered confirmatory evidence.
Future larger-slice runs must pass dependency preflight and persist the complete
manifest before the first provider call.

Retriever experiments first replay exactly the frozen agent queries and saved
BM25 rankings. Model, query strings, query count, top-k, corpus, and relevance
labels are held fixed while only the retriever changes. A live dense run is
allowed only after replay clears its preregistered recall gate. Dense and BM25
are separate retriever variants and must never be described as a same-retriever
comparison.

The standard-loop adapter pins Pi 0.84.1 but clears Pi's system prompt and all
ambient coding context. Pi is held fixed in harness ablations and is not an
innovation variable. Matching the reference loop, tool calls already emitted
by the terminal model response are executed before the next-turn budget check.
Search calls include every successful or failed invocation. Repeated-context
provider tokens are counted as reported, even when the same evidence appears
across several model turns.

The official evaluator contract is separately pinned in
`benchmarks/browsecomp_plus_v0/official_evaluator.json`. Its Qwen3-32B revision
and decoding settings are evaluation controls, not a replacement research
agent and not part of either candidate's generation budget.

If the judge is downloaded through a mirror, inference may begin only after
`official_judge_assets.json` verifies all required shard, config, index, and
tokenizer hashes against the pinned upstream revision. Record the verifier
output, upstream repository commit, evaluator script hash, `uv.lock` hash,
vLLM version, selected GPU IDs, and raw per-query judgments. A quantized judge,
different weights, altered evaluator script, or substituted API judge is an
exploratory evaluator ablation, not the official result.

Gold answers are fetched only inside the scorer after generation. Generation receives the public question and fetched web evidence, while saved public artifacts contain predictions, hashes, and aggregate counts rather than copied benchmark gold. Parser or post-generation failures must still load persisted `RunState` usage so paid calls are not reported as zero-cost failures.

The following metrics remain required for registered comparisons:

- citation support rate: claims whose cited evidence supports the claim;
- evidence recall / coverage against a curated reference set;
- report completeness and factuality under a fixed rubric;
- median and tail latency, total tokens, and cost per completed task;
- completion rate and recoverable-failure rate.

## Bad-case-driven iteration

Start by saving a concrete failure with its inputs and trace. State `problem -> evidence -> hypothesis -> change -> metric -> conclusion`. Add only the smallest mechanism that tests that hypothesis, then add a focused regression test. A more complex orchestration is not accepted merely because it produces a more fluent report.

## Pilot metric boundary

Pilot v0 separates deterministic metrics from semantic judgments. Evidence-ID recall, obligation-level retrieval recall, and citation referential integrity are computed from contracts. Obligation coverage, exact claim support, unsupported claims, and conflict handling require human annotation. Do not label structural citation integrity as factual correctness.

When comparing B0 and B1 reports, use the generated blind-review packet rather than run directories. Candidate order is deterministic from the registered seed but the mapping remains in a separate answer key until annotation is locked.

The scorer enforces a mechanical review lock before reading the answer key: all candidates must be present, every claim and cited claim must be classified exactly once, and all referenced obligation/claim IDs must exist in the packet. `ai_assisted` review is calibration-only and must not be reported as a human result.

The static reviewer workspace is generated from `review_packet.json` only and should be placed outside the directory containing `answer_key.json`. The exported submission must pass `validate-review`; record its SHA-256 before running `score-review`. Browser local storage and draft exports are convenience state, not accepted experiment artifacts until the Python validator succeeds.

## Translated reviewer view

The `zh-CN` workspace is a reading aid for the same blinded human review, not an AI-assisted annotation. Generate translations from `review_packet.json` only, bind the translation bundle to the packet SHA-256, and retain provider/model/token/cost/latency provenance separately from the evaluated harness runs. The renderer rejects a bundle with a different packet hash, a missing/extra source string, or changed citation markers, URLs, or claim/evidence identifiers.

The reviewer may toggle every translated task, report, evidence excerpt, and claim back to its English original. In any ambiguity, the English original controls. Translation calls must not be added to either variant's token-matched or cost-matched budget and must not be reported as research-agent cost or performance.
