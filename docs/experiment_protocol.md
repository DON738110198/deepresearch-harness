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

For paid larger-slice work, `run_browsecomp_plus_repeats.ps1 -RegisterOnly`
must be run first. This mode writes the complete experiment grid and exits with
`generation_started=false`; the later `-Resume` path compares the requested
configuration to that frozen manifest before any provider call.
The registered manifest includes the normalized SHA-256 of the complete
question artifact. A path-only registration is invalid even when no gold is
present, because an in-place question-file change would otherwise evade the
resume comparison.

Provider-failure recovery is status-selective. Only an item whose latest
status is `failed` may be called again; a succeeded or budget-exhausted item is
never regenerated. Resume must preserve the manifest, model, prompt/control
policy, retriever, per-attempt 10k output allowance, and question hash. Before
the new call, the previous request and run/error are hash-validated and
archived with the exact source summary. All attempt Token, API cost, searches,
compiler calls, and latency remain in the cumulative summary and therefore in
matched-budget reporting. The official evaluator receives only the latest
successful prediction. Any retry rule introduced after an observed provider
failure must be disclosed as an operational amendment; it cannot upgrade the
experiment to stronger confirmatory status.
All newly registered repeat grids use contract v1, which freezes this recovery
policy before generation and caps a variant at three resume invocations. A v0
manifest is never edited in place to appear preregistered after a failure.
Before a paid resume, run `audit-pi-browsecomp-resume` with the exact registered
model, control policy, retriever identity, query artifact, search URL, and output
directory. The audit must pass with `provider_calls=0`, `gold_accessed=false`,
and a retry-eligible count equal to the summary's failed count. Its source
summary hash must remain unchanged. This is readiness evidence only, not an
effectiveness result and not extra budget.

Layer promotion uses
`benchmarks/browsecomp_plus_v0/promotion_gates.json`, not an informal reading of
the result table. A candidate must have at least three trials and 25 fixed
development questions, `pre_generation` registration, zero official-Judge
parse failures, evidence-recall improvement of at least 10 percentage points,
and no decrease in mean official accuracy. All gates are conjunctive. A
five-question run may validate the mechanism and tooling but returns
`insufficient_scope`. Passing promotes a harness component into the next frozen
development experiment only; it is not a sealed-holdout, model-capability, or
leaderboard claim.
The JSON gate was formalized after the current partial run, but its two effect
thresholds are bound to pre-run commit `dd25e78`; this provenance distinction
must remain visible in the decision artifact and report.

After judging, incorrect candidate observations are assigned one mutually
exclusive diagnostic class: malformed answer contract, zero gold/evidence
document recall, or relevant-document recall with an incorrect answer. The
next intervention follows the dominant reproducible class: retrieval misses
justify query/retriever replay, while evidence-present failures justify an
answer-control experiment. No new prompt layer is paid for until this report is
complete.

Retriever experiments first replay exactly the frozen agent queries and saved
BM25 rankings. Model, query strings, query count, top-k, corpus, and relevance
labels are held fixed while only the retriever changes. A live dense run is
allowed only after replay clears its preregistered recall gate. Dense and BM25
are separate retriever variants and must never be described as a same-retriever
comparison.

The passage retrieval-representation gate operated offline on the complete
corpus after freezing the corpus hash, tokenizer, passage length, overlap,
passage-to-document mapping, and maximum returned documents. It preserved full
source and development-gold index coverage but reached only 2/7 diagnosed cases
at collapsed document Recall@20 against the registered 4/7 gate. The passage
branch is therefore frozen rather than retuned on those ranks. The next planned
diagnostic held the 48 recorded queries fixed and measured dense gold-document
rank without provider, online-search, or Judge calls. Dense top-20 was 0/7 and
top-100 was 1/7 against two registered 4/7 gates; top-1000 was 5/7 but remained
diagnostic only. Candidate-depth and reranker work are therefore frozen on this
cluster. The next planned oracle may test one lexical pivot only when that term
is already present in a visible non-gold document, is also present in a gold
document, and is absent from the question, recorded queries, and answer. Only a
passing gold-blind retrieval screen may permit a new, preregistered fresh
development comparison with the model, prompt, search provider, query-call cap,
Token budget, cost budget, and Judge held fixed. All unrun accuracy, citation,
latency, Token, and cost fields remain `planned`.

The registered Visible-Pivot oracle passed its lexical existence gate exactly
4/7 after 1,088 offline BM25 queries. That result is not a selector metric: it
used gold-document overlap to choose terms, and one successful term came only
from frontmatter coordinate-format metadata. Any follow-up selector must strip
wrappers and frontmatter before candidate extraction, receive neither gold
documents nor answers, preserve the candidate snippet and source-query
provenance, and freeze both slate size and actual pivot-search calls.

That gold-blind selector was then run with two body candidates per case. The
slate was persisted before the evaluator opened gold; exact-answer, gold-docid,
and frontmatter-only leakage were all zero. Nevertheless, its 14 fixed pivot
queries recovered 0/7 cases and retained 0/4 oracle rescues. This registered
negative freezes rarity-first ordering and slate expansion on the observed
cluster. A future candidate-representation experiment must keep the retriever,
Agent count, and query budget fixed, and all fresh/official metrics remain
`planned` until it passes.

Uniform multi-query RRF was then registered as the no-model aggregation lower
bound. Its builder persisted all 48 frozen BM25 top-1000 rankings and fused
top-100 slates before the scorer opened gold. With `k=60`, fused Recall@20 and
Recall@100 were both 0/7, versus best-single-query 0/7 and 2/7. The registered
4/7 gates failed, so fusion weights, `k`, tie breaks, and a reranker over that
zero-coverage top-100 are frozen. Source verification then corrected an
input-contract error: the official BrowseComp-Plus reproduction recipe uses a
512-token query input and a 4096-token document input. The prebuilt-index
repository contains vector shards and hashes but does not bind its historical
preprocessing metadata. The next diagnostic is therefore limited to answer-span
visibility under the documented 4096-token recipe, with that provenance
limitation explicit. The registered audit retained an answer-bearing input for
7/7 cases and 17/18 gold documents, so 4096-token head truncation is rejected as
the dominant explanation and passage-dense is not admitted on this evidence.
The result remains a gold-aware tokenizer diagnosis; it cannot be promoted into
an effectiveness metric. The subsequent two-phase raw-question audit persisted
seven top-1000 slates before opening gold or prior outcomes. Raw-question search
reached 1/7 at top-20, 1/7 at top-100, and 2/7 at top-1000, compared with 0/7,
1/7, and 5/7 for the frozen generated-query baseline. It won only one of seven
per-case rank comparisons, so every registered 4/7 gate failed. A general
raw-question anchor is therefore frozen on this cluster. Existing bridge and
hypothesis probes were then audited: six prompt-only variants had already been
rejected on another saved three-case cluster, including typed contrastive,
draft-blind, firewall, Pro-substitution, and corpus-grounded generation. Those
mechanisms must not be renamed and repeated on the seven current cases. The
current cluster becomes regression-only, and the next registered action is a
single frozen-policy profile over all 175 development questions to acquire a
broader failure distribution. It is not a fresh confirmation. All unrun new
mechanisms, sealed-holdout, official, and leaderboard fields remain `planned`.

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

The optional single-GPU screening path is frozen separately in
`benchmarks/browsecomp_plus_v0/screening_judge_v0.json`. It binds GPU 5,
vLLM 0.9.0.1, a loopback-only OpenAI-compatible endpoint, the official grader
template hash, Qwen3-32B-AWQ revision, and the same temperature/top-p/top-k,
output limit, and non-thinking setting. The launcher must see one idle GPU in
three consecutive checks, expose only the registered served-model name, and
write its command and runtime before inference. No remote API key is used.

AWQ screening is enabled only if all calibration gates pass on the 150 labels
already produced by the pinned BF16 evaluator: at least 90% label agreement,
Cohen's kappa at least 0.75, zero parse failures, at most three percentage
points absolute pooled-accuracy drift, and the same sign for the paired
candidate-minus-baseline delta. A pass permits cheaper development-variant
triage only. Final development claims, sealed-holdout claims, and leaderboard
claims still require the pinned BF16 official evaluator.

The persistent two-GPU BF16 service is a separate registered adapter in
`benchmarks/browsecomp_plus_v0/persistent_bf16_judge_v0.json`. It keeps the
official Qwen3-32B revision, grader-template hash, vLLM version, decoding
values, and non-thinking setting fixed, but it does not inherit official status:
OpenAI-compatible request scheduling replaces the upstream evaluator's
in-process batch call. Before any development score is accepted, the service
must reproduce the existing 150-label BF16 reference with at least 95% label
agreement, Cohen's kappa of 0.85, zero parse failures, no more than two
percentage points pooled-accuracy drift, and the same paired-delta sign. These
thresholds are conjunctive and were registered before the service calibration.

Every persistent-service run binds the ready service registration to the
byte-verified official model assets. A single-variant development score also
revalidates the accepted calibration, frozen prediction summary, prediction
hashes, and post-prediction development-gold slice before inference. It stores
raw prompts, raw judge responses, parsed labels, usage, latency, and hashes.
The resulting metric is named
`calibrated_development_diagnostic_not_official`; it cannot replace the pinned
upstream evaluator for final comparisons, sealed holdout, or submission.

For a repeat grid, score every registered trial and both variants. Do not pick
the best trial or reuse one label across repetitions. The repeat-service Judge
must validate each summary, prediction, gold slice, raw item result, and
calibration hash, then report pooled and trial-level accuracy plus paired query
outcomes. The final Dense Retrieval decision applies every conjunctive gate in
`dense_confirmation_v1.json`, including the generation-cost cap and cost ratio.
A missed gate is retained as a negative result; it is not repaired by changing
the threshold, dropping a trial, or adding a multi-Agent layer post hoc.

The Evidence Bandwidth confirmation follows the same rule. Its pre-registration
freezes a fresh 25-query development slice, three alternating paired trials,
adapter v8, BM25 top-5, dense top-20, and a 1792-token aggregate snippet budget.
Promotion requires all of the following at once: candidate schema completeness
at least 96%, zero output overshoot and final provider failures, evidence-recall
delta at least +10 points, positive query-level evidence wins minus losses,
non-negative calibrated-Judge accuracy delta, zero Judge parse/request
failures, search-call ratio in [0.9, 1.1], total-Token and provider-cost ratios
at most 1.15, and total generation cost no more than the registered cap. Payload
calibration alone is not a matched end-to-end budget because the Agent may issue
different numbers of searches; the final ratios are computed over all saved
provider traces.

The v0 result is retained as a negative result: retrieval recall passed, while
Judge accuracy and the search/Token/cost ratios failed. Progressive Disclosure
was therefore registered as a new mechanism that preserves BM25 full-snippet
anchors, exposes deduplicated dense-only candidates as bounded leads, and
requires explicit document opens. Every search/open payload and cumulative
evidence-ingress token is persisted.

The first fresh-five Progressive Disclosure run was also rejected: Judge
accuracy stayed at 2/5, while 65 searches replayed context until the candidate
used 3.131361 times the selected baseline Tokens. The next experiment changed
only the loop policy: adapter v10 synchronously reserves at most eight searches
per query before any asynchronous tool call and records exhaustion without
turning a scoreable answer into `budget_exhausted`. Its fresh-five run cut the
Token and cost ratios to 0.195385 and 0.445916 and raised evidence recall by
23.33 points, but both variants remained 0/5 under the calibrated Judge. That
run was rejected rather than promoted on resource metrics alone.

Saved traces then exposed a narrower interface bug: all four relevant dense
leads in that slice were truncated before any title value or passage. A
zero-provider-call calibration compared the existing head preview with a
query-aware paragraph window and moved selectable relevant leads from 0/4 to
4/4 under the same payload ceiling. The subsequent fresh-five run held Judge
accuracy at 3/5, raised evidence recall from 55.833333% to 80%, and used
0.324788 times the baseline Tokens and 0.536012 times its provider cost. This
promoted only to the separately preregistered paired-25 confirmation.

`query_aware_preview_confirmation_v0.json` freezes 25 previously unevaluated
development questions, DeepSeek V4 Flash, the empty system prompt, BM25 top-5
baseline, Query-Aware Progressive Disclosure candidate, eight-search governor,
and the accepted persistent BF16 Judge. Promotion requires 25/25 success for
both variants, at least 96% schema completeness, zero output overshoot and Judge
errors, Judge non-regression, at least +5 evidence-recall points, search/Token/
cost ratios at most 1.0/0.75/1.0, and combined generation cost at most $1.50.
Passing one trial only permits a three-trial stability confirmation. It does not
open the sealed holdout or authorize an official/leaderboard claim.

The completed trial is retained as `reject`. Candidate Judge accuracy improved
by 4 points, evidence recall by 8.76 points, and all resource ratios passed, but
baseline schema completeness was 92% against the registered 96% per-variant
minimum. The other 23 gates do not override that failure. Saved-trace diagnosis
is allowed; treating the same run as a pass, deleting the baseline gate, or
repeating until baseline happens to clear it is forbidden.

Any Evidence-Debt Search Reserve follow-up is a new mechanism and needs a new
registration. Its outcome-selected regression/improvement set is calibration
only. A later effectiveness comparison must use unseen development questions,
hold total search calls at eight and total output at 10k, score both variants
with the same calibrated Judge, and preserve every result before gold access.

The official evaluator handoff uses two immutable records. The batch manifest
is created before judge inference and binds each unique
`trial -> variant -> query` input to its prediction hash, ground truth export,
repeat comparison, evaluator contract, and judge asset manifest. The execution
registration is written after three consecutive idle-GPU checks but before the
Qwen call; it records runtime versions, GPU IDs, exact command, upstream hashes,
the copied asset-verification audit, and any allowlisted transport-only
environment override. `NCCL_P2P_DISABLE=1` is permitted only when explicitly
registered; it does not authorize weight, evaluator, prediction, or decoding
changes. The execution result separately binds
the exit code, stdout/stderr, and every evaluator output. Aggregation requires
all expected `_eval.json` files and no extras, zero parse failures, unchanged
response hashes, and exact question/answer bindings. A five-query result is an
official-evaluator development-slice diagnostic, not a leaderboard submission.

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
