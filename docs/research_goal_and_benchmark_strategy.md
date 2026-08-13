# Research Goal and Benchmark Strategy

Status: active research target, not a completed result.

## Goal

Build a training-free Deep Research harness around frozen DeepSeek V4 Flash or
DeepSeek V4 Pro. The project should contribute several connected harness-level
mechanisms, each justified by a measured failure, and obtain a competitive,
reproducible result on a public benchmark.

The project must never describe a harness gain as an improvement to the base
model's intrinsic capability. A high leaderboard position is a stretch outcome,
not evidence by itself: the same-model baseline, fixed controls, ablations,
cost, latency, and failed cases remain mandatory.

## Primary Target: BrowseComp-Plus

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) is the primary
target, subject to a local feasibility gate. It is a better fit than a dynamic
web benchmark for the current research question because it provides:

- 830 reasoning-intensive queries and a fixed corpus of roughly 100K documents;
- human-verified supporting documents and hard negatives;
- separate retrieval and end-to-end metrics;
- accuracy, recall, search-call, and calibration measurements;
- an official submission path and public leaderboard;
- a standard search contract that can be held fixed across harness variants.

This lets us distinguish retriever improvements from controller improvements.
It also removes the uncontrolled live-search failure that dominated the first
LiveDRBench preview. LiveDRBench remains a useful live-web transfer test, but is
not the primary optimization target.

The official end-to-end evaluator uses Qwen3-32B as its judge. Its revision and
decoding contract are pinned, and hash-bound evaluator inputs have been
exported. Inference remains `planned_not_run`: the local 4 GiB GPU cannot host
the judge, and the checked remote GPUs were occupied, so no other workload was
preempted.

## Frozen Leaderboard Snapshot

Snapshot fetched on 2026-08-13 from the official leaderboard JSON:

- entries: 84;
- top-5 accuracy floor: 86.27%;
- top-10 accuracy floor: 78.41%;
- top-20 accuracy floor: 63.86%;
- median accuracy: 37.35%;
- best listed DeepSeek-R1-0528 result: 16.39%;
- snapshot SHA-256: `149482ea50abda17baa7153a47f1209fd12217ce48fdba4c91324d12acbae205`.

Ranks will move, so experiments compare against these frozen thresholds as
well as the live leaderboard at submission time.

The machine-readable pins, split rule, and target ladder are stored in
`benchmarks/browsecomp_plus_v0/target_manifest.json`.

Target ladder:

1. Reproduce the official tool and evaluation contract.
2. Publish the first auditable DeepSeek V4 Flash and Pro standard-scaffold baselines.
3. Beat the same-model standard scaffold under matched search-call, Token, and cost budgets.
4. Reach the frozen top-20 threshold of 63.86% accuracy.
5. Stretch: reach the frozen top-10 threshold of 78.41% accuracy.

The gold-free runtime portion of item 1 has passed on five development queries.
A strict-budget Flash candidate with phase-adaptive reasoning and pinned dense
retrieval is exported for the evaluator, but Qwen3-32B is still
`planned_not_run`, so item 1 is not complete. Items 2-5 remain `planned`. The
current five-query exact and retrieval values are diagnostic evidence, not an
accuracy, effectiveness, or rank claim.

## One Thesis, Several Layers

The mechanisms should form one coherent thesis rather than a feature list:

> Explicit evidence debt can act as a shared control signal for retrieval,
> evidence verification, search allocation, stopping, and answer compilation.

### Layer 1: Constraint-Aware Retrieval

Problem: one broad query drifts toward lexically popular but irrelevant results.

Candidate mechanism: compile the task into entity, relation, temporal, and
numeric constraints; issue a small portfolio of lexical and semantic queries;
fuse results while preserving which constraint caused each retrieval.

Acceptance metrics: evidence nDCG@10, evidence/gold Recall@5/100, and end-to-end
accuracy under a fixed retrieval budget.

### Layer 2: Evidence-Debt Graph

Problem: retrieved documents are mistaken for an answered task, while required
constraints remain unsupported or contradictory.

Candidate mechanism: represent answer obligations, candidate entities, evidence
edges, contradictions, and unresolved debt explicitly. Evidence must be linked
to the exact constraint it supports rather than only attached to a report.

Acceptance metrics: obligation coverage, contradiction detection, unsupported
answer rate, and downstream accuracy with retrieved documents held fixed.

### Layer 3: Marginal-Value Search Controller

Problem: fixed query counts either stop too early or spend calls on redundant
branches.

Candidate mechanism: select the next query or document inspection by estimated
evidence-debt reduction per marginal search call, Token, or monetary cost. Stop
when no admissible action clears the value threshold or the budget is exhausted.

Acceptance metrics: accuracy-recall curves versus search calls, Tokens, cost,
and latency; calibration error is a required secondary metric.

### Layer 4: Constraint-Locked Answer Compiler

Problem: the system can retrieve the right evidence but emit a malformed,
partially supported, or overconfident answer.

Candidate mechanism: compile the final answer only from verified candidate
nodes that satisfy the task's output schema and main constraints. Permit an
auditable abstention when the evidence threshold is not met.

Acceptance metrics: exact task accuracy, output-schema validity, unsupported
answer rate, and calibration error with retrieval/controller traces frozen.

### Layer 5: Counterfactual Trace Replay

Problem: an end-to-end score cannot tell whether a gain came from retrieval,
controller policy, more context, or more spending.

Candidate mechanism: replay persisted retrieval results and swap one policy at
a time, so selected controller and compiler ablations can run without paying
for or changing retrieval again.

Acceptance metrics: reproducible per-task decisions, identical replay inputs,
and paired ablations that attribute gains to one changed mechanism.

Pi may be used as a generic agent-loop baseline or an optional execution adapter.
It is not itself an innovation claim, and adopting it must not replace the
benchmark contracts, evidence graph, trace schema, or matched-budget controls.

## Leakage and Comparison Contract

Before accessing answers or relevance labels:

1. Pin the benchmark commit, dataset revisions, indexes, evaluator, and judge.
2. Hash-split query IDs into a development partition and a sealed holdout.
3. Store gold material behind an evaluator-only boundary.
4. Freeze predictions and their SHA-256 before scoring the holdout.
5. Limit holdout evaluations; preserve every result, including failed runs.

Development can use the development partition. The primary scientific claim is
based on the sealed holdout. The official 830-query leaderboard result is
reported separately and must disclose that the public benchmark has an
accessible development subset.

Flash and Pro are separate model tracks. Within an ablation, model identifier,
thinking mode, prompt policy, retriever/index, task set, search interface,
maximum search calls, Token/cost budget, concurrency, and evaluator are fixed.

## Immediate Execution Order

The first causal sequence is now complete. The true strict standard baseline
produced 0/5 schema-complete answers. An 8k high-thinking exploration plus 2k
non-thinking compilation policy reached 4/5 without exceeding the same 10k
allowance. Frozen-query counterfactual replay then justified a dense retriever
run. In the corrected, exactly matched snippet contract, end-to-end evidence
recall rose from 40.00% to 54.92% on the five-query development diagnostic,
schema completion reached 5/5, and strict exact remained 1/5. Two further
answers appear semantically equivalent and await the official judge. An earlier
stronger-looking dense run is excluded from the primary comparison because it
normalized short snippets differently; the run-to-run difference also requires
repeat trials rather than selecting the best trace.

The remaining zero-recall case was not fixed by a first-turn deadline, one
forced bootstrap search, or a three-query rare-anchor portfolio. Those negative
controls localize the issue to query compilation, but further tuning on one
question would overfit the development slice.

1. Run the pinned Qwen3-32B judge on both frozen BM25 and dense five-query predictions; require zero parse failures and retain every per-query judgment.
2. Under adapter v6, run at least three independent paired trials per query for BM25 and dense; report means, standard deviations, and paired wins without best-run selection.
3. Freeze a 25-query development slice and stop changing prompts before running the same-model, same-phase-policy BM25/dense paired ablation.
4. Promote dense retrieval only if evidence recall improves by at least 10 percentage points and official accuracy does not decline; report search calls, Token, cost, and latency together.
5. Cluster at least ten retrieval failures before designing Constraint Portfolio v1. First require query-only replay gains; do not pay for end-to-end runs until recall moves.
6. Run all 175 development queries only after the candidate policy and acceptance thresholds are frozen.
7. Keep the 655-query holdout sealed until the mechanism and thresholds are preregistered; use DeepSeek V4 Pro only as a separate final track.

Exact run hashes, costs, rejected controls, and the inference boundary are in
`browsecomp_plus_layered_results_2026-08-13.zh-CN.md`.
