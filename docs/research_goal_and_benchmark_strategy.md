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

The official end-to-end evaluator uses Qwen3-32B as its judge. Running that
evaluator and reproducing its environment are currently `planned`, not done.

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

Items 1-5 are all `planned`. No rank or effectiveness claim has been earned.

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

1. Reproduce the BrowseComp-Plus data/index/tool contract on five queries without scoring.
2. Freeze the development/holdout manifest before opening gold data.
3. Run a standard-scaffold DeepSeek V4 Flash baseline on the development split.
4. Diagnose its largest error cluster and implement only Layer 1's smallest justified change.
5. Promote a layer only after a same-model matched-budget ablation passes its gate.
6. Use DeepSeek V4 Pro only after the Flash pipeline is stable; do not hide model substitution inside a harness comparison.
