# BrowseComp-Plus Preflight (2026-08-13)

Status: the pinned standard BM25 retrieval runtime and corrected five-query
DeepSeek V4 Flash standard-budget smoke are complete. A preliminary diagnostic
exposed and preserved a generation-budget bug. Development questions were
projected and decrypted only after the split was frozen. No benchmark answer,
relevance label, gold document, or official score has been opened or produced.

## Pinned Inputs

- official repository commit: `046949032b0328319cc9a02663a759ec601d9402`;
- query dataset revision: `144cff8e35b5eaef7e526346aa60774a9deb941f`;
- corpus revision: `b27b02bc3e45511b8b82a13e6f90ce761df726f6`;
- index dataset revision: `b3f37f70c33829eb09d04784a54277a31871fd63`;
- initial reproducibility baseline: official BM25 index and standard top-5 search contract.
- query split artifact: `query_partitions.json`, 175 development IDs and 655
  sealed-holdout IDs, query-ID SHA-256
  `6e003b0f559a1f67fd8d6b041aa72a133581da164808194d1ff40b16bbdd1b95`.

The reference repository is checked out read-only under ignored `runs/_external`
for source inspection. None of its code is copied into this project.

## Storage Estimate

Sizes were read from the pinned Hugging Face repository metadata:

| Artifact | Bytes | GiB |
| --- | ---: | ---: |
| encrypted query/evaluation dataset | 2,781,436,779 | 2.590 |
| fixed document corpus | 1,761,586,179 | 1.641 |
| official BM25 index | 2,170,624,147 | 2.022 |
| total before caches/environments | 6,713,647,105 | 6.253 |

The `G:` drive had 104.13 GiB free during preflight, so storage is not the
current blocker.

## Local Runtime

- Python 3.12.7: available;
- uv 0.10.9: available;
- system Node.js 22.14.0: available but below Pi's declared minimum;
- bundled Node.js 24 runtime: used for the Pi adapter and tests;
- project-local Temurin Java 21.0.12+8: available under ignored `runs/_external`;
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB;
- `DEEPSEEK_API_KEY`: present in the user environment;
- local provider model: `deepseek-v4-flash`, with no inline API key.

The 2,170,624,147-byte official BM25 index is present under ignored
`runs/_external` and all Git LFS objects match their pinned SHA-256 values. A
project-local Python 3.12 environment runs Pyserini 1.2.0 and the pinned
Qwen/Qwen3-0.6B tokenizer revision. A live search probe returned exactly five
results. The official end-to-end evaluator uses Qwen3-32B; the local 4 GiB GPU
cannot reproduce that judge. Official scoring therefore needs a suitably sized
remote GPU and remains `planned_not_run`.

The evaluator contract is separately frozen in
`benchmarks/browsecomp_plus_v0/official_evaluator.json`: official repository
commit, evaluator script, Qwen3-32B revision
`9216db5781bf21249d130ec9da846c4624c16137`, temperature 0.7, top-p 0.8,
top-k 20, 4,096 output tokens, and thinking disabled. These are pins, not proof
that the evaluator has run. The exact remote image, CUDA stack, and vLLM version
must be added to the score-run manifest before execution.

`benchmarks/browsecomp_plus_v0/deepseek_provider_snapshot.json` records the
officially documented served versions (`DeepSeek-V4-Flash-0731` and
`DeepSeek-V4-Pro-0813`) and the price table used by these traces. That price
table is documented as changing at `2026-08-16T16:00:00Z`; cost-matched runs
after that instant must create and bind a new snapshot rather than reuse these
rates.

## Adapter Boundary

The official OpenAI client drives the OpenAI Responses API. DeepSeek V4 exposes
an OpenAI-compatible Chat Completions interface instead. Therefore the official
agent client cannot be used unchanged.

The pinned contract now records BM25, top-5, 512 Qwen-token snippets, no
`get_document`, the official search-only query template, at most 100 model
rounds, a requested 10,000 generated-token ceiling, and an empty system prompt.

Pi 0.84.1 is now the thin standard-loop adapter. A compatibility spike proved
that its coding-agent system prompt, skills, extensions, prompt templates, and
ancestor context can all be disabled; the saved run contract rejects a
non-empty system prompt. The adapter uses DeepSeek Chat Completions and stores
every model message, search query/result, provider usage, cost, and latency.
Pi is infrastructure, not an innovation claim.

## Preliminary Five-Query Diagnostic (Non-Standard)

Five development questions were selected deterministically and answered with
DeepSeek V4 Flash using the pinned BM25 retrieval interface. This first adapter
version did **not** reproduce the official generation budget. It incorrectly applied the
10,000-token value independently to each provider request and added a separate
100-search-call guard. It is retained as negative implementation evidence, not
as the standard baseline. Aggregate observations:

- completed question traces: 5/5;
- total BM25 search calls: 218;
- provider-reported total tokens: 13,091,832;
- estimated DeepSeek fee from provider usage: $0.16223506;
- summed wall latency: 1,470,194 ms (about 24.5 minutes, sequential);
- all five emitted the requested `Exact Answer` and `Confidence` fields;
- one trace abstained after 102 searches rather than identifying an answer.

This is a runtime diagnostic, not an accuracy result. It exposes how badly a
misinterpreted budget can inflate work: one question reached 102 recorded
search invocations and more than 9 million provider-reported tokens.
Provider-reported `total_tokens` sums repeated context across model requests;
it is a billing/work measure, not the length of one context window.

The first three runs were produced before `model_requests` became a new trace
field, so their saved traces contain full assistant turns and search calls but
leave this derived count absent. The last two runs report 52 and 14 model
requests respectively. The raw questions and traces remain ignored under
`runs/`.

## Protocol Repair Log

Two later adapter defects were also preserved rather than folded into the
baseline. First, terminal tool calls were rejected before the reference loop's
next-turn budget check; that partial run is under ignored
`pi_flash_prestandard_terminal_search_guard_20260813`. Second, a terminal answer
that crossed its requested allowance was incorrectly classified as unfinished;
the five-query run is under ignored
`pi_flash_prestandard_completion_status_20260813` with summary SHA-256
`7aabffdf8e267eade5a9c2d8d4f25921326093fd90c5cefac41467d1a4f75b9b`.
Neither run is used as a baseline or effectiveness result.

## Corrected Standard-Budget Smoke

The adapter now treats 10,000 generated tokens as a global per-question budget,
caps every next provider request at the remaining allowance, executes tool calls
already emitted by the terminal response, and allows at most 100 model rounds.
Pi compaction is disabled and the trace records every requested and applied
provider limit. DeepSeek may report more reasoning/output tokens than requested;
the exact overshoot is preserved. A terminal answer remains scoreable, while an
unfinished tool loop is marked `budget_exhausted`.

Observed results:

- run termination: 1 `succeeded`, 4 `budget_exhausted`, 0 provider/runtime failures;
- model requests: 37;
- BM25 calls: 71, all successful;
- provider-reported output tokens: 49,486; four questions individually
  overshot their requested allowance by a total of 1,915 tokens (range 74-764);
- provider-reported total tokens: 1,044,708, including repeated context;
- estimated DeepSeek fee from provider usage: $0.04028363;
- summed wall latency: 518,180 ms (about 8.6 minutes, sequential);
- empty system prompt and recorded per-request limits: 5/5 traces;
- required `Exact Answer` plus `Confidence` fields: 1/5 traces.

The aggregate summary is at ignored
`runs/browsecomp_plus_v0/pi_flash_standard_smoke_20260813/summary.json`; its
SHA-256 is
`cb50982652823681b8e489a25759d7e026f37e46a59bc12accda804b73145f1d`.
Its prediction and run hashes freeze this diagnostic before any development
gold is opened. This is not an accuracy result. It localizes a failure earlier
than retrieval quality: under the 10,000-token global contract, high-thinking
Flash usually consumes the remaining allowance before compiling a scoreable
answer.

The gold-free official-shape export contains one `completed` and four
`incomplete` inputs. Its export-manifest SHA-256 is
`85de377efcc71ff326f7206fd3059a96f5520edc354ebd53bfa3863cd60ef1ee`.
This proves evaluator compatibility and hash lineage only; Qwen3-32B has not
run, so no accuracy value is available.

## Next Gate

1. Treat the completed 10,000-token run as the standard-scaffold baseline and
   preserve both failed and partial predictions.
2. Test the simplest budget-aware alternative first: reserve a fixed part of
   the same 10,000-token allowance for final answer compilation instead of
   increasing the budget. Acceptance is at least 4/5 schema-complete answers,
   with no change to model, prompt, BM25, top-k, or total output allowance.
3. Open development gold only inside the evaluator boundary after predictions
   are frozen and hashed; keep the 655-query holdout sealed.
4. Run the official Qwen3-32B judge on a suitably sized remote GPU. Accuracy and
   leaderboard claims remain `planned_not_run` until that succeeds.
5. Diagnose scoreable failures into retrieval, evidence integration, stopping,
   and answer-compilation clusters; only then select the next innovation layer.
