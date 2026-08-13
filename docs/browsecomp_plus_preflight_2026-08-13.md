# BrowseComp-Plus Preflight (2026-08-13)

Status: generation and retrieval preflight passed; official Qwen3-32B judging is
`planned_not_run`. No official accuracy or leaderboard rank is claimed.

The detailed Chinese experiment record is
`browsecomp_plus_layered_results_2026-08-13.zh-CN.md`.

## Frozen Target

- benchmark repository commit: `046949032b0328319cc9a02663a759ec601d9402`;
- query dataset revision: `144cff8e35b5eaef7e526346aa60774a9deb941f`;
- corpus dataset revision: `b27b02bc3e45511b8b82a13e6f90ce761df726f6`;
- index dataset revision: `b3f37f70c33829eb09d04784a54277a31871fd63`;
- official judge revision: `9216db5781bf21249d130ec9da846c4624c16137`;
- DeepSeek tracks: `deepseek-v4-flash` and `deepseek-v4-pro`;
- query split: 175 development and 655 sealed holdout;
- standard tool contract: top-5, at most 512 Qwen tokens per snippet, no
  `get_document`, empty system prompt, at most 100 iterations, and 10,000
  generated tokens.

The leaderboard snapshot and its SHA-256 are stored in
`benchmarks/browsecomp_plus_v0/target_manifest.json`. Its top-20 and top-10
accuracy floors are frozen targets, not achieved results.

## Local Assets

The official BM25 index, Qwen3-Embedding-0.6B index, Qwen3-Embedding-0.6B
model, snippet tokenizer, Java 21 runtime, and Python/Node adapters are stored
below ignored `runs/` or user caches. The dense candidate is machine-readable
in `benchmarks/browsecomp_plus_v0/retriever_candidates.json`, including model
and shard hashes.

The local RTX 3050 Laptop GPU has 4 GiB and cannot host the official Qwen3-32B
judge.

Remote preparation completed on 2026-08-14 without running inference:

- the upstream repository is a clean detached checkout at
  `046949032b0328319cc9a02663a759ec601d9402`;
- the frozen `uv.lock` SHA-256 is
  `45d3e6d00719dbf732160b25e3419ed4599121e5d832723357ff2fea01477c43`;
- the upstream evaluator SHA-256 is
  `1a21233937c377ab6323c98ff9af67742756a57fbacab4ebf9bc30852eae530a`;
- the installed runtime imports `torch 2.7.0+cu126`, `transformers 4.53.2`,
  and `vllm 0.9.0.1`, with CUDA visible;
- Qwen3-32B was obtained through a mirror, then all 24 required assets were
  checked against the pinned Hugging Face revision. The audit passed 24/24,
  with artifact SHA-256
  `f919e78fe5e8346aa84432616cbbec8bc32588387eebd18ce7d893c919215719`;
- all 30 frozen official-input files and the five-row prediction-bound
  development ground truth are staged. The ground-truth JSONL SHA-256 is
  `9a975130c225bc66fa5a1fa206098bb2458ca782150e86339a72b63417c7d259`.

The official judge still requires two idle 48 GiB GPUs for tensor parallelism.
The latest read-only check found workloads on all eight GPUs. No process was
stopped and no workload was preempted, so inference remains `planned_not_run`.

## Leakage Boundary

1. Query IDs were hash-partitioned before gold access.
2. Generation reads development questions only; plaintext stays under ignored
   `runs/`.
3. Each prediction and run file is SHA-256 frozen before development scoring.
4. The development scorer projects only question, answer, gold docids, and
   evidence docids. It excludes document text, URLs, and negatives.
5. The 655-query holdout remains sealed.
6. Official evaluator export never reads gold or rewrites model output.

## Provider Budget Correction

Pi 0.84.1 auto-detected `max_completion_tokens` for DeepSeek. The API accepted
that field, but observed reasoning output was not bounded by it. All runs using
that path are retained as failed protocol diagnostics and excluded from
matched-budget evidence.

Adapter v1+ forces DeepSeek's documented `max_tokens` field. Every provider
attempt records the requested and applied limit, global/phase remaining budget,
phase, and thinking type. Python contracts reject v1+ traces that do not use
`max_tokens`. All strict runs below have zero output-budget overshoot.

## Strict Five-Query Runs

| Run | Retriever | Policy | Schema | Search calls | Output tokens | Cost USD | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| standard | BM25 | high thinking, no reserve | 0/5 | 80 | 50,000 | 0.04845 | valid strict baseline |
| reserve high | BM25 | 8k high + 2k high | 2/5 | 66 | 49,583 | 0.06668 | gate failed |
| reserve larger compile | BM25 | 6k high + 4k high | 0/5 | 48 | 49,998 | 0.05323 | hypothesis rejected |
| phase adaptive | BM25 | 8k high + 2k thinking-off | 4/5 | 70 | 43,120 | 0.06867 | schema gate passed |
| phase adaptive | dense 0.6B | 8k high + 2k thinking-off | 5/5 | 31 | 28,854 | 0.02690 | corrected candidate |

The strict normalized exact diagnostic is 1/5 for both BM25 and dense. Evidence
recall is 40.00% and 54.92%, respectively. These are development diagnostics,
not official accuracy. Two additional dense outputs are semantically consistent
with the references but include extra wording; that inference remains
unconfirmed until the official judge runs. An earlier dense run reached 2/5
strict exact and 77.14% evidence recall, but it is excluded from the primary
comparison because its short-snippet normalization did not exactly match the
BM25 contract. The spread between runs requires repeat trials rather than
best-run selection.

## Adapter-v6 Paired Repeats

Three alternating-order BM25/dense trials completed on the same five questions
with 30 unique provider run IDs and zero output-budget overshoot. Trial-level
BM25 versus dense mean +/- sample standard deviation was:

- schema completion: 86.67% +/- 11.55 versus 100.00% +/- 0.00;
- strict exact diagnostic: 6.67% +/- 11.55 versus 46.67% +/- 11.55;
- evidence recall: 17.36% +/- 2.88 versus 60.00% +/- 14.84;
- search calls per query: 11.27 +/- 2.52 versus 6.40 +/- 2.80;
- total tokens per query: 208,353.07 +/- 63,467.92 versus
  81,843.93 +/- 56,046.36;
- estimated cost per query: USD 0.011472 +/- 0.002356 versus
  USD 0.005662 +/- 0.002053;
- latency per query: 88.88 s +/- 9.22 versus 59.77 s +/- 7.17.

Dense won/lost/tied 8/2/5 query-trial evidence-recall pairs and 6/0/9 strict
exact pairs. These are repeated observations of five fixed development
questions, not 15 independent questions and not official accuracy. The first
automation run was interrupted after trial 1 BM25, development gold was opened,
and the manifest was reconstructed on resume. Generation controls were not
changed, but the result is correctly labeled `reconstructed_after_interruption`
rather than preregistered confirmatory evidence.

Reported dollars are DeepSeek API estimates only; dense-index hosting is not
priced. Both retrievers receive the same maximum output allowance, but realized
input/total tokens and cost are not matched. This is a retriever ablation under
one output-budget contract, not a full-system cost-matched claim.

## Counterfactual Retrieval Gate

The replay used all 70 frozen BM25 agent queries without regeneration:

- BM25 evidence recall: 40.00%;
- dense evidence recall: 55.95%;
- delta: +15.95 percentage points;
- BM25+dense RRF: 54.37%.

This isolated retriever improvement justified the paid dense agent run. RRF did
not beat pure dense on the slice and was not promoted.

## Failed Control Probes

The remaining bad case made zero search calls. Three one-query probes were
retained:

- a 512-token first-turn cap plus prompt nudge still made zero searches;
- a non-thinking tool bootstrap made one search but retrieved no relevant doc;
- a three-search rare-anchor portfolio also retrieved no relevant doc.

These probes repaired mechanics but not query quality. They are negative
results, not promoted innovations. Further single-question prompt tuning is
stopped to avoid development overfitting.

## Next Gate

1. Run the pinned official Qwen3-32B judge on all six frozen repeat exports;
   retain 30 per-query judgments and require zero parse failures.
2. Freeze a 25-query development slice and run paired BM25/dense Flash variants
   without changing prompts.
3. Require dense evidence recall to improve by at least 10 percentage points
   with no official-accuracy decline before promotion.
4. Develop another query compiler only after at least ten larger-slice failures
   share that diagnosis, and require query-only replay gains first.
5. Keep the 655-query holdout sealed until the final policy, budgets, and gates
   are preregistered.
