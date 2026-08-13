# BrowseComp-Plus Preflight (2026-08-13)

Status: metadata and environment preflight completed; no benchmark query,
answer, relevance label, corpus, or index has been downloaded or scored.

## Pinned Inputs

- official repository commit: `046949032b0328319cc9a02663a759ec601d9402`;
- query dataset revision: `144cff8e35b5eaef7e526346aa60774a9deb941f`;
- corpus revision: `b27b02bc3e45511b8b82a13e6f90ce761df726f6`;
- index dataset revision: `b3f37f70c33829eb09d04784a54277a31871fd63`;
- initial reproducibility baseline: official BM25 index and standard top-5 search contract.

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
- Node.js 22.14.0: available;
- Java: missing;
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB;
- `DEEPSEEK_API_KEY`: present in the user environment;
- local provider model: `deepseek-v4-flash`, with no inline API key.

The official BM25 stack uses Pyserini and requires Java 21. Install or provide
Java before downloading the index. The official end-to-end evaluator uses
Qwen3-32B; the local 4 GiB GPU cannot reproduce that judge. Official scoring
therefore needs a suitably sized remote GPU and remains `planned_not_run`.

## Adapter Boundary

The official OpenAI client drives the OpenAI Responses API. DeepSeek V4 exposes
an OpenAI-compatible Chat Completions interface instead. Therefore the official
agent client cannot be used unchanged.

The smallest responsible implementation is a DeepSeek Chat Completions adapter
that preserves the official benchmark's search tool definition, top-k, snippet
limit, query template, iteration cap, and persisted output schema. The adapter
must be validated against an offline fake tool loop before any paid benchmark
run.

Pi remains an optional implementation route for this adapter and a useful
generic-loop baseline. The decision should be made by a small compatibility
spike, not by rewriting the existing harness first.

## Next Gate

1. Add and test the deterministic split function from the target manifest.
2. Install Java 21 without changing project dependencies.
3. Download only the pinned BM25 baseline artifacts.
4. Run five unscored queries through the exact tool contract.
5. Inspect trace completeness and cost before opening any gold data.
