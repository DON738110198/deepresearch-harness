# Architecture

## Design position

This project is a **training-free harness**. Model parameters remain frozen. Any observed difference belongs to the orchestration, prompt, evidence collection, and budget policy under the documented evaluation setup; it must not be described as a model capability improvement.

## Current baseline

```text
CLI input
  -> ProviderFactory (fake | OpenAI-compatible)
  -> BaselineResearchPipeline | PrimarySourceResearchPipeline
       plan(question)
       collect(plan.search_queries, local corpus | bounded live web)
       build_ledger(evidence)
       write_report(question, ledger)
  -> RunStore(state.json, report.md)
```

`RunState` is the source of truth. It has a status history and serialized task plus optional decision context, plan, evidence, claims, citations, trace events, and totals. `RunStore` writes atomically by replacing a temporary JSON file, so an interrupted process does not leave a partial `state.json` in place.

The BrowseComp-Plus path has a deliberately separate adapter boundary. Pi
0.84.1 supplies only the generic DeepSeek tool loop. The project disables Pi's
coding prompt and all ambient resources, injects the pinned benchmark prompt,
serves pinned BM25, dense top-5, or Evidence Bandwidth top-20 retrieval over
loopback with a Qwen-tokenizer snippet contract, and validates the result back
in Python with strict Pydantic contracts. The top-20 policy allocates one fixed
snippet-token budget across the result set rather than granting 20 full-size
snippets. Query preparation projects
only `query_id` and encrypted `query` columns; answer and evidence columns never
enter generation. Decrypted questions and raw traces are restricted to ignored
`runs/` paths. The adapter clamps each provider turn to the remaining global
output allowance, executes tool calls emitted by that turn, then checks whether
another model round is allowed. Provider-reported overshoot is retained as a
separate audit field; a terminal answer remains scoreable, while an unfinished
tool loop becomes `budget_exhausted`. A hash-verifying exporter then projects
these traces into the official evaluator's run shape without reading benchmark
gold or altering prediction text. A separate replay boundary sends frozen agent
queries through a pinned dense index, compares BM25/dense/RRF docids, and opens
development docid labels only after prediction hashes exist. This separates a
retriever effect from a query-policy or provider effect before another paid run.

`browsecomp_repeats.py` is the reliability boundary around nondeterministic
provider runs. A repeat experiment normally writes its manifest before the first
generation, alternates baseline/candidate execution order, and uses a distinct
provider run ID for every query-trial observation. Aggregation fails closed
unless all trials use the adapter version and per-variant result widths named by
the pre-generation manifest, the same model, control policy, query file, prompt
hashes, 10k no-overshoot contract, and expected retriever identities. It
also verifies the source summary, diagnostic, run, prediction, and official
export hashes, recomputes exact/recall rows from each prediction-bound gold
slice, and requires identical canonical gold rows across all variants before
reporting trial means, sample standard deviations, or paired outcomes.
Interrupted automation can be resumed only from a structurally complete frozen
summary. Successful and budget-exhausted records are immutable. A `failed`
record may be retried only after its request, latest run or error, prediction,
controls, and all prior-attempt hashes validate. The old live artifacts and the
source summary move into an append-only attempt history; cumulative Token,
cost, search, compiler-call, and latency totals include abandoned attempts,
while the latest successful prediction alone is exported for judging. The
summary is atomically checkpointed after each retried query. Derived gold,
diagnostic, and export artifacts are archived before regeneration so a stale
score cannot bind to a new prediction. This is operational recovery, not a new
harness variant or free generation budget. A manifest reconstructed after
generation is permanently labeled `reconstructed_after_interruption` rather
than presented as a preregistration.
New `pre_generation` manifests must also bind the normalized query-artifact
SHA-256. Resume compares this hash as well as the path, so replacing the
plaintext question artifact in place cannot silently change a paid experiment.
Repeat-contract v1 additionally preregisters the only allowed provider-failure
recovery: retry `failed` records only, keep succeeded and budget-exhausted
records immutable, preserve every superseded attempt, accumulate all observed
usage, and stop after three resume invocations. Legacy v0 experiments remain
loadable but any later recovery is labeled a post-failure operational amendment
rather than silently backfilled into the original registration.
`audit-pi-browsecomp-resume` exposes the same validation boundary as a read-only
preflight. It reconstructs each latest request from the registered controls,
validates live and archived attempt hashes, and reports only the immutable and
retry-eligible sets plus cumulative usage. It neither requires an API key nor
starts the retriever, writes artifacts, reads gold, or invokes the provider.

Two machine-readable snapshots close the external-version boundary:
`deepseek_provider_snapshot.json` binds API aliases to documented served
versions and a dated price regime, while `official_evaluator.json` binds the
Qwen3-32B judge revision and inference settings. The evaluator also hash-binds
`official_judge_assets.json`; its standalone verifier checks mirror downloads
against all 17 upstream LFS shards and the runtime config/tokenizer Git blobs
before model loading. These snapshots are validated against the
normalized target-manifest hash so platform drift cannot silently enter a
matched comparison.

`browsecomp_judge.py` closes the evaluator handoff. It stages a self-contained
batch containing uniquely named inputs, prediction-bound development ground
truth, repeat manifests, evaluator contract, and judge asset manifest. The
batch manifest binds all 30 input and prediction hashes before inference. The
standalone launcher then records the clean upstream commit, evaluator and
lockfile hashes, byte-verified write-protected model assets, runtime versions,
selected GPUs, three consecutive idle checks, exact command, logs, and output
hashes. Runtime transport overrides are allowlisted and recorded; the current
host required only `NCCL_P2P_DISABLE=1` after an otherwise unchanged launch
stalled during communicator initialization. Official-result aggregation
recomputes the expected result filename
grid, rejects parse failures or response substitution, and reports trial-level
accuracy distributions plus paired query outcomes. Development-slice output is
explicitly labeled `not_submitted`, never as a leaderboard score.

`screening_judge.py` is a separate evaluator-ablation boundary for machines
with only one 48 GB GPU. A launcher starts a loopback-only vLLM
OpenAI-compatible service on the registered physical GPU and serves the pinned
Qwen3-32B-AWQ revision. The client reuses the exact upstream grader template
and decoding values, sends concurrent Chat Completions requests, and persists
every raw response, parsed label, usage record, latency, and hash. AWQ labels
are accepted only for development-candidate screening if they pass the tracked
150-label calibration against the existing two-GPU BF16 run. Quantization and
the service adapter change the evaluator contract, so this path can never emit
an official score.

The same module now exposes a generic service-judge calibration boundary for
the persistent two-GPU BF16 deployment. The tracked manifest distinguishes
model identity and precision from the execution adapter, while
`run_browsecomp_plus_existing_service_judge.py` binds an already-running
loopback service to its PID/GPU registration and the byte-verified official
assets before replaying the frozen 150-item calibration batch. Model startup is
therefore amortized without pretending that a Chat Completions adapter is the
upstream evaluator.

`development_judge.py` is the single-variant scoring boundary after that
calibration passes. It revalidates the calibration against its official
reference, then binds a frozen `PiSmokeSummary`, every `run.json` and prediction
hash, and the matching post-generation development-gold slice. Concurrent
requests persist raw prompts/responses, parsed labels, token usage, latency, and
hashes. Its result type is explicitly diagnostic and fail-closed on request or
parse failures.

`repeat_development_judge.py` applies that same validated boundary to every
trial/variant in a frozen repeat manifest. It scores all six variants rather
than selecting a favorable replicate, preserves the individual result files,
and aggregates trial distributions plus paired query wins/losses. The
comparison is bound to the repeat manifest, repeat comparison, Judge manifest,
and accepted calibration. Its status remains
`calibrated_development_diagnostic_not_official`.

`dense_confirmation_decision.py` applies the gates registered in
`dense_confirmation_v1.json` to the repeat and Judge comparisons. In addition
to retrieval and Judge deltas, it checks structure, output overshoot, paired
query wins, final provider failures, combined generation cost, and the
candidate/baseline cost ratio. It reuses saved candidate diagnostics to route
the next experiment by failure layer and emits a hash-bound decision under
ignored `runs/`.

`evidence_bandwidth.py` and `evidence_bandwidth_server.py` implement the next
retrieval-interface experiment without changing the frozen generator. The
server retrieves dense top-20, assigns every result a minimum allocation, then
water-fills the remaining fixed snippet-token budget in rank order. Adapter v8
records the widened result contract while older v7 runs retain their original
hashes and behavior. `dense_depth_probe.py` and `wide_selector_probe.py` are
zero-provider-call localization tools: the first measures whether relevant
documents exist below rank 5; the second tests whether a simpler fixed top-5
selector is sufficient before paying for a wider live run.

`evidence_bandwidth_decision.py` revalidates the pre-registration, candidate
manifest, six generation summaries, automatic comparison, persistent-Judge
execution, and calibration before applying all 16 conjunctive gates. In
addition to recall and resource ratios, it compares the saved diagnostic rows
with paired Judge labels. This exposes a synthesis-loss signal when broader
retrieval reduces no-evidence failures but increases failures where relevant
evidence was already present. The signal routes the next experiment; it is not
treated as causal proof or a model-capability change.

`evidence_selectivity_probe.py` is the zero-provider-call bridge from that
decision to the next mechanism. It revalidates all 150 saved generation traces,
their prediction-bound development docids, paired Judge labels, and source
hashes. For each query-trial pair it records unique versus repeated result
slots, distinct query strings, first relevant call, minimum relevant rank,
tokens, and latency. Grouping by candidate improvement/regression/both outcomes
showed that dense rescues long-tail evidence while also losing BM25 anchors and
reinjecting the same documents across different searches.

`progressive_disclosure.py` and `progressive_disclosure_server.py` implement the
smallest mechanism suggested by that diagnosis: each search returns full BM25
anchors, deduplicated dense leads, and opaque document IDs that may be opened
under a separate ingress budget. The server owns run-scoped disclosure state;
the adapter records search payload composition, every open, cumulative ingress,
and terminal state. The first fresh-five run was rejected because it preserved
Judge accuracy but replayed context across 65 searches, using 3.13 times the
stored baseline Tokens. Adapter v10 therefore adds a concurrency-safe hard
eight-search governor and an auditable stop reason before adding any planner,
critic, or Agent.

`evidence_preview.py` fixes the next measured interface failure. Dense leads had
contained only a frontmatter marker and no selectable title or passage. Its
query-window policy parses frontmatter and selects one bounded paragraph by
distinct query-term overlap. A zero-provider-call calibration moved selectable
relevant leads from 0/4 to 4/4 without increasing the registered search payload
cap. On a new five-query development slice, the combined preview and governor
kept calibrated Judge accuracy at 3/5, increased evidence recall from 55.833333%
to 80%, and used 0.324788 times the baseline Tokens and 0.536012 times its API
cost. This promoted only to a fresh paired-25 confirmation; it is not an
official benchmark or model-capability result.

`query_aware_confirmation_decision.py` is the paired-25 decision boundary. It
revalidates the preregistration, question-only slice, frozen adapter and server
hashes, both generation summaries and run hashes, prediction-bound diagnostics,
and both calibrated service-Judge executions. Its 24 gates are conjunctive and
cover completion, schema, research-budget traces, Judge errors and accuracy,
evidence recall, search/Token/cost ratios, and the combined generation-cost cap.

`browsecomp_decision.py` is the mechanism-selection boundary after official
evaluation. It revalidates the repeat comparison, judge batch, execution, and
comparison before applying the tracked `promotion_gates.json`. It does not pick
the best replicate. It checks minimum query/trial scope, clean registration,
evidence-recall delta, official-accuracy non-regression, and parser integrity;
then reports search/Token/cost/latency deltas and classifies each incorrect
candidate observation as format failure, no relevant document retrieved, or
relevant evidence present but answer incorrect. The resulting next action is
therefore tied to an observed failure layer. Query IDs and derived profiles
remain under ignored `runs/`; only gate definitions and bounded aggregate
conclusions belong in the repository.

B0 and B1 share that state contract. B0 performs direct question retrieval and one structured write call; B1 adds planning plus an intermediate ledger. In both variants, citation identifiers and markers are generated by deterministic harness code rather than trusted to model numbering.

`BenchmarkResearchPipeline` keeps B1's three model calls but replaces free-form report writing with a task-defined JSON answer contract. The structured answer is persisted in `RunState`; the Markdown artifact is compiled deterministically with the claim ledger and source links. This is output adaptation, not an additional research mechanism.

B2 keeps B1's three provider calls and the same collector boundary, but turns the plan into an explicit answer contract. Each planned obligation has a dedicated evidence query and a persisted `EvidenceDebt` record linking obligation -> evidence -> claim or marking the obligation `open`. Open debt is appended to the report deterministically, so missing support cannot disappear between planning and writing.

## Evidence Debt diagnostic boundary

The current research loop diagnoses an unresolved obligation before adding more
orchestration. `evidence_span_oracle.py` asks whether an already retrieved
document contains a selectable answer-bearing span. `document_target_oracle.py`
tests whether obligation-conditioned ranking can identify a bounded document
slate without reading gold labels. `post_run_overlay.py` then applies a
fail-closed replacement rule: it may replace the baseline answer only when the
short answer and supporting quote are literal members of every cited span.

For persistent retrieval misses, `corpus_answerability.py` separates corpus
absence from retrieval failure, and `lexical_rank_audit.py` measures where gold
documents appear under the frozen full-document BM25 representation. These are
diagnostic or calibration tools, not additional Agent roles and not evidence of
model improvement. The next planned boundary is a gold-independent,
passage-level candidate index with frozen tokenizer, passage size, overlap,
corpus hash, and candidate cap. It must clear its registered offline recall gate
before any paid fresh-slice comparison is allowed.

## Contracts and audit boundary

- `Task`, `Plan`, `Evidence`, `Claim`, and `Citation` are validated Pydantic models.
- Every provider call produces a `TraceEvent` with provider/model, token usage, estimated cost, latency, and stage outcome.
- Per-run budget limits cap LLM calls and output tokens, then stop subsequent work when provider-reported total tokens or cache-aware estimated cost exceed the registered bound. Because an OpenAI-compatible API does not expose a pre-call exact input-token cap, observed total-token and cost limits are checked after each response and may overshoot by that response; the trace records this explicitly.
- `HarnessConfig` validates the provider, pricing, and run-budget settings. Keys are not configuration values: `api_key_env` names the only permitted credential source.
- `Citation` explicitly connects a report marker to evidence and claim IDs.
- The collector has a small `EvidenceCollector` interface. The MVP implementation is a deterministic local corpus collector that ranks each query independently and fills top-k round-robin, so one obligation cannot consume every evidence slot. It avoids untracked network behavior in smoke tests.
- `LiveWebCollector` preserves the same interface. Repository-shaped queries prefer GitHub's public repository API; general search falls back to DuckDuckGo Lite and Bing RSS. Each search/fetch is a trace event. Fetching rejects local/reserved network targets, caps response bytes and time, and extracts bounded readable text. This is a local CLI safety boundary, not a hardened crawler sandbox.
- `PrimarySourceResearchPipeline` is the live B1 policy. It asks for one primary-source query per named entity and stores decision context so recommendations do not ignore already-implemented project capabilities. It does not add another LLM call.
- `public_benchmark.py` fetches only the selected rows from a pinned LiveDRBench revision, passes questions but never gold answers into generation, writes official-style predictions, and records per-task usage even when a later parser stage fails. Its local exact scorer checks main-claim strings and official shape compatibility; the official judge remains a separate, unrun boundary.
- Human review uses a static workspace generated from the blind packet only. Draft state stays in browser local storage or reviewer-exported JSON; Python validates a complete submission and records its hash before a separate command can read the answer key.
- Optional reviewer translation is a packet-bound presentation layer. `ReviewTranslationBundle` records the source hash, exact source/translation pairs, provider/model, token/cost/latency trace, and exposes the untouched English source in the workspace. It never reads the answer key and is excluded from harness evaluation budgets.

## Extension boundaries, not implemented behavior

| Boundary | MVP implementation | Future responsibility |
| --- | --- | --- |
| `LLMProvider` | fake and chat-completions provider | structured-output retries, provider-specific adapters |
| `EvidenceCollector` | local corpus plus bounded no-key web collector | stable search API, richer parsers, provenance policy |
| `BaselineResearchPipeline` | linear B0/B1/B2 and live primary-source B1 | bounded re-planning, then only evidenced DAG needs |
| `RunState.trace` | append-only LLM/search/fetch events and budget totals | optional trace exporters |
| claim ledger | atomic claims plus B2 obligation/evidence-debt links | entailment critic and repair queue |
| benchmark contract | controlled pilot, batch aggregation, blinded semantic review | larger registered evaluation sets |
| public benchmark adapter | pinned five-task LiveDRBench preview compatibility pilot | stable search comparison and official evaluator integration |
| reviewer presentation | English source plus validated `zh-CN` reading aid | additional packet-bound locales |
| benchmark agent loop | pinned Pi adapter with empty system prompt | evidence-debt controller policies owned by this project |
| BrowseComp retrieval | pinned BM25 anchors plus query-aware dense leads and bounded evidence opens | only replay-justified retrievers or rerankers |
| BrowseComp control | strict global/phase limits, non-thinking answer compilation, and an eight-search governor | evidence-debt marginal-value stopping after larger bad-case clustering |
| BrowseComp reliability | alternating-order paired repeats with hash-bound aggregation | preregistered larger-slice confidence intervals and official judging |
| BrowseComp evaluation | two-GPU BF16 official path plus calibrated persistent-service diagnostics | official slice score, then full development and sealed submission |

Do not add DAG fan-out, critic loops, or automatic re-planning until a saved bad case makes their decision boundary measurable.

The project-specific control hypothesis and its simpler baselines are documented in `problem_statement.md`; the first diagnostic suite is documented in `pilot_design.md`. Neither document claims a measured improvement.
