# Pi BrowseComp Adapter

This is a thin, pinned execution adapter for the BrowseComp-Plus standard
search-only baseline. Pi supplies the provider/tool loop; the benchmark prompt,
search contract, leakage controls, trace schema, and later harness mechanisms
remain owned by this project.

The adapter deliberately disables Pi extensions, skills, prompt templates,
themes, context files, and its coding-agent system prompt. The resulting system
prompt is the empty string. It accepts only `deepseek-v4-flash` or
`deepseek-v4-pro`, reads `DEEPSEEK_API_KEY` from the environment, and rejects
inline credentials.

The adapter follows the reference loop's budget order: it caps every provider
request at the remaining part of the 10,000-token global output budget,
executes tool calls already emitted by that response, and only then stops
before another provider request. Pi compaction is disabled so hidden context
rewrites cannot change the benchmark contract. The run trace records every
requested/applied provider limit and the provider-reported usage. A provider
can still report more generated tokens than requested (notably reasoning
tokens); the exact overshoot is retained and never silently treated as budget
compliant. If that terminal response already contains a final answer and no
tool call, it remains scoreable as `succeeded`; otherwise it is
`budget_exhausted`.

Adapter v1 explicitly sends DeepSeek's documented `max_tokens` field. Pi
0.84.1 otherwise auto-detects `max_completion_tokens` for this provider; that
field was accepted but did not cap generated reasoning in the observed V4
calls. Every v1 provider-attempt trace therefore records
`output_limit_field=max_tokens`, and Python rejects a v1 trace without it.

Adapter v6 also records the sampling boundary. DeepSeek documents temperature,
top-p, and penalty fields as unsupported in thinking mode, so the adapter omits
them there rather than pretending they control randomness. Non-thinking phases
use `temperature=0`. Because this API contract exposes no seed, effectiveness
experiments require repeated runs and report dispersion instead of selecting
the best trace.

The `answer_reserve_nonthinking_v0` candidate holds the registered total at
10,000 while limiting high-thinking exploration to 8,000 output tokens. If the
exploration turn has not already produced the complete benchmark schema, it
disables Search and thinking, then spends at most the remaining 2,000-token
phase allowance on a fixed answer-compiler prompt. Provider-reported overshoot
remains an explicit protocol violation; it is never treated as a Token-matched
result. Traces bind every request to its phase and observed thinking type.

`first_tool_deadline_v0`, `tool_bootstrap_v0`, and
`rare_anchor_portfolio_v0` are retained experimental controls for a saved
zero-search bad case. They respectively test a capped first turn, a deterministic
non-thinking tool phase, and a three-query constraint portfolio. All three
failed the relevance/accuracy diagnostic and are not promoted candidate
policies. Their prompts, phase allocations, and negative traces remain auditable
so the failed hypotheses are not silently discarded.

`constraint_portfolio_v1` is the adapter-v7, development-only successor. It
uses a general typed portfolio of rare anchors, chronology/relations, and an
orthogonal clue pair rather than a prompt written around one saved question.
Its fresh-query probe and stop gates are pinned in
`benchmarks/browsecomp_plus_v0/query_compiler_v1_gates.json`; it is a planned
candidate until those gates are actually run.

Transient provider retries are fixed at three. Every attempted provider
request receives its own indexed limit record, so a retry cannot disappear from
the audit trail.

Install and test with the bundled Node.js 24 runtime or another supported Node
runtime:

```powershell
npm ci
npm test
node src\runner.mjs path\to\ignored-request.json > path\to\ignored-run.json
```

Requests and runs use versioned JSON contracts. Benchmark question text and raw
traces belong under the ignored `runs/` directory.
