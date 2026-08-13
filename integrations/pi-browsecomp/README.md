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
