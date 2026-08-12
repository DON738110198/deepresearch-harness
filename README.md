# Training-Free Deep Research Harness

A small, auditable baseline for improving end-to-end Deep Research task execution **without changing model weights**. It calls an OpenAI-compatible chat-completions API and evaluates harness behavior, not intrinsic model capability.

The first MVP is deliberately narrow:

```text
Question -> Plan -> Search / collect evidence -> Claim-Evidence Ledger -> cited report
```

Every run persists its normalized input, plan, evidence, claims, citations, status transitions, and per-stage token/cost/latency trace. The default `fake` provider and fixture corpus are deterministic, so the project is runnable offline with no API key.

## Quick start

```powershell
cd G:\Obsidian\Study\AI_infra_interm\training-free-deepresearch-harness
python -m pip install -e ".[dev]"
python -m deepresearch_harness.cli demo --output-dir runs\smoke
python -m deepresearch_harness.cli validate-pilot --suite benchmarks\pilot_v0\tasks.json
python -m pytest
```

The demo writes `runs/smoke/<run-id>/state.json` and `report.md`. `state.json` is the audit artifact; it includes prompts only as stage metadata, not credentials.

## Real API run

Copy `config.example.json` to a local config file, set only the named environment variable, then run:

```powershell
$env:OPENAI_API_KEY = "..."
python -m deepresearch_harness.cli run `
  --question "What evidence supports a phased rollout?" `
  --corpus examples/offline_corpus.json `
  --config config.local.json `
  --output-dir runs
```

No API key is accepted through CLI flags or configuration values. `api_key_env` names the environment variable to read at runtime.

## Scope and limits

- This is a single-agent Plan-Search-Write baseline, not a multi-agent research system.
- The MVP collector searches a supplied JSON evidence corpus. It does not claim live-web coverage or source freshness.
- The fake provider is for deterministic pipeline tests, not quality evaluation.
- Report citations point to collected evidence IDs; source quality and claim entailment still require evaluation.
- All outcome metrics are **planned** until measured with the protocol in `docs/experiment_protocol.md`.

## Repository map

- `src/deepresearch_harness/contracts.py`: Pydantic run-state contracts.
- `src/deepresearch_harness/providers.py`: fake and OpenAI-compatible providers.
- `src/deepresearch_harness/pipeline.py`: baseline orchestration and persistence.
- `src/deepresearch_harness/benchmark.py`: pilot contracts, asset validation, and scoring boundaries.
- `benchmarks/pilot_v0/`: ten controlled diagnostic tasks and a synthetic corpus.
- `docs/problem_statement.md`: problem-first design position and falsifiable hypotheses.
- `docs/pilot_design.md`: B0/B1/B2 comparison and stage gates.
- `docs/architecture.md`: current boundaries and next-stage extension points.
- `docs/experiment_protocol.md`: fair-comparison and bad-case evaluation protocol.

## Next implementation order

1. Implement B0 Search-Write against the same state and trace contracts; acceptance: one command runs B0 or B1 without changing corpus or provider.
2. Add a versioned experiment manifest with enforced token and fee caps; acceptance: a run stops with an explicit budget reason rather than silently exceeding its cap.
3. Run B0 and B1 on the ten-task controlled pilot and annotate reports blind to variant; acceptance: every task has raw state, report, config digest, and scoring record.
4. Select one repeated bad case and implement only its smallest causal fix; Evidence-Debt, Critic-Repair, re-planning, or DAG execution remain hypotheses until selected by evidence.
5. Expand to a 20-50 task external-source evaluation set only after the controlled pilot's contracts and annotation rubric are stable.
