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
- `docs/architecture.md`: current boundaries and next-stage extension points.
- `docs/experiment_protocol.md`: fair-comparison and bad-case evaluation protocol.

## Next implementation order

1. Build a fixed 20-50 question evaluation set with evidence relevance, citation support, completeness, and cost fields; acceptance: a baseline run is reproducible from one command.
2. Categorize baseline bad cases (missed evidence, unsupported claim, redundant search, budget exhaustion); acceptance: each category has saved input, expected behavior, and a regression test.
3. Add budget-aware re-planning only for a measured failure category; acceptance: same model, tool, corpus, and token/fee budget, with comparison artifacts per run.
4. Add Critic-Repair for unsupported claims before considering a Research DAG; acceptance: it reduces a pre-defined unsupported-citation rate without exceeding the matched budget.
5. Add a Research DAG only when independent evidence branches improve a measured coverage failure; acceptance: branch-level traces and a fair budget-matched baseline comparison.

