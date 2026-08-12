# Offline Smoke Result

Recorded on 2026-08-12 with:

```powershell
python -m deepresearch_harness.cli demo --output-dir runs\smoke
python -m pytest
```

Result: `succeeded` for run `4142c01ac6594d0f978ee8ed6272813b`; the run emitted three collected evidence records, three ledger claims, three report citations, and four trace stages (`plan`, `collect`, `ledger`, `write`). The persisted audit artifacts are `runs/smoke/4142c01ac6594d0f978ee8ed6272813b/state.json` and `report.md`.

The same verification run reported `3 passed in 0.33s`. This is an offline structural smoke check using `deterministic-fake-v1`; it is not a research-quality benchmark and establishes no model or harness outcome metric.
