# Pilot Design

## Purpose

`benchmarks/pilot_v0` is a controlled ten-task diagnostic suite. It is not a benchmark result and not evidence of production research quality. Its synthetic corpus removes web freshness, crawler failures, and search-engine drift so the first experiments can isolate harness behavior.

Each task represents an AI-infrastructure decision memo with three answer obligations. Two tasks target each anticipated failure category:

- coverage gaps;
- citation mismatch;
- conflict omission;
- redundant search;
- budget prioritization.

The suite records the target user, decision context, forbidden shortcuts, required obligations, eligible evidence IDs, and counter-evidence IDs. Synthetic evidence is visibly marked with `synthetic: true` and uses the reserved `.test` domain.

## Measurement layers

Deterministic scoring can calculate evidence-ID recall, obligation-level retrieval recall, and citation referential integrity from `state.json`. It cannot establish semantic entailment.

Human annotation is required for:

- whether the report actually covers an obligation;
- whether evidence supports the exact claim strength;
- whether a material claim is unsupported;
- whether conflicting evidence is acknowledged and reconciled.

LLM-as-a-Judge is intentionally excluded from pilot v0. It adds another model, prompt, cost, and bias before the basic rubric is stable.

## Experiment matrix

| Control | B0 | B1 | B2 |
| --- | --- | --- | --- |
| Model/provider | fixed | same | same |
| Corpus snapshot | `pilot_v0/corpus.json` | same | same |
| Maximum token budget | fixed before launch | same | same |
| Maximum fee budget | separately fixed before launch | same | same |
| Retrieval implementation and top-k | fixed | same | same |
| Report rubric | fixed | same | same |
| Additional control | none | plan + ledger | obligation/evidence-debt policy only |

Token-matched and cost-matched experiments are separate result tables. Actual consumption is reported alongside caps. A run that changes model, corpus, tool, prompt policy, or retry allowance is an ablation, not a fair comparison.

## Stage acceptance

1. **Pilot schema:** ten tasks load, every gold/counter evidence ID exists, and failure categories are balanced two each.
2. **B0 implementation:** one-command batch run persists the same state/trace contract as B1.
3. **Baseline run:** B0 and B1 finish all ten tasks under frozen configurations; raw reports are annotated blind to variant.
4. **Bad-case decision:** select one mechanism only if at least two saved failures share a causal category that the mechanism directly addresses.
5. **B2 acceptance:** improve the targeted metric under both a declared token cap and a separate declared fee cap, without worsening citation support or completion rate beyond a pre-registered tolerance.

The exact budgets and tolerances remain **planned** until the real provider smoke establishes supported token controls and pricing metadata. They must be written into a versioned experiment manifest before the first comparison run, never chosen after seeing results.
