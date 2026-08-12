# Experiment Protocol

## Claim boundary

The comparison target is harness-level end-to-end task performance. Keep model weights frozen and do not describe harness gains as a base-model capability gain.

## Required controls

For a baseline versus a harness variant, fix:

- model identifier, provider endpoint, decoding parameters, and system prompt policy;
- tool implementations, tool versions, corpus snapshot or web-search date window;
- question set, evidence eligibility rules, scoring rubric, and retry policy;
- concurrency, timeout policy, and random seed where supported.

Run each configuration under two separately reported budgets:

1. **Token-matched:** same maximum input plus output token budget per task.
2. **Cost-matched:** same maximum provider/tool monetary budget per task, with the price table version recorded.

Do not substitute a stronger model, fresher search index, or different tool allowance in one condition. If any control differs, label the result as an ablation or exploratory result rather than a fair head-to-head comparison.

## Per-run artifacts

Persist the question ID, config digest, corpus/tool version, model and provider, run state, all trace events, total tokens, estimated cost, latency, selected evidence, ledger, report, and scoring output. Redact credentials and sensitive raw data before sharing.

## Planned metrics

No outcome metrics have been run in this repository. The following are **planned**:

- citation support rate: claims whose cited evidence supports the claim;
- evidence recall / coverage against a curated reference set;
- report completeness and factuality under a fixed rubric;
- median and tail latency, total tokens, and cost per completed task;
- completion rate and recoverable-failure rate.

## Bad-case-driven iteration

Start by saving a concrete failure with its inputs and trace. State `problem -> evidence -> hypothesis -> change -> metric -> conclusion`. Add only the smallest mechanism that tests that hypothesis, then add a focused regression test. A more complex orchestration is not accepted merely because it produces a more fluent report.

