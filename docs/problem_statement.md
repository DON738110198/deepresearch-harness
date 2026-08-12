# Problem Statement and Design Position

## User problem

The target user is an engineer or technical decision-maker asking a multi-constraint research question. A useful answer must cover the decision obligations, connect each material claim to eligible evidence, expose conflicts, and stop within a fixed budget. A fluent report that misses one constraint or attaches a nearby-but-insufficient citation is a failure.

## Why a harness

The base model and tools are held fixed. The harness is justified only if failures arise from execution policy: incomplete query decomposition, untracked evidence gaps, premature stopping, or claim/citation assembly. If a single search followed by a single write call performs just as well under the same budget, the extra harness is not justified.

## Simpler alternatives first

| Variant | Mechanism | Question it answers | Status |
| --- | --- | --- | --- |
| B0 Search-Write | Search directly from the user question, then write | Is planning needed at all? | planned |
| B1 Plan-Search-Ledger-Write | Decompose queries, collect, normalize claims and citations, then write | Does explicit planning and a ledger fix the observed failure? | implemented baseline |
| B2 Evidence-Debt | Track answer obligations and continue only for unresolved evidence gaps | Does evidence-aware control add value beyond B1? | hypothesis, not implemented |

B2 is permitted only after B0/B1 traces show an identifiable coverage, support, conflict, or stopping failure.

## DeerFlow as a reference, not a template

The adjacent DeerFlow checkout was inspected at commit `4e449385516c03b6b279ced004ff0ada493d56ef`. Its general harness composes planning, subagent limits, loop detection, token budgets, and clarification through middleware; its lead-agent prompt also requires inline citations for web research.

This project adopts the useful engineering principles: explicit runtime boundaries, durable state, tool/provider abstraction, budget traces, and cited output. It deliberately does not reproduce DeerFlow's frontend, sandbox, memory, skill system, generic tool loop, or multi-agent runtime.

Our narrower research question is:

> With model, corpus/search tool, and budget fixed, can an explicit answer-obligation and evidence-gap policy improve end-to-end research completeness and citation support over simpler Search-Write and Plan-Search-Write baselines?

The proposed distinction is not "more agents." It is a testable stopping and control rule: stop when required evidence obligations are resolved or the budget is exhausted, rather than merely when the model emits no further tool call.

## Falsifiable hypotheses

- H0: B0 and B1 have no meaningful difference on the pilot; explicit planning is unnecessary for this task set.
- H1: B1 improves obligation-level evidence recall over B0 under the same model, corpus, and maximum budget.
- H2: If B1 still stops with unresolved obligations, B2 reduces those gaps without increasing unsupported claims under a matched budget.
- H3: If conflicts are already handled by B1, conflict-specific search or critic logic is unnecessary.

No hypothesis has been tested yet. Passing unit tests or the fake-provider smoke does not count as experimental evidence.

## Complexity gate

Add a mechanism only when the preceding baseline supplies all four items:

1. saved failing task and trace;
2. named failure category;
3. causal hypothesis for the smallest change;
4. metric and matched-budget acceptance condition.

Research DAGs and subagents are last-mile options for demonstrated independent-branch latency or coverage problems, not default architecture.
