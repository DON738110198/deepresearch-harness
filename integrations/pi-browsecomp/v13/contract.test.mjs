import test from "node:test";
import assert from "node:assert/strict";

import {
  ADAPTER_VERSION,
  ANSWER_FIRST_COMPILER_REQUEST_TOKENS,
  MAXIMUM_EXPLORATION_SEARCH_CALLS,
  MAXIMUM_SEARCH_CALLS,
  auditAnswerFirstDraft,
  reserveExplorationSearchCall,
  reserveSearchCall,
  validateDisclosureSearchResponse,
  validateOpenEvidenceResponse,
  validateRequest,
  validateSearchResults,
} from "./contract.mjs";


test("v13 admits an explicitly pinned 20-result local search contract", () => {
  const wideResults = Array.from({ length: 20 }, (_, index) => ({
    docid: `doc-${index}`,
    score: 1.0,
    snippet: "x",
  }));
  const request = validateRequest({
    schema_version: "pi-browsecomp-request-v0",
    run_id: "fixture-run",
    query_id: "fixture-query",
    question: "fixture?",
    model: "deepseek-v4-flash",
    thinking_level: "high",
    max_output_tokens: 10000,
    max_iterations: 100,
    control_policy: "answer_reserve_nonthinking_v0",
    search: {
      kind: "fixture",
      max_results: 20,
      results: wideResults,
    },
  });

  assert.equal(ADAPTER_VERSION, "pi-browsecomp-v13");
  assert.equal(MAXIMUM_SEARCH_CALLS, 8);
  assert.equal(ANSWER_FIRST_COMPILER_REQUEST_TOKENS, 1000);
  assert.equal(request.search.max_results, 20);
  assert.equal(validateSearchResults(wideResults, { maxResults: 20 }).length, 20);
  assert.throws(() => validateSearchResults(wideResults), /at most 5/);
});


test("v13 global research budget admits exactly eight server searches", () => {
  let reserved = 0;
  for (let index = 1; index <= MAXIMUM_SEARCH_CALLS; index += 1) {
    const result = reserveSearchCall(reserved);
    assert.equal(result.allowed, true);
    reserved = result.reserved_search_calls;
    assert.equal(result.exhausted, index === MAXIMUM_SEARCH_CALLS);
  }
  const blocked = reserveSearchCall(reserved);
  assert.equal(blocked.allowed, false);
  assert.equal(blocked.reserved_search_calls, MAXIMUM_SEARCH_CALLS);
  assert.equal(blocked.stop_reason, "search_call_limit_reached:8");
});


test("v13 stops ordinary exploration after seven searches", () => {
  let reserved = 0;
  for (let index = 1; index <= MAXIMUM_EXPLORATION_SEARCH_CALLS; index += 1) {
    const result = reserveExplorationSearchCall(reserved);
    assert.equal(result.allowed, true);
    reserved = result.reserved_search_calls;
    assert.equal(result.exploration_exhausted, index === MAXIMUM_EXPLORATION_SEARCH_CALLS);
  }
  const blocked = reserveExplorationSearchCall(reserved);
  assert.equal(blocked.allowed, false);
  assert.equal(blocked.reserved_search_calls, 7);
  assert.equal(blocked.stop_reason, "exploration_search_reserve_reached:7");
});


test("answer-first audit opens only explicit or low-confidence unsupported debt", () => {
  const open = auditAnswerFirstDraft({
    question: "In what year was the foundation established?",
    answerText: [
      "Explanation: The year is not fully confirmed by the cited biography [1].",
      "Exact Answer: 2002",
      "Confidence: 45%",
    ].join("\n"),
    evidence: [{ docid: "1", text: "The athlete won a race and received an honor." }],
  });
  assert.equal(open.audit_status, "open");
  assert.equal(open.repair_queries.length, 1);

  const supported = auditAnswerFirstDraft({
    question: "In what year was the foundation established?",
    answerText: [
      "Explanation: The foundation was established in 2011 [2].",
      "Exact Answer: 2011",
      "Confidence: 90%",
    ].join("\n"),
    evidence: [{ docid: "2", text: "The foundation was established in 2011." }],
  });
  assert.equal(supported.audit_status, "supported");
  assert.deepEqual(supported.repair_queries, []);

  const abstained = auditAnswerFirstDraft({
    question: "Who is the person?",
    answerText: [
      "Explanation: The available evidence suggests this answer [3].",
      "Exact Answer: Ada Example",
      "Confidence: 90%",
    ].join("\n"),
    evidence: [{ docid: "3", text: "A different person is discussed." }],
  });
  assert.equal(abstained.audit_status, "no_repair_trigger");
  assert.deepEqual(abstained.repair_queries, []);
});


test("progressive disclosure responses are strict and budget-auditable", () => {
  const state = {
    run_id: "run-1",
    search_calls: 1,
    open_attempts: 0,
    successful_open_calls: 0,
    seen_docids: ["a", "b"],
    eligible_open_docids: ["b"],
    opened_docids: ["a"],
    cumulative_ingress_tokens: 20,
    search_ingress_tokens: 20,
    open_ingress_tokens: 0,
    remaining_ingress_tokens: 492,
    remaining_search_ingress_tokens: 476,
    remaining_open_ingress_tokens: 16,
  };
  const response = validateDisclosureSearchResponse({
    results: [
      { docid: "a", score: 1, snippet: "anchor" },
      { docid: "b", score: 0.9, snippet: "lead" },
    ],
    disclosure: {
      search_index: 1,
      anchors_returned: 1,
      leads_returned: 1,
      within_channel_duplicate_slots: 0,
      prior_context_duplicate_slots: 0,
      cross_channel_duplicate_slots: 0,
      new_ingress_tokens: 20,
      cumulative_ingress_tokens: 20,
      remaining_ingress_tokens: 492,
      remaining_search_ingress_tokens: 476,
      remaining_open_ingress_tokens: 16,
      ingress_budget_exhausted: false,
    },
    state,
    latency_ms: 3,
  });
  assert.equal(response.state.eligible_open_docids[0], "b");

  const opened = validateOpenEvidenceResponse({
    result: {
      docid: "b",
      outcome: "opened",
      content: "full evidence",
      ingress_tokens: 12,
      cumulative_ingress_tokens: 32,
      remaining_ingress_tokens: 480,
      remaining_search_ingress_tokens: 476,
      remaining_open_ingress_tokens: 4,
    },
    state: {
      ...state,
      open_attempts: 1,
      successful_open_calls: 1,
      eligible_open_docids: [],
      opened_docids: ["a", "b"],
      cumulative_ingress_tokens: 32,
      search_ingress_tokens: 20,
      open_ingress_tokens: 12,
      remaining_ingress_tokens: 480,
      remaining_search_ingress_tokens: 476,
      remaining_open_ingress_tokens: 4,
    },
    latency_ms: 1,
  });
  assert.equal(opened.result.outcome, "opened");
});
