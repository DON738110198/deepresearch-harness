import test from "node:test";
import assert from "node:assert/strict";

import {
  ADAPTER_VERSION,
  MAXIMUM_SEARCH_CALLS,
  reserveSearchCall,
  validateDisclosureSearchResponse,
  validateOpenEvidenceResponse,
  validateRequest,
  validateSearchResults,
} from "./contract.mjs";


test("v10 admits an explicitly pinned 20-result local search contract", () => {
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
    search: {
      kind: "fixture",
      max_results: 20,
      results: wideResults,
    },
  });

  assert.equal(ADAPTER_VERSION, "pi-browsecomp-v10");
  assert.equal(MAXIMUM_SEARCH_CALLS, 8);
  assert.equal(request.search.max_results, 20);
  assert.equal(validateSearchResults(wideResults, { maxResults: 20 }).length, 20);
  assert.throws(() => validateSearchResults(wideResults), /at most 5/);
});


test("v10 research budget admits exactly eight server searches", () => {
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
