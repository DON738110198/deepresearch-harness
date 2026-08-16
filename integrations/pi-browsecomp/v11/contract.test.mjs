import test from "node:test";
import assert from "node:assert/strict";

import {
  ADAPTER_VERSION,
  EVIDENCE_DEBT_AUDIT_TOKENS,
  EVIDENCE_DEBT_COMPILATION_TOKENS,
  MAXIMUM_EXPLORATION_SEARCH_CALLS,
  MAXIMUM_REPAIR_SEARCH_CALLS,
  MAXIMUM_SEARCH_CALLS,
  answerReserveAllocation,
  parseResolvedDebtAudit,
  reserveExplorationSearchCall,
  reserveRepairSearchCall,
  reserveSearchCall,
  validateDebtRepairArguments,
  validateDisclosureSearchResponse,
  validateOpenEvidenceResponse,
  validateRequest,
  validateSearchResults,
} from "./contract.mjs";


test("v11 admits an explicitly pinned 20-result local search contract", () => {
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

  assert.equal(ADAPTER_VERSION, "pi-browsecomp-v11");
  assert.equal(MAXIMUM_SEARCH_CALLS, 8);
  assert.equal(request.search.max_results, 20);
  assert.equal(validateSearchResults(wideResults, { maxResults: 20 }).length, 20);
  assert.throws(() => validateSearchResults(wideResults), /at most 5/);
});


test("v11 retains a hard global budget of eight server searches", () => {
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


test("v11 partitions six exploration and two repair searches", () => {
  let reserved = 0;
  for (let index = 1; index <= MAXIMUM_EXPLORATION_SEARCH_CALLS; index += 1) {
    const result = reserveExplorationSearchCall(reserved);
    assert.equal(result.allowed, true);
    reserved = result.reserved_search_calls;
    assert.equal(result.phase_exhausted, index === MAXIMUM_EXPLORATION_SEARCH_CALLS);
  }
  assert.equal(reserveExplorationSearchCall(reserved).allowed, false);

  let repairs = 0;
  for (let index = 1; index <= MAXIMUM_REPAIR_SEARCH_CALLS; index += 1) {
    const result = reserveRepairSearchCall(reserved, repairs);
    assert.equal(result.allowed, true);
    reserved = result.reserved_search_calls;
    repairs = result.repair_search_calls;
  }
  assert.equal(reserved, MAXIMUM_SEARCH_CALLS);
  assert.equal(reserveRepairSearchCall(reserved, repairs).allowed, false);
});


test("v11 freezes the 8000 plus 512 plus 1488 output allocation", () => {
  const allocation = answerReserveAllocation("answer_reserve_nonthinking_v0");
  assert.deepEqual(allocation, {
    exploration: 8000,
    audit: EVIDENCE_DEBT_AUDIT_TOKENS,
    compilation: EVIDENCE_DEBT_COMPILATION_TOKENS,
  });
  assert.equal(
    allocation.exploration + allocation.audit + allocation.compilation,
    10000,
  );
});


test("v11 debt checkpoint accepts only typed repair or resolved payloads", () => {
  const repair = validateDebtRepairArguments({
    obligation_id: "requested_year",
    claim: "The foundation was established in 2002.",
    query: "Haile Gebrselassie foundation founded year",
    supporting_docids: ["10"],
  });
  assert.equal(repair.obligation_id, "requested_year");
  assert.throws(
    () => validateDebtRepairArguments({ ...repair, unexpected: true }),
    /unknown fields/,
  );

  const resolved = parseResolvedDebtAudit(JSON.stringify({
    status: "resolved",
    candidate_answer: "Ewuare II",
    supporting_docids: ["20"],
  }));
  assert.equal(resolved.candidate_answer, "Ewuare II");
  assert.equal(parseResolvedDebtAudit("not json"), null);
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
