import test from "node:test";
import assert from "node:assert/strict";

import {
  ADAPTER_VERSION,
  validateRequest,
  validateSearchResults,
} from "./contract.mjs";


test("v8 admits an explicitly pinned 20-result local search contract", () => {
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

  assert.equal(ADAPTER_VERSION, "pi-browsecomp-v8");
  assert.equal(request.search.max_results, 20);
  assert.equal(validateSearchResults(wideResults, { maxResults: 20 }).length, 20);
  assert.throws(() => validateSearchResults(wideResults), /at most 5/);
});
