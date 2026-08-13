import test from "node:test";
import assert from "node:assert/strict";
import {
  classifyRunOutcome,
  clampProviderOutput,
  formatBenchmarkPrompt,
  sha256,
  validateRequest,
  validateSearchResults,
} from "../src/contract.mjs";

function request() {
  return {
    schema_version: "pi-browsecomp-request-v0",
    run_id: "fixture-run",
    query_id: "fixture-query",
    question: "What is the fixture code?",
    model: "deepseek-v4-flash",
    thinking_level: "high",
    max_output_tokens: 10000,
    max_iterations: 100,
    search: {
      kind: "fixture",
      results: [{ docid: "doc-1", score: 1.0, snippet: "The code is Q7-BENCH." }],
    },
  };
}

test("formats the official search-only prompt", () => {
  const prompt = formatBenchmarkPrompt("Who is named in the source?");
  assert.match(prompt, /using the search tool provided/);
  assert.match(prompt, /Question: Who is named in the source\?/);
  assert.match(prompt, /Exact Answer:/);
  assert.equal(
    sha256(prompt),
    "4110be54126ad7b501f219585ea64b8b26ab9b1d589d873ca2d367b957e23075",
  );
});

test("accepts a pinned DeepSeek request", () => {
  assert.equal(validateRequest(request()).model, "deepseek-v4-flash");
});

test("clamps each provider turn to the remaining global output budget", () => {
  assert.equal(clampProviderOutput({ max_tokens: 10000 }, 417).max_tokens, 417);
  assert.equal(
    clampProviderOutput({ max_completion_tokens: 10000 }, 23).max_completion_tokens,
    23,
  );
});

test("keeps a terminal answer scoreable while auditing provider overshoot", () => {
  assert.deepEqual(
    classifyRunOutcome({
      answerText: "Exact Answer: Q7-BENCH",
      finalHasToolCall: false,
      budgetStop: "global_output_token_budget_exhausted:10000",
      outputBudgetOvershootTokens: 14,
    }),
    {
      status: "succeeded",
      stopReason: "completed_with_output_token_overshoot:14",
    },
  );
  assert.equal(
    classifyRunOutcome({
      answerText: "Let me search again",
      finalHasToolCall: true,
      budgetStop: "global_output_token_budget_exhausted:10000",
      outputBudgetOvershootTokens: 14,
    }).status,
    "budget_exhausted",
  );
  assert.equal(
    classifyRunOutcome({
      answerText: "provider error",
      finalHasToolCall: false,
      finalStopReason: "error",
      outputBudgetOvershootTokens: 0,
    }).status,
    "failed",
  );
});

test("rejects inline credentials and non-loopback search endpoints", () => {
  assert.throws(() => validateRequest({ ...request(), api_key: "secret" }), /unknown fields/);
  assert.throws(
    () => validateRequest({ ...request(), search: { kind: "http", url: "http://example.com", timeout_ms: 1000 } }),
    /loopback/,
  );
  assert.throws(
    () => validateRequest({ ...request(), search: { kind: "http", url: "https://127.0.0.1/search", timeout_ms: 1000 } }),
    /unauthenticated HTTP/,
  );
  assert.throws(
    () => validateSearchResults([{ docid: "doc", score: Number.NaN, snippet: "x" }]),
    /finite/,
  );
});
