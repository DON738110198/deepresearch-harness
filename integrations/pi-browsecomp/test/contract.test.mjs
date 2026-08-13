import test from "node:test";
import assert from "node:assert/strict";
import {
  ANSWER_COMPILER_PROMPT,
  applySamplingPolicy,
  FIRST_TOOL_DEADLINE_PROMPT,
  FIRST_TOOL_DEADLINE_TOKENS,
  TOOL_BOOTSTRAP_EXPLORATION_TOKENS,
  TOOL_BOOTSTRAP_TOKENS,
  RARE_ANCHOR_BOOTSTRAP_TOKENS,
  RARE_ANCHOR_EXPLORATION_TOKENS,
  answerReserveAllocation,
  classifyRunOutcome,
  clampProviderOutput,
  formatBenchmarkPrompt,
  formatRareAnchorBootstrapPrompt,
  formatToolBootstrapPrompt,
  hasRequiredAnswerSchema,
  remainingProviderOutput,
  sha256,
  shouldInvokeAnswerCompiler,
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
  assert.equal(validateRequest(request()).control_policy, "standard");
  assert.equal(
    validateRequest({ ...request(), control_policy: "answer_reserve_v0" }).model,
    "deepseek-v4-flash",
  );
  assert.deepEqual(answerReserveAllocation("answer_reserve_v1"), {
    exploration: 6000,
    compilation: 4000,
  });
  assert.deepEqual(answerReserveAllocation("answer_reserve_nonthinking_v0"), {
    exploration: 8000,
    compilation: 2000,
  });
  assert.deepEqual(answerReserveAllocation("first_tool_deadline_v0"), {
    exploration: 8000,
    compilation: 2000,
  });
  assert.deepEqual(answerReserveAllocation("tool_bootstrap_v0"), {
    bootstrap: 512,
    exploration: 7488,
    compilation: 2000,
  });
  assert.equal(TOOL_BOOTSTRAP_TOKENS, 512);
  assert.equal(TOOL_BOOTSTRAP_EXPLORATION_TOKENS, 7488);
  assert.deepEqual(answerReserveAllocation("rare_anchor_portfolio_v0"), {
    bootstrap: 1024,
    exploration: 6976,
    compilation: 2000,
  });
  assert.equal(RARE_ANCHOR_BOOTSTRAP_TOKENS, 1024);
  assert.equal(RARE_ANCHOR_EXPLORATION_TOKENS, 6976);
  assert.equal(FIRST_TOOL_DEADLINE_TOKENS, 512);
  assert.equal(
    sha256(FIRST_TOOL_DEADLINE_PROMPT),
    "3632980cad37d8e0f578ac8f38e6e7624ab2a841459cc9b1e91a1e4ffc70c8d7",
  );
  assert.throws(
    () => validateRequest({ ...request(), control_policy: "unregistered" }),
    /control_policy/,
  );
});

test("formats a deterministic rare-anchor portfolio prompt", () => {
  const prompt = formatRareAnchorBootstrapPrompt("Who is the scientist?");
  assert.match(prompt, /exactly three search tool calls/);
  assert.match(prompt, /rare chronology/);
  assert.equal(
    sha256(prompt),
    "6224db89eac7f9d41b26e96bc6b5c5da8c4b10d774fd674ec7eb14a576bb9add",
  );
});

test("formats a deterministic one-search bootstrap prompt", () => {
  const prompt = formatToolBootstrapPrompt("Who is the scientist?");
  assert.match(prompt, /search tool exactly once/);
  assert.match(prompt, /Question: Who is the scientist\?/);
  assert.equal(
    sha256(prompt),
    "f429186c8942f9e82834e07b3615bbd20aa3fc70037dc11534edfae0642e8afe",
  );
});

test("clamps each provider turn to the remaining global output budget", () => {
  assert.equal(clampProviderOutput({ max_tokens: 10000 }, 417).max_tokens, 417);
  assert.equal(
    clampProviderOutput({ max_completion_tokens: 10000 }, 23).max_tokens,
    23,
  );
  assert.equal(
    Object.hasOwn(clampProviderOutput({ max_completion_tokens: 10000 }, 23), "max_completion_tokens"),
    false,
  );
});

test("pins non-thinking temperature and omits unsupported thinking sampling", () => {
  assert.deepEqual(
    applySamplingPolicy({
      thinking: { type: "disabled" },
      temperature: 1,
      top_p: 0.9,
    }),
    { thinking: { type: "disabled" }, temperature: 0 },
  );
  assert.deepEqual(
    applySamplingPolicy({ thinking: { type: "enabled" }, temperature: 0.2 }),
    { thinking: { type: "enabled" } },
  );
});

test("reserves a fixed 2000-token answer-compilation phase", () => {
  assert.equal(
    remainingProviderOutput({
      controlPolicy: "answer_reserve_v0",
      phase: "exploration",
      maxOutputTokens: 10000,
      totalOutputTokens: 7600,
      phaseOutputTokens: 7600,
    }),
    400,
  );
  assert.equal(
    remainingProviderOutput({
      controlPolicy: "answer_reserve_v0",
      phase: "compilation",
      maxOutputTokens: 10000,
      totalOutputTokens: 8100,
      phaseOutputTokens: 0,
    }),
    1900,
  );
  assert.equal(
    remainingProviderOutput({
      controlPolicy: "answer_reserve_v1",
      phase: "compilation",
      maxOutputTokens: 10000,
      totalOutputTokens: 6000,
      phaseOutputTokens: 0,
    }),
    4000,
  );
  assert.equal(
    sha256(ANSWER_COMPILER_PROMPT),
    "3e86775e1455eadc50433cf24b93f17585a79c22c0b548cdcd08dc8de5547861",
  );
});

test("invokes the compiler only for an unfinished or malformed reserved run", () => {
  const complete = [
    "Explanation: supported by [doc-1]",
    "Exact Answer: Q7-BENCH",
    "Confidence: 80%",
  ].join("\n");
  assert.equal(hasRequiredAnswerSchema(complete), true);
  assert.equal(
    shouldInvokeAnswerCompiler({
      controlPolicy: "answer_reserve_v0",
      answerText: complete,
      finalHasToolCall: false,
      totalOutputTokens: 5000,
      maxOutputTokens: 10000,
    }),
    false,
  );
  assert.equal(
    shouldInvokeAnswerCompiler({
      controlPolicy: "answer_reserve_v0",
      answerText: "searching",
      finalHasToolCall: true,
      totalOutputTokens: 8100,
      maxOutputTokens: 10000,
    }),
    true,
  );
  assert.equal(
    shouldInvokeAnswerCompiler({
      controlPolicy: "answer_reserve_v0",
      answerText: "malformed terminal answer",
      finalHasToolCall: false,
      totalOutputTokens: 6000,
      maxOutputTokens: 10000,
    }),
    true,
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
  assert.deepEqual(
    classifyRunOutcome({
      answerText: "partial answer",
      finalHasToolCall: false,
      finalStopReason: "error",
      outputBudgetOvershootTokens: 0,
      failureReason: "provider_or_runtime_error:fixture",
    }),
    {
      status: "failed",
      stopReason: "provider_or_runtime_error:fixture",
    },
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
