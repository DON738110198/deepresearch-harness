import { createHash } from "node:crypto";

export const ADAPTER_VERSION = "pi-browsecomp-v0";
export const PI_VERSION = "0.84.1";

const REQUEST_KEYS = new Set([
  "schema_version",
  "run_id",
  "query_id",
  "question",
  "model",
  "thinking_level",
  "max_output_tokens",
  "max_iterations",
  "search",
]);

export function validateRequest(value) {
  requireObject(value, "request");
  rejectUnknownKeys(value, REQUEST_KEYS, "request");
  requireEqual(value.schema_version, "pi-browsecomp-request-v0", "schema_version");
  requireString(value.run_id, "run_id");
  requireString(value.query_id, "query_id");
  requireString(value.question, "question");
  if (!["deepseek-v4-flash", "deepseek-v4-pro"].includes(value.model)) {
    throw new Error("model must be deepseek-v4-flash or deepseek-v4-pro");
  }
  requireEqual(value.thinking_level, "high", "thinking_level");
  requireInteger(value.max_output_tokens, "max_output_tokens", 1, 10000);
  requireInteger(value.max_iterations, "max_iterations", 1, 100);
  validateSearch(value.search);
  return value;
}

export function formatBenchmarkPrompt(question) {
  requireString(question, "question");
  return [
    "You are a deep research agent. You need to answer the given question by interacting with a search engine, using the search tool provided. Please perform reasoning and use the tool step by step, in an interleaved manner. You may use the search tool multiple times.",
    "",
    `Question: ${question}`,
    "",
    "Your response should be in the following format:",
    "Explanation: {your explanation for your final answer. For this explanation section only, you should cite your evidence documents inline by enclosing their docids in square brackets [] at the end of sentences. For example, [20].}",
    "Exact Answer: {your succinct, final answer}",
    "Confidence: {your confidence score between 0% and 100% for your answer}",
  ].join("\n");
}

export function sha256(text) {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

export function clampProviderOutput(payload, remainingOutputTokens) {
  requireObject(payload, "provider payload");
  requireInteger(remainingOutputTokens, "remaining output tokens", 1, 10000);
  const bounded = { ...payload };
  if (Object.hasOwn(bounded, "max_completion_tokens")) {
    bounded.max_completion_tokens = Math.min(
      bounded.max_completion_tokens ?? remainingOutputTokens,
      remainingOutputTokens,
    );
  } else {
    bounded.max_tokens = Math.min(
      bounded.max_tokens ?? remainingOutputTokens,
      remainingOutputTokens,
    );
  }
  return bounded;
}

export function classifyRunOutcome({
  answerText,
  finalHasToolCall,
  finalStopReason,
  budgetStop,
  outputBudgetOvershootTokens,
}) {
  if (typeof answerText !== "string") throw new Error("answerText must be a string");
  if (typeof finalHasToolCall !== "boolean") {
    throw new Error("finalHasToolCall must be a boolean");
  }
  if (finalStopReason !== undefined && typeof finalStopReason !== "string") {
    throw new Error("finalStopReason must be a string or undefined");
  }
  requireInteger(
    outputBudgetOvershootTokens,
    "outputBudgetOvershootTokens",
    0,
    Number.MAX_SAFE_INTEGER,
  );
  const hasTerminalAnswer =
    answerText.trim().length > 0 &&
    !finalHasToolCall &&
    !["error", "aborted"].includes(finalStopReason);
  if (hasTerminalAnswer) {
    return {
      status: "succeeded",
      stopReason: outputBudgetOvershootTokens > 0
        ? `completed_with_output_token_overshoot:${outputBudgetOvershootTokens}`
        : "completed",
    };
  }
  return budgetStop
    ? { status: "budget_exhausted", stopReason: budgetStop }
    : { status: "failed", stopReason: "no_final_answer" };
}

export function validateSearchResults(results, { allowEmpty = true } = {}) {
  if (!Array.isArray(results) || results.length > 5 || (!allowEmpty && results.length === 0)) {
    throw new Error("search results must contain at most 5 items");
  }
  for (const [index, result] of results.entries()) {
    requireObject(result, `search.results[${index}]`);
    rejectUnknownKeys(result, new Set(["docid", "score", "snippet"]), `search.results[${index}]`);
    requireString(result.docid, `search.results[${index}].docid`);
    requireString(result.snippet, `search.results[${index}].snippet`);
    if (typeof result.score !== "number" || !Number.isFinite(result.score)) {
      throw new Error(`search.results[${index}].score must be finite`);
    }
  }
  return results;
}

function validateSearch(search) {
  requireObject(search, "search");
  if (search.kind === "fixture") {
    rejectUnknownKeys(search, new Set(["kind", "results"]), "search");
    validateSearchResults(search.results, { allowEmpty: false });
    return;
  }
  if (search.kind === "http") {
    rejectUnknownKeys(search, new Set(["kind", "url", "timeout_ms"]), "search");
    requireString(search.url, "search.url");
    const url = new URL(search.url);
    if (url.protocol !== "http:" || url.username || url.password) {
      throw new Error("BM25 search URL must be unauthenticated HTTP");
    }
    if (!['127.0.0.1', 'localhost', '::1'].includes(url.hostname)) {
      throw new Error("BM25 search URL must use a loopback host");
    }
    requireInteger(search.timeout_ms, "search.timeout_ms", 1, 120000);
    return;
  }
  throw new Error("search.kind must be fixture or http");
}

function rejectUnknownKeys(value, allowed, label) {
  const unknown = Object.keys(value).filter((key) => !allowed.has(key));
  if (unknown.length > 0) {
    throw new Error(`${label} has unknown fields: ${unknown.sort().join(", ")}`);
  }
}

function requireObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function requireString(value, label) {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${label} must be a non-empty string`);
  }
}

function requireEqual(value, expected, label) {
  if (value !== expected) {
    throw new Error(`${label} must equal ${expected}`);
  }
}

function requireInteger(value, label, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new Error(`${label} must be an integer between ${minimum} and ${maximum}`);
  }
}
