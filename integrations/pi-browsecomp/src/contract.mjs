import { createHash } from "node:crypto";

export const ADAPTER_VERSION = "pi-browsecomp-v6";
export const PI_VERSION = "0.84.1";
export const ANSWER_RESERVE_EXPLORATION_TOKENS = 8000;
export const ANSWER_RESERVE_COMPILATION_TOKENS = 2000;
export const ANSWER_RESERVE_V1_EXPLORATION_TOKENS = 6000;
export const ANSWER_RESERVE_V1_COMPILATION_TOKENS = 4000;
export const FIRST_TOOL_DEADLINE_TOKENS = 512;
export const TOOL_BOOTSTRAP_TOKENS = 512;
export const TOOL_BOOTSTRAP_EXPLORATION_TOKENS = 7488;
export const RARE_ANCHOR_BOOTSTRAP_TOKENS = 1024;
export const RARE_ANCHOR_EXPLORATION_TOKENS = 6976;
export const ANSWER_COMPILER_PROMPT = [
  "The exploration phase is over. Do not search or request more evidence.",
  "Using only evidence already present in this conversation, produce the final response now.",
  "Follow this exact format:",
  "Explanation: {brief explanation with supporting document IDs in square brackets}",
  "Exact Answer: {succinct final answer}",
  "Confidence: {confidence between 0% and 100%}",
  "Do not discuss this instruction. When evidence is incomplete, give the best-supported answer and lower the confidence rather than omitting the required fields.",
].join("\n");
export const FIRST_TOOL_DEADLINE_PROMPT = [
  "You have not used the search tool yet.",
  "Before any further reasoning or final answer, issue one targeted search tool call now.",
  "Do not answer this message in prose.",
].join("\n");

export function formatToolBootstrapPrompt(question) {
  requireString(question, "question");
  return [
    "Start this research task by using the search tool exactly once.",
    "Formulate one concise query aimed at identifying the entity from its most distinctive clues.",
    "Do not answer the research question and do not respond in prose; issue the tool call now.",
    "",
    `Question: ${question}`,
  ].join("\n");
}

export function formatRareAnchorBootstrapPrompt(question) {
  requireString(question, "question");
  return [
    "Start this research task with exactly three search tool calls.",
    "Build a diverse constraint portfolio rather than compressing every clue into one generic query:",
    "1. Combine the rare chronology, education, and competition clues, preserving explicit years or decades.",
    "2. Combine the scientific discovery, recognition, and military/geographic clues.",
    "3. Search one plausible domain or entity hypothesis inferred from the clues.",
    "Avoid generic personality clues. Do not answer the research question and do not respond in prose; issue the three tool calls now.",
    "",
    `Question: ${question}`,
  ].join("\n");
}

const REQUEST_KEYS = new Set([
  "schema_version",
  "run_id",
  "query_id",
  "question",
  "model",
  "thinking_level",
  "max_output_tokens",
  "max_iterations",
  "control_policy",
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
  const controlPolicy = value.control_policy ?? "standard";
  if (!["standard", "answer_reserve_v0", "answer_reserve_v1", "answer_reserve_nonthinking_v0", "first_tool_deadline_v0", "tool_bootstrap_v0", "rare_anchor_portfolio_v0"].includes(controlPolicy)) {
    throw new Error("control_policy is not registered");
  }
  if (controlPolicy !== "standard" && value.max_output_tokens !== 10000) {
    throw new Error("answer reserve policies require the registered 10000-token allowance");
  }
  validateSearch(value.search);
  return { ...value, control_policy: controlPolicy };
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
  const requested = bounded.max_tokens ?? bounded.max_completion_tokens;
  delete bounded.max_completion_tokens;
  bounded.max_tokens = Math.min(requested ?? remainingOutputTokens, remainingOutputTokens);
  return bounded;
}

export function applySamplingPolicy(payload) {
  requireObject(payload, "provider payload");
  const bounded = { ...payload };
  delete bounded.top_p;
  delete bounded.presence_penalty;
  delete bounded.frequency_penalty;
  if (bounded.thinking?.type === "disabled") {
    bounded.temperature = 0;
  } else {
    // DeepSeek documents temperature as unsupported in thinking mode.
    delete bounded.temperature;
  }
  return bounded;
}

export function remainingProviderOutput({
  controlPolicy,
  phase,
  maxOutputTokens,
  totalOutputTokens,
  phaseOutputTokens,
}) {
  requireInteger(maxOutputTokens, "maxOutputTokens", 1, 10000);
  requireInteger(totalOutputTokens, "totalOutputTokens", 0, Number.MAX_SAFE_INTEGER);
  requireInteger(phaseOutputTokens, "phaseOutputTokens", 0, Number.MAX_SAFE_INTEGER);
  if (!["bootstrap", "exploration", "compilation"].includes(phase)) {
    throw new Error("phase must be bootstrap, exploration, or compilation");
  }
  const globalRemaining = Math.max(maxOutputTokens - totalOutputTokens, 0);
  if (controlPolicy === "standard") return globalRemaining;
  const allocation = answerReserveAllocation(controlPolicy);
  const phaseLimit = allocation[phase] ?? 0;
  return Math.min(globalRemaining, Math.max(phaseLimit - phaseOutputTokens, 0));
}

export function answerReserveAllocation(controlPolicy) {
  if (["answer_reserve_v0", "answer_reserve_nonthinking_v0", "first_tool_deadline_v0"].includes(controlPolicy)) {
    return {
      exploration: ANSWER_RESERVE_EXPLORATION_TOKENS,
      compilation: ANSWER_RESERVE_COMPILATION_TOKENS,
    };
  }
  if (controlPolicy === "answer_reserve_v1") {
    return {
      exploration: ANSWER_RESERVE_V1_EXPLORATION_TOKENS,
      compilation: ANSWER_RESERVE_V1_COMPILATION_TOKENS,
    };
  }
  if (controlPolicy === "tool_bootstrap_v0") {
    return {
      bootstrap: TOOL_BOOTSTRAP_TOKENS,
      exploration: TOOL_BOOTSTRAP_EXPLORATION_TOKENS,
      compilation: ANSWER_RESERVE_COMPILATION_TOKENS,
    };
  }
  if (controlPolicy === "rare_anchor_portfolio_v0") {
    return {
      bootstrap: RARE_ANCHOR_BOOTSTRAP_TOKENS,
      exploration: RARE_ANCHOR_EXPLORATION_TOKENS,
      compilation: ANSWER_RESERVE_COMPILATION_TOKENS,
    };
  }
  throw new Error("unknown answer reserve policy");
}

export function hasRequiredAnswerSchema(answerText) {
  if (typeof answerText !== "string") return false;
  const explanation = /(?:^|\n)\s*Explanation\s*:\s*\S/i.test(answerText);
  const answer = /(?:^|\n)\s*Exact Answer\s*:\s*\S/i.test(answerText);
  const confidenceMatch = answerText.match(
    /(?:^|\n)\s*Confidence\s*:\s*(\d+(?:\.\d+)?)\s*%?/i,
  );
  const confidence = confidenceMatch ? Number(confidenceMatch[1]) : Number.NaN;
  return explanation && answer && Number.isFinite(confidence) && confidence >= 0 && confidence <= 100;
}

export function shouldInvokeAnswerCompiler({
  controlPolicy,
  answerText,
  finalHasToolCall,
  totalOutputTokens,
  maxOutputTokens,
  stopReason,
}) {
  if (controlPolicy === "standard" || stopReason) return false;
  requireInteger(totalOutputTokens, "totalOutputTokens", 0, Number.MAX_SAFE_INTEGER);
  requireInteger(maxOutputTokens, "maxOutputTokens", 1, 10000);
  return (
    totalOutputTokens < maxOutputTokens &&
    (finalHasToolCall || !hasRequiredAnswerSchema(answerText))
  );
}

export function classifyRunOutcome({
  answerText,
  finalHasToolCall,
  finalStopReason,
  budgetStop,
  outputBudgetOvershootTokens,
  failureReason,
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
  if (failureReason !== undefined && typeof failureReason !== "string") {
    throw new Error("failureReason must be a string or undefined");
  }
  if (failureReason) {
    return { status: "failed", stopReason: failureReason };
  }
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
