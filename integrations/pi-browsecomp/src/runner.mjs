import { readFile } from "node:fs/promises";
import process from "node:process";
import {
  createAgentSession,
  DefaultResourceLoader,
  defineTool,
  SessionManager,
  SettingsManager,
} from "@earendil-works/pi-coding-agent";
import { getModel } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import {
  ADAPTER_VERSION,
  PI_VERSION,
  classifyRunOutcome,
  clampProviderOutput,
  formatBenchmarkPrompt,
  sha256,
  validateRequest,
  validateSearchResults,
} from "./contract.mjs";

const request = validateRequest(JSON.parse(await readInput()));
if (!process.env.DEEPSEEK_API_KEY) {
  throw new Error("DEEPSEEK_API_KEY is not set; inline API keys are not accepted");
}

const catalogModel = getModel("deepseek", request.model);
if (!catalogModel) {
  throw new Error(`${request.model} is absent from Pi ${PI_VERSION}'s model catalog`);
}
const model = { ...catalogModel, maxTokens: request.max_output_tokens };
const prompt = formatBenchmarkPrompt(request.question);
const searchCalls = [];
const providerRequestLimits = [];
let modelRequests = 0;
let providerOutputTokens = 0;
let budgetStop;

const searchTool = defineTool({
  name: "search",
  label: "Search",
  description: "Perform a search on a knowledge source. Returns top-5 hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits).",
  parameters: Type.Object(
    { query: Type.String({ description: "Search query string" }) },
    { additionalProperties: false },
  ),
  execute: async (_toolCallId, params) => {
    const callStarted = performance.now();
    try {
      // The reference loop executes tool calls emitted by the terminal model
      // response, then checks the global output/iteration budget.
      const results = await executeSearch(request.search, params.query);
      searchCalls.push({
        query: params.query,
        outcome: "ok",
        latency_ms: Math.round(performance.now() - callStarted),
        results,
      });
      return { content: [{ type: "text", text: JSON.stringify(results, null, 2) }], details: { results } };
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      searchCalls.push({
        query: params.query,
        outcome: "error",
        latency_ms: Math.round(performance.now() - callStarted),
        detail,
        results: [],
      });
      return { content: [{ type: "text", text: JSON.stringify({ error: detail }) }], details: { error: detail } };
    }
  },
});

const resourceLoader = new DefaultResourceLoader({
  cwd: process.cwd(),
  agentDir: process.cwd(),
  noExtensions: true,
  noSkills: true,
  noPromptTemplates: true,
  noThemes: true,
  noContextFiles: true,
  systemPromptOverride: () => " ",
  extensionFactories: [
    (pi) => {
      pi.on("before_agent_start", () => ({ systemPrompt: "" }));
      pi.on("before_provider_request", (event) => {
        const remainingOutputTokens = Math.max(
          request.max_output_tokens - providerOutputTokens,
          1,
        );
        const bounded = clampProviderOutput(event.payload, remainingOutputTokens);
        providerRequestLimits.push({
          request_index: providerRequestLimits.length + 1,
          remaining_output_tokens: remainingOutputTokens,
          requested_output_tokens:
            event.payload.max_completion_tokens ?? event.payload.max_tokens ?? null,
          applied_output_tokens:
            bounded.max_completion_tokens ?? bounded.max_tokens ?? null,
        });
        return bounded;
      });
      pi.on("message_end", (event) => {
        if (event.message.role !== "assistant") return;
        modelRequests += 1;
        providerOutputTokens += event.message.usage?.output ?? 0;
      });
    },
  ],
});
await resourceLoader.reload();

const startedAt = new Date();
const started = performance.now();
const { session } = await createAgentSession({
  cwd: process.cwd(),
  model,
  thinkingLevel: request.thinking_level,
  tools: ["search"],
  customTools: [searchTool],
  resourceLoader,
  sessionManager: SessionManager.inMemory(process.cwd()),
  settingsManager: SettingsManager.inMemory({
    compaction: { enabled: false },
    retry: {
      enabled: true,
      maxRetries: 3,
      provider: { maxRetries: 3 },
    },
  }),
});
session.agent.shouldStopAfterTurn = () => {
  if (providerOutputTokens >= request.max_output_tokens) {
    budgetStop = `global_output_token_budget_exhausted:${request.max_output_tokens}`;
    return true;
  }
  if (modelRequests >= request.max_iterations) {
    budgetStop = `max_iterations_exceeded:${request.max_iterations}`;
    return true;
  }
  return false;
};

try {
  try {
    await session.prompt(prompt);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    if (budgetStop) {
      // The full partial trace is persisted below.
    } else if (
      detail.includes("global_output_token_budget_exhausted") ||
      detail.includes("max_iterations_exceeded")
    ) {
      budgetStop = detail;
    } else {
      throw error;
    }
  }
  const messages = session.agent.state.messages;
  const final = [...messages].reverse().find((message) => message.role === "assistant");
  const finalHasToolCall = final?.content?.some((part) => part.type === "toolCall") ?? false;
  const answerText = final?.content
    ?.filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n") ?? "";
  const usage = aggregateUsage(messages);
  const outputBudgetOvershootTokens = Math.max(
    usage.output_tokens - request.max_output_tokens,
    0,
  );
  if (outputBudgetOvershootTokens > 0 && !budgetStop) {
    budgetStop = `global_output_token_budget_overshot:${request.max_output_tokens}`;
  }
  const { status, stopReason } = classifyRunOutcome({
    answerText,
    finalHasToolCall,
    finalStopReason: final?.stopReason,
    budgetStop,
    outputBudgetOvershootTokens,
  });
  const output = {
    schema_version: "pi-browsecomp-run-v0",
    adapter_version: ADAPTER_VERSION,
    pi_version: PI_VERSION,
    run_id: request.run_id,
    query_id: request.query_id,
    model: request.model,
    thinking_level: request.thinking_level,
    system_prompt: session.agent.state.systemPrompt,
    prompt_sha256: sha256(prompt),
    started_at: startedAt.toISOString(),
    latency_ms: Math.round(performance.now() - started),
    status,
    stop_reason: stopReason,
    answer_text: answerText,
    usage,
    output_budget_overshoot_tokens: outputBudgetOvershootTokens,
    model_requests: modelRequests,
    provider_request_limits: providerRequestLimits,
    search_calls: searchCalls,
    messages,
  };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
} finally {
  session.dispose();
}

async function readInput() {
  const path = process.argv[2];
  if (path) return readFile(path, "utf8");
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

async function executeSearch(search, query) {
  if (search.kind === "fixture") return search.results;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), search.timeout_ms);
  try {
    const response = await fetch(search.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`BM25 server returned HTTP ${response.status}`);
    const body = await response.json();
    return validateSearchResults(body.results);
  } finally {
    clearTimeout(timeout);
  }
}

function aggregateUsage(messages) {
  const total = {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_write_tokens: 0,
    reasoning_tokens: 0,
    total_tokens: 0,
    cost_usd: 0,
  };
  for (const message of messages) {
    if (message.role !== "assistant" || !message.usage) continue;
    total.input_tokens += message.usage.input ?? 0;
    total.output_tokens += message.usage.output ?? 0;
    total.cache_read_tokens += message.usage.cacheRead ?? 0;
    total.cache_write_tokens += message.usage.cacheWrite ?? 0;
    total.reasoning_tokens += message.usage.reasoning ?? 0;
    total.total_tokens += message.usage.totalTokens ?? 0;
    total.cost_usd += message.usage.cost?.total ?? 0;
  }
  return total;
}
