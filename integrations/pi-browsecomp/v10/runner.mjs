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
  ANSWER_COMPILER_PROMPT,
  FIRST_TOOL_DEADLINE_PROMPT,
  FIRST_TOOL_DEADLINE_TOKENS,
  MAXIMUM_SEARCH_CALLS,
  formatConstraintPortfolioPrompt,
  formatRareAnchorBootstrapPrompt,
  PI_VERSION,
  answerReserveAllocation,
  applySamplingPolicy,
  classifyRunOutcome,
  clampProviderOutput,
  formatBenchmarkPrompt,
  formatToolBootstrapPrompt,
  hasRequiredAnswerSchema,
  remainingProviderOutput,
  reserveSearchCall,
  sha256,
  shouldInvokeAnswerCompiler,
  validateDisclosureSearchResponse,
  validateOpenEvidenceResponse,
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
const model = {
  ...catalogModel,
  maxTokens: request.max_output_tokens,
  compat: {
    ...catalogModel.compat,
    // Pi 0.84.1 auto-detects max_completion_tokens for DeepSeek, while the
    // provider's Chat Completions contract documents max_tokens.
    maxTokensField: "max_tokens",
  },
};
const prompt = formatBenchmarkPrompt(request.question);
const toolBootstrapPrompt = request.control_policy === "tool_bootstrap_v0"
  ? formatToolBootstrapPrompt(request.question)
  : request.control_policy === "rare_anchor_portfolio_v0"
    ? formatRareAnchorBootstrapPrompt(request.question)
    : request.control_policy === "constraint_portfolio_v1"
      ? formatConstraintPortfolioPrompt(request.question)
    : null;
const searchCalls = [];
const evidenceOpenCalls = [];
const blockedSearchCalls = [];
const providerRequestLimits = [];
let disclosureState = null;
let reservedSearchCalls = 0;
let modelRequests = 0;
let observedOutputTokens = 0;
let phase = ["tool_bootstrap_v0", "rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy)
  ? "bootstrap"
  : "exploration";
const phaseOutputTokens = { bootstrap: 0, exploration: 0, compilation: 0 };
let answerCompilerInvoked = false;
let firstToolDeadlineTriggered = false;
let explorationStopReason;
let bootstrapStopReason;
let budgetStop;
let researchBudgetStop;
let runtimeError;

const searchTool = defineTool({
  name: "search",
  label: "Search",
  description: `Perform a search on a knowledge source. Returns up to ${request.search.max_results ?? 5} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits).`,
  parameters: Type.Object(
    { query: Type.String({ description: "Search query string" }) },
    { additionalProperties: false },
  ),
  execute: async (_toolCallId, params) => {
    const callStarted = performance.now();
    const reservation = reserveSearchCall(reservedSearchCalls);
    reservedSearchCalls = reservation.reserved_search_calls;
    researchBudgetStop = reservation.stop_reason ?? researchBudgetStop;
    if (!reservation.allowed) {
      blockedSearchCalls.push({
        query: params.query,
        reason: researchBudgetStop,
      });
      const result = {
        error: researchBudgetStop,
        search_budget_exhausted: true,
        maximum_search_calls: MAXIMUM_SEARCH_CALLS,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        details: result,
      };
    }
    try {
      // The reference loop executes tool calls emitted by the terminal model
      // response, then checks the global output/iteration budget.
      const response = await executeSearch(request.search, request.run_id, params.query);
      const results = response.results;
      disclosureState = response.state;
      searchCalls.push({
        query: params.query,
        outcome: "ok",
        latency_ms: Math.round(performance.now() - callStarted),
        results,
        disclosure: response.disclosure,
        state: response.state,
      });
      const visible = { results, disclosure: response.disclosure };
      return { content: [{ type: "text", text: JSON.stringify(visible, null, 2) }], details: response };
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      searchCalls.push({
        query: params.query,
        outcome: "error",
        latency_ms: Math.round(performance.now() - callStarted),
        detail,
        results: [],
        disclosure: null,
        state: null,
      });
      return { content: [{ type: "text", text: JSON.stringify({ error: detail }) }], details: { error: detail } };
    }
  },
});

const openEvidenceTool = defineTool({
  name: "open_evidence",
  label: "Open evidence",
  description: "Open the full evidence for one docid previously returned as a Dense lead. Use this only when the short preview is relevant enough to verify before relying on it.",
  parameters: Type.Object(
    { docid: Type.String({ description: "Dense lead docid returned by search" }) },
    { additionalProperties: false },
  ),
  execute: async (_toolCallId, params) => {
    const callStarted = performance.now();
    try {
      const response = await executeOpenEvidence(request.search, request.run_id, params.docid);
      disclosureState = response.state;
      evidenceOpenCalls.push({
        docid: params.docid,
        outcome: "ok",
        latency_ms: Math.round(performance.now() - callStarted),
        result: response.result,
        state: response.state,
      });
      return {
        content: [{ type: "text", text: JSON.stringify(response.result, null, 2) }],
        details: response,
      };
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      evidenceOpenCalls.push({
        docid: params.docid,
        outcome: "error",
        latency_ms: Math.round(performance.now() - callStarted),
        detail,
        result: null,
        state: null,
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
        const globalRemainingOutputTokens = Math.max(
          request.max_output_tokens - observedOutputTokens,
          0,
        );
        const allocation = request.control_policy === "standard"
          ? { exploration: request.max_output_tokens, compilation: 0 }
          : answerReserveAllocation(request.control_policy);
        const phaseLimit = allocation[phase];
        const phaseRemainingOutputTokens = Math.max(
          phaseLimit - phaseOutputTokens[phase],
          0,
        );
        const remainingOutputTokens = remainingProviderOutput({
          controlPolicy: request.control_policy,
          phase,
          maxOutputTokens: request.max_output_tokens,
          totalOutputTokens: observedOutputTokens,
          phaseOutputTokens: phaseOutputTokens[phase],
        });
        if (remainingOutputTokens <= 0) {
          throw new Error(`no_output_budget_remaining:${phase}`);
        }
        const policyCapOutputTokens =
          request.control_policy === "first_tool_deadline_v0" && modelRequests === 0
            ? FIRST_TOOL_DEADLINE_TOKENS
            : null;
        const effectiveOutputTokens = policyCapOutputTokens === null
          ? remainingOutputTokens
          : Math.min(remainingOutputTokens, policyCapOutputTokens);
        const sampled = applySamplingPolicy(event.payload);
        const bounded = clampProviderOutput(sampled, effectiveOutputTokens);
        providerRequestLimits.push({
          request_index: providerRequestLimits.length + 1,
          phase,
          output_limit_field: "max_tokens",
          thinking_type: event.payload.thinking?.type ?? null,
          temperature: bounded.temperature ?? null,
          remaining_output_tokens: effectiveOutputTokens,
          global_remaining_output_tokens: globalRemainingOutputTokens,
          phase_remaining_output_tokens: phaseRemainingOutputTokens,
          policy_cap_output_tokens: policyCapOutputTokens,
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
        const outputTokens = event.message.usage?.output ?? 0;
        observedOutputTokens += outputTokens;
        phaseOutputTokens[phase] += outputTokens;
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
  tools: ["search", "open_evidence"],
  customTools: [searchTool, openEvidenceTool],
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
  if (observedOutputTokens >= request.max_output_tokens) {
    budgetStop = `global_output_token_budget_exhausted:${request.max_output_tokens}`;
    return true;
  }
  if (researchBudgetStop && phase === "exploration") {
    explorationStopReason = researchBudgetStop;
    return true;
  }
  if (["tool_bootstrap_v0", "rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy) && phase === "bootstrap") {
    const bootstrapLimit = answerReserveAllocation(request.control_policy).bootstrap;
    const targetSearchCalls = ["rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy) ? 3 : 1;
    if (searchCalls.length >= targetSearchCalls) {
      bootstrapStopReason = "bootstrap_search_completed";
      return true;
    }
    if (phaseOutputTokens.bootstrap >= bootstrapLimit) {
      bootstrapStopReason = `bootstrap_output_token_limit_reached:${bootstrapLimit}`;
      return true;
    }
  }
  if (
    request.control_policy !== "standard" &&
    phase === "exploration" &&
    phaseOutputTokens.exploration >= answerReserveAllocation(request.control_policy).exploration
  ) {
    const explorationLimit = answerReserveAllocation(request.control_policy).exploration;
    explorationStopReason = `exploration_output_token_limit_reached:${explorationLimit}`;
    return true;
  }
  if (modelRequests >= request.max_iterations) {
    budgetStop = `max_iterations_exceeded:${request.max_iterations}`;
    return true;
  }
  return false;
};

try {
  if (["tool_bootstrap_v0", "rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy)) {
    session.setThinkingLevel("off");
    await runPrompt(toolBootstrapPrompt);
    if (!bootstrapStopReason) {
      bootstrapStopReason = `bootstrap_incomplete_search_calls:${searchCalls.length}`;
    }
    if (!runtimeError && !budgetStop) {
      phase = "exploration";
      session.setThinkingLevel(request.thinking_level);
      await runPrompt(prompt);
    }
  } else {
    await runPrompt(prompt);
  }
  if (
    request.control_policy === "first_tool_deadline_v0" &&
    searchCalls.length === 0 &&
    !runtimeError &&
    !budgetStop &&
    phaseOutputTokens.exploration < answerReserveAllocation(request.control_policy).exploration
  ) {
    firstToolDeadlineTriggered = true;
    await runPrompt(FIRST_TOOL_DEADLINE_PROMPT);
  }
  let messages = session.agent.state.messages;
  let final = lastAssistantMessage(messages);
  let finalHasToolCall = hasToolCall(final);
  let answerText = textContent(final);
  if (
    shouldInvokeAnswerCompiler({
      controlPolicy: request.control_policy,
      answerText,
      finalHasToolCall,
      totalOutputTokens: observedOutputTokens,
      maxOutputTokens: request.max_output_tokens,
      stopReason: runtimeError ?? budgetStop,
    })
  ) {
    answerCompilerInvoked = true;
    phase = "compilation";
    session.setActiveToolsByName([]);
    if (["answer_reserve_nonthinking_v0", "first_tool_deadline_v0", "tool_bootstrap_v0", "rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy)) {
      session.setThinkingLevel("off");
    }
    await runPrompt(ANSWER_COMPILER_PROMPT);
    messages = session.agent.state.messages;
    final = lastAssistantMessage(messages);
    finalHasToolCall = hasToolCall(final);
    answerText = textContent(final);
  }
  const usage = aggregateUsage(messages);
  if (usage.output_tokens !== observedOutputTokens && !runtimeError) {
    runtimeError = (
      `usage_output_mismatch:events=${observedOutputTokens}:messages=${usage.output_tokens}`
    );
  }
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
    failureReason: runtimeError,
  });
  const output = {
    schema_version: "pi-browsecomp-run-v0",
    adapter_version: ADAPTER_VERSION,
    pi_version: PI_VERSION,
    run_id: request.run_id,
    query_id: request.query_id,
    model: request.model,
    thinking_level: request.thinking_level,
    max_search_results: request.search.max_results ?? 5,
    compilation_thinking_level:
      answerCompilerInvoked && ["answer_reserve_nonthinking_v0", "first_tool_deadline_v0", "tool_bootstrap_v0", "rare_anchor_portfolio_v0", "constraint_portfolio_v1"].includes(request.control_policy)
        ? "off"
        : request.thinking_level,
    control_policy: request.control_policy,
    system_prompt: session.agent.state.systemPrompt,
    prompt_sha256: sha256(prompt),
    started_at: startedAt.toISOString(),
    latency_ms: Math.round(performance.now() - started),
    status,
    stop_reason: stopReason,
    answer_text: answerText,
    usage,
    exploration_stop_reason: explorationStopReason ?? null,
    bootstrap_stop_reason: bootstrapStopReason ?? null,
    bootstrap_output_tokens: phaseOutputTokens.bootstrap,
    bootstrap_prompt_sha256:
      toolBootstrapPrompt === null ? null : sha256(toolBootstrapPrompt),
    exploration_output_tokens: phaseOutputTokens.exploration,
    compilation_output_tokens: phaseOutputTokens.compilation,
    answer_compiler_invoked: answerCompilerInvoked,
    answer_compiler_prompt_sha256:
      answerCompilerInvoked ? sha256(ANSWER_COMPILER_PROMPT) : null,
    first_tool_deadline_triggered: firstToolDeadlineTriggered,
    first_tool_deadline_prompt_sha256:
      firstToolDeadlineTriggered ? sha256(FIRST_TOOL_DEADLINE_PROMPT) : null,
    answer_schema_complete: hasRequiredAnswerSchema(answerText),
    output_budget_overshoot_tokens: outputBudgetOvershootTokens,
    model_requests: modelRequests,
    provider_request_limits: providerRequestLimits,
    search_calls: searchCalls,
    evidence_open_calls: evidenceOpenCalls,
    disclosure_state: disclosureState,
    research_budget: {
      maximum_search_calls: MAXIMUM_SEARCH_CALLS,
      reserved_search_calls: reservedSearchCalls,
      executed_search_calls: searchCalls.length,
      blocked_search_calls: blockedSearchCalls,
      exhausted: researchBudgetStop !== undefined,
      stop_reason: researchBudgetStop ?? null,
    },
    messages,
  };
  process.stdout.write(`${JSON.stringify(output, null, 2)}\n`);
} finally {
  session.dispose();
}

async function runPrompt(text) {
  try {
    await session.prompt(text, { expandPromptTemplates: false });
  } catch (error) {
    const detail = redactSecrets(error instanceof Error ? error.message : String(error));
    if (budgetStop) return;
    if (
      detail.includes("global_output_token_budget_exhausted") ||
      detail.includes("max_iterations_exceeded")
    ) {
      budgetStop = detail;
      return;
    }
    runtimeError = `provider_or_runtime_error:${detail}`;
  }
}

function lastAssistantMessage(messages) {
  return [...messages].reverse().find((message) => message.role === "assistant");
}

function hasToolCall(message) {
  return message?.content?.some((part) => part.type === "toolCall") ?? false;
}

function textContent(message) {
  return message?.content
    ?.filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n") ?? "";
}

function redactSecrets(text) {
  const apiKey = process.env.DEEPSEEK_API_KEY ?? "";
  return apiKey ? text.replaceAll(apiKey, "[REDACTED]") : text;
}

async function readInput() {
  const path = process.argv[2];
  if (path) return readFile(path, "utf8");
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

async function executeSearch(search, runId, query) {
  if (search.kind === "fixture") {
    return {
      results: search.results,
      disclosure: {
        search_index: 1,
        anchors_returned: search.results.length,
        leads_returned: 0,
        within_channel_duplicate_slots: 0,
        prior_context_duplicate_slots: 0,
        cross_channel_duplicate_slots: 0,
        new_ingress_tokens: 0,
        cumulative_ingress_tokens: 0,
        remaining_ingress_tokens: 0,
        remaining_search_ingress_tokens: 0,
        remaining_open_ingress_tokens: 0,
        ingress_budget_exhausted: true,
      },
      state: {
        run_id: runId,
        search_calls: 1,
        open_attempts: 0,
        successful_open_calls: 0,
        seen_docids: search.results.map((row) => row.docid),
        eligible_open_docids: [],
        opened_docids: search.results.map((row) => row.docid),
        cumulative_ingress_tokens: 0,
        search_ingress_tokens: 0,
        open_ingress_tokens: 0,
        remaining_ingress_tokens: 0,
        remaining_search_ingress_tokens: 0,
        remaining_open_ingress_tokens: 0,
      },
      latency_ms: 0,
    };
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), search.timeout_ms);
  try {
    const response = await fetch(search.url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ run_id: runId, query }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`progressive disclosure server returned HTTP ${response.status}`);
    const body = await response.json();
    return validateDisclosureSearchResponse(body, {
      maxResults: search.max_results ?? 5,
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function executeOpenEvidence(search, runId, docid) {
  if (search.kind !== "http") {
    throw new Error("open_evidence requires the HTTP progressive disclosure server");
  }
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), search.timeout_ms);
  try {
    const response = await fetch(siblingEndpoint(search.url, "open"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ run_id: runId, docid }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(`progressive disclosure server returned HTTP ${response.status}`);
    return validateOpenEvidenceResponse(await response.json());
  } finally {
    clearTimeout(timeout);
  }
}

function siblingEndpoint(searchUrl, endpoint) {
  const url = new URL(searchUrl);
  if (!url.pathname.endsWith("/search")) {
    throw new Error("progressive disclosure search URL must end with /search");
  }
  url.pathname = `${url.pathname.slice(0, -"search".length)}${endpoint}`;
  return url.toString();
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
