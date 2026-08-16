import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";

const [registrationArgument, outputArgument] = process.argv.slice(2);
if (!registrationArgument || !outputArgument) {
  throw new Error("usage: node scripts/run_answer_first_audit_parity.mjs REGISTRATION OUTPUT");
}

const root = process.cwd();
const registrationPath = path.resolve(registrationArgument);
const outputPath = path.resolve(outputArgument);
const registrationBytes = await readFile(registrationPath);
const registration = JSON.parse(registrationBytes.toString("utf8"));
if (registration.schema_version !== "browsecomp-plus-answer-first-audit-parity-registration-v0") {
  throw new Error("unsupported answer-first parity registration");
}

const contractPath = path.resolve(root, registration.implementation.contract_path);
await requireHash(contractPath, registration.implementation.contract_sha256);
const runnerPath = path.resolve(root, registration.implementation.runner_path);
await requireHash(runnerPath, registration.implementation.runner_sha256);
const {
  DETERMINISTIC_AUDIT_POLICY,
  auditAnswerFirstDraft,
  sha256,
} = await import(pathToFileURL(contractPath).href);
if (sha256(DETERMINISTIC_AUDIT_POLICY) !== registration.implementation.policy_sha256) {
  throw new Error("registered deterministic policy hash does not match Pi v12");
}

const rows = [];
for (const calibrationCase of registration.cases) {
  const runPath = path.resolve(root, calibrationCase.run_path);
  await requireHash(runPath, calibrationCase.run_sha256);
  const run = JSON.parse(await readFile(runPath, "utf8"));
  if (String(run.query_id) !== String(calibrationCase.query_id)) {
    throw new Error(`query id mismatch for ${calibrationCase.run_path}`);
  }
  const audit = auditAnswerFirstDraft({
    question: extractQuestion(run.messages),
    answerText: run.answer_text,
    evidence: collectEvidence(run.search_calls, run.evidence_open_calls ?? []),
  });
  const observedRepairTrigger = audit.audit_status === "open";
  rows.push({
    query_id: String(run.query_id),
    paired_outcome: calibrationCase.paired_outcome,
    expected_repair_trigger: calibrationCase.expected_repair_trigger,
    observed_status: audit.audit_status,
    observed_repair_trigger: observedRepairTrigger,
    matched_expectation: observedRepairTrigger === calibrationCase.expected_repair_trigger,
    reasons: audit.reasons,
    repair_queries: audit.repair_queries,
    provider_calls: 0,
    search_calls: 0,
    source: {
      path: calibrationCase.run_path,
      sha256: calibrationCase.run_sha256,
    },
  });
}

const regressions = rows.filter((row) => row.paired_outcome === "candidate_regression");
const improvements = rows.filter((row) => row.paired_outcome === "candidate_improvement");
const regressionTriggerCount = regressions.filter((row) => row.observed_repair_trigger).length;
const improvementFalseTriggerCount = improvements.filter((row) => row.observed_repair_trigger).length;
const maximumRepairQueries = Math.max(...rows.map((row) => row.repair_queries.length));
const acceptance = registration.acceptance;
const gates = [
  gate("regression_trigger_recall", regressionTriggerCount / regressions.length, "eq", acceptance.regression_trigger_recall_must_equal),
  gate("improvement_false_trigger_count", improvementFalseTriggerCount, "eq", acceptance.improvement_false_trigger_count_must_equal),
  gate("provider_calls", 0, "eq", acceptance.provider_calls_must_equal),
  gate("search_calls", 0, "eq", acceptance.search_calls_must_equal),
  gate("maximum_repair_queries", maximumRepairQueries, "le", acceptance.maximum_repair_queries_per_open_case),
];
const decision = gates.every((item) => item.passed) ? "pass" : "reject";
const result = {
  schema_version: "browsecomp-plus-answer-first-audit-parity-v0",
  created_at: new Date().toISOString(),
  status: "outcome_selected_calibration_not_effectiveness_evidence",
  decision,
  regression_count: regressions.length,
  regression_trigger_count: regressionTriggerCount,
  regression_trigger_recall: regressionTriggerCount / regressions.length,
  improvement_count: improvements.length,
  improvement_false_trigger_count: improvementFalseTriggerCount,
  provider_calls: 0,
  search_calls: 0,
  maximum_repair_queries: maximumRepairQueries,
  rows,
  gates,
  sources: {
    registration: {
      path: slash(path.relative(root, registrationPath)),
      sha256: digest(registrationBytes),
    },
    contract: {
      path: registration.implementation.contract_path,
      sha256: registration.implementation.contract_sha256,
    },
    policy_sha256: registration.implementation.policy_sha256,
  },
  next_action: decision === "pass"
    ? "permit_outcome_selected_pi_v12_live_calibration"
    : "reject_pi_v12_before_provider_calls",
  claim_boundary: "Known-outcome saved-trace implementation parity only; no benchmark effectiveness claim.",
};

await atomicWrite(outputPath, `${JSON.stringify(result, null, 2)}\n`);
process.stdout.write(`output=${outputArgument}\ndecision=${decision}\n`);
process.stdout.write(`regression_trigger_recall=${regressionTriggerCount}/${regressions.length}\n`);
process.stdout.write(`improvement_false_trigger_count=${improvementFalseTriggerCount}/${improvements.length}\n`);
process.stdout.write("provider_calls=0\nsearch_calls=0\n");
if (decision !== "pass") process.exitCode = 2;

function extractQuestion(messages) {
  for (const message of messages ?? []) {
    if (message.role !== "user" || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (part?.type !== "text" || typeof part.text !== "string") continue;
      const match = part.text.match(/\bQuestion:\s*(.+?)\n\nYour response should be/s);
      if (match) return match[1].trim();
    }
  }
  throw new Error("saved Pi trace does not contain the benchmark question");
}

function collectEvidence(searchCalls, evidenceOpenCalls) {
  const chunks = new Map();
  for (const call of searchCalls ?? []) {
    for (const result of call.results ?? []) {
      const values = chunks.get(result.docid) ?? [];
      values.push(result.snippet);
      chunks.set(result.docid, values);
    }
  }
  for (const call of evidenceOpenCalls ?? []) {
    if (!call.result?.content) continue;
    const values = chunks.get(call.docid) ?? [];
    values.push(call.result.content);
    chunks.set(call.docid, values);
  }
  return [...chunks.entries()].map(([docid, values]) => ({
    docid,
    text: values.join("\n"),
  }));
}

function gate(gateId, observed, operator, threshold) {
  const passed = operator === "eq" ? observed === threshold : observed <= threshold;
  return { gate_id: gateId, observed, operator, threshold, passed };
}

async function requireHash(filePath, expected) {
  const observed = digest(await readFile(filePath));
  if (observed !== expected) {
    throw new Error(`hash mismatch for ${filePath}: expected ${expected}, observed ${observed}`);
  }
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function slash(value) {
  return value.replaceAll("\\", "/");
}

async function atomicWrite(filePath, text) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, text, "utf8");
  await rename(temporaryPath, filePath);
}
