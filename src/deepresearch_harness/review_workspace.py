from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Literal

from .review import BlindReviewPacket, BlindReviewSubmission, file_sha256, validate_blind_submission
from .review_translation import load_review_translation_bundle


def render_review_workspace(
    *,
    packet_path: Path,
    output_path: Path,
    locale: Literal["en", "zh-CN"] = "en",
    translations_path: Path | None = None,
) -> Path:
    packet = BlindReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    packet_json = json.dumps(packet.model_dump(mode="json"), ensure_ascii=True).replace("</", "<\\/")
    translations: dict[str, str] = {}
    translator_model = ""
    if locale == "zh-CN":
        if translations_path is None:
            raise ValueError("translations_path is required for locale zh-CN")
        bundle = load_review_translation_bundle(
            packet_path=packet_path,
            translations_path=translations_path,
        )
        translations = {item.source: item.translated for item in bundle.entries}
        translator_model = bundle.model

    html = _HTML_TEMPLATE
    if locale == "zh-CN":
        for source, translated in sorted(_ZH_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
            html = html.replace(source, translated)
    translation_json = json.dumps(translations, ensure_ascii=True).replace("</", "<\\/")
    html = html.replace("__HTML_LANG__", "zh-CN" if locale == "zh-CN" else "en")
    html = html.replace("__EXPERIMENT_ID__", escape(packet.experiment_id))
    html = html.replace("__PACKET_SHA256__", file_sha256(packet_path))
    html = html.replace("__PACKET_JSON__", packet_json)
    html = html.replace("__TRANSLATION_JSON__", translation_json)
    html = html.replace("__TRANSLATOR_MODEL__", escape(translator_model))
    html = html.replace("__LOCALE__", locale)
    html = html.replace("__TRANSLATION_ONLY_CLASS__", "" if locale == "zh-CN" else "hidden")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def validate_review_submission_file(*, packet_path: Path, annotations_path: Path) -> tuple[int, str, str]:
    packet = BlindReviewPacket.model_validate_json(packet_path.read_text(encoding="utf-8"))
    submission = BlindReviewSubmission.model_validate_json(annotations_path.read_text(encoding="utf-8"))
    validated = validate_blind_submission(packet, submission)
    return len(validated), file_sha256(annotations_path), submission.reviewer_type.value


_ZH_REPLACEMENTS = {
    "Blind Semantic Review": "人工盲审工作台",
    "Blind Review": "人工盲审",
    "0 / 20 candidates complete": "0 / 20 个候选已完成",
    "candidates complete": "个候选已完成",
    "Import draft": "导入草稿",
    "Download draft": "下载草稿",
    "Export final": "导出最终标注",
    "Review tasks": "评审任务",
    "Review criteria": "评审标准",
    "Forbidden shortcuts": "禁止的捷径",
    "Review A/B reports for obligation coverage, claim support, and citation match. Do not guess which system variant produced a candidate.": "逐项判断 A/B 报告的回答义务覆盖、Claim 支持和引用匹配；不要猜测候选由哪个系统版本生成。",
    "Tasks": "任务",
    "Packet SHA-256": "盲审包 SHA-256",
    "Candidate report": "候选报告",
    "Evidence supplied to this candidate": "候选使用的证据",
    "Obligation coverage": "回答义务覆盖",
    "Claim and citation review": "Claim 与引用审核",
    "Reviewer notes": "评审备注",
    "Record the concrete reason for omissions, mismatches, or ambiguity.": "记录遗漏、不匹配或歧义的具体原因。",
    "Claim classifications incomplete": "Claim 分类尚未完成",
    "Candidate classification complete": "本候选已完成",
    "Previous": "上一个",
    "Next": "下一个",
    "View English original": "查看英文原文",
    "View Chinese translation": "查看中文辅助翻译",
    "Decision context": "决策背景",
    "Open source": "打开来源",
    "Counter-evidence acknowledged and reconciled": "已识别并处理反向证据",
    "Claim support": "Claim 支持情况",
    "Supports exact claim": "精确支持该 Claim",
    "Supported": "有证据支持",
    "Unsupported": "无证据支持",
    "Citation match": "引用匹配",
    "Mismatch": "不匹配",
    "Decision-irrelevant claim": "与决策无关",
    "Candidate ${": "候选 ${",
    '" - Complete"': '" - 已完成"',
    "Evidence:": "证据：",
    "Draft does not match this human review packet.": "草稿与当前人工盲审包不匹配。",
    "Unknown candidate": "未知候选",
    "Draft imported.": "草稿已导入。",
}


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="__HTML_LANG__">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blind Review - __EXPERIMENT_ID__</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --surface: #ffffff;
      --ink: #18202a;
      --muted: #5c6673;
      --line: #d8dde5;
      --accent: #176b5b;
      --accent-soft: #e5f3ef;
      --warning: #9a5a09;
      --warning-soft: #fff3db;
      --danger: #9b2c2c;
      --danger-soft: #fdeaea;
      --focus: #1666b1;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px;
      line-height: 1.5;
      letter-spacing: 0;
    }

    button, textarea, input { font: inherit; }
    button { cursor: pointer; }
    button:focus-visible, textarea:focus-visible, input:focus-visible {
      outline: 3px solid rgba(22, 102, 177, 0.25);
      outline-offset: 2px;
    }

    .app-header {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 64px;
      padding: 10px 20px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }

    .title-block { min-width: 0; }
    h1 { margin: 0; font-size: 18px; line-height: 1.25; }
    .experiment { color: var(--muted); overflow-wrap: anywhere; }
    .header-actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .progress { min-width: 132px; color: var(--muted); font-variant-numeric: tabular-nums; }

    .button {
      min-height: 36px;
      padding: 7px 12px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
    }
    .button:hover { border-color: #aeb7c3; background: #f8f9fb; }
    .button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .button.primary:hover { background: #11594c; }
    .button:disabled { opacity: 0.45; cursor: not-allowed; }

    .shell {
      display: grid;
      grid-template-columns: minmax(230px, 280px) minmax(0, 1fr);
      min-height: calc(100vh - 64px);
    }

    .sidebar {
      padding: 16px 12px;
      background: #edf0f4;
      border-right: 1px solid var(--line);
    }
    .sidebar-label { margin: 0 8px 8px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .task-list { display: grid; gap: 4px; }
    .task-button {
      width: 100%;
      display: grid;
      grid-template-columns: 28px minmax(0, 1fr) auto;
      align-items: center;
      gap: 8px;
      min-height: 44px;
      padding: 7px 8px;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--ink);
      text-align: left;
    }
    .task-button:hover { background: rgba(255,255,255,0.72); }
    .task-button.active { background: var(--surface); border-color: #b8c0ca; }
    .task-number { color: var(--muted); font-variant-numeric: tabular-nums; }
    .task-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .task-status { width: 9px; height: 9px; border-radius: 50%; background: #aeb6c1; }
    .task-status.partial { background: #d9952d; }
    .task-status.done { background: var(--accent); }

    main { min-width: 0; padding: 20px clamp(16px, 3vw, 42px) 48px; }
    .task-context { padding-bottom: 18px; border-bottom: 1px solid var(--line); }
    .task-context h2 { margin: 0 0 8px; font-size: 20px; line-height: 1.35; }
    .context { margin: 0; color: var(--muted); }
    .review-purpose { margin: 0 0 8px; color: var(--accent); font-weight: 600; }
    .rubric-line { margin-top: 12px; padding: 10px 12px; background: var(--warning-soft); border-left: 3px solid var(--warning); }
    .translation-note { margin-top: 10px; color: var(--muted); font-size: 12px; }
    .review-guidance { margin-top: 12px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
    .review-guidance summary { cursor: pointer; font-weight: 700; }
    .review-guidance ul { margin: 8px 0 0; padding-left: 20px; }
    .review-guidance li + li { margin-top: 4px; }
    .shortcut-label { margin: 10px 0 0; color: var(--warning); font-weight: 700; }

    .obligation-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 8px;
      margin-top: 14px;
    }
    .obligation {
      min-height: 72px;
      padding: 10px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .obligation-id { display: block; color: var(--accent); font-weight: 700; overflow-wrap: anywhere; }

    .candidate-tabs { display: flex; gap: 0; margin: 18px 0; }
    .candidate-tab {
      min-width: 120px;
      min-height: 38px;
      border: 1px solid var(--line);
      background: var(--surface);
      color: var(--ink);
    }
    .candidate-tab:first-child { border-radius: 6px 0 0 6px; }
    .candidate-tab:last-child { border-radius: 0 6px 6px 0; }
    .candidate-tab + .candidate-tab { border-left: 0; }
    .candidate-tab.active { background: var(--accent-soft); color: #0f5146; font-weight: 700; }

    .review-grid { display: grid; grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr); gap: 24px; }
    section { margin-bottom: 22px; }
    h3 { margin: 0 0 10px; font-size: 15px; }
    .report {
      margin: 0;
      max-height: 620px;
      overflow: auto;
      padding: 16px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      font: 13px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace;
    }

    .evidence-list { display: grid; gap: 8px; }
    .evidence-item { padding: 10px 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
    .evidence-head { display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }
    .evidence-id { color: var(--accent); font: 600 12px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .evidence-title { font-weight: 700; }
    .source-link { margin-left: auto; color: var(--focus); font-size: 12px; }
    .evidence-excerpt { margin: 5px 0 0; color: #374151; }

    .check-group { display: grid; gap: 8px; }
    .check-row { display: flex; gap: 9px; align-items: flex-start; padding: 9px 10px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
    .check-row input { margin-top: 3px; flex: 0 0 auto; }
    .check-detail { min-width: 0; }
    .check-id { color: var(--accent); font-weight: 700; overflow-wrap: anywhere; }
    .conflict-note { color: var(--warning); font-size: 12px; }

    .claim-list { display: grid; gap: 10px; }
    .claim-item { padding: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; }
    .claim-item.incomplete { border-color: #d9a441; }
    .claim-head { display: flex; justify-content: space-between; gap: 10px; }
    .claim-id { color: var(--accent); font: 600 12px ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .claim-evidence { color: var(--muted); font-size: 12px; text-align: right; overflow-wrap: anywhere; }
    .claim-text { margin: 8px 0 10px; }
    .field-label { margin: 8px 0 5px; color: var(--muted); font-size: 12px; font-weight: 700; }
    .choice-row { display: flex; gap: 14px; flex-wrap: wrap; }
    .choice { display: inline-flex; gap: 6px; align-items: center; }
    .choice input { margin: 0; }
    .irrelevant { margin-top: 10px; padding-top: 8px; border-top: 1px solid #e7eaf0; }

    textarea { width: 100%; min-height: 92px; resize: vertical; padding: 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); }
    .candidate-footer { display: flex; justify-content: space-between; align-items: center; gap: 12px; padding-top: 14px; border-top: 1px solid var(--line); }
    .candidate-state { color: var(--warning); }
    .candidate-state.done { color: var(--accent); font-weight: 700; }
    .footer-nav { display: flex; gap: 8px; }

    .notice { display: none; position: fixed; right: 18px; bottom: 18px; max-width: min(420px, calc(100vw - 36px)); padding: 12px 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); box-shadow: 0 8px 24px rgba(21, 28, 38, 0.16); }
    .notice.show { display: block; }
    .notice.error { color: var(--danger); border-color: #e5b2b2; background: var(--danger-soft); }
    .packet-hash { margin-top: 14px; color: var(--muted); font: 11px/1.4 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .hidden { display: none; }

    @media (max-width: 960px) {
      .shell { grid-template-columns: 1fr; }
      .sidebar { position: static; border-right: 0; border-bottom: 1px solid var(--line); }
      .task-list { grid-template-columns: repeat(5, minmax(0, 1fr)); }
      .task-button { grid-template-columns: 1fr; justify-items: center; min-height: 40px; }
      .task-name { display: none; }
      .review-grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 680px) {
      .app-header { position: static; align-items: flex-start; flex-direction: column; padding: 12px; }
      .header-actions { width: 100%; }
      .progress { width: 100%; }
      main { padding: 16px 12px 40px; }
      .task-list { grid-template-columns: repeat(5, minmax(44px, 1fr)); }
      .candidate-tabs { width: 100%; }
      .candidate-tab { flex: 1; }
      .candidate-footer { align-items: flex-start; flex-direction: column; }
      .footer-nav { width: 100%; }
      .footer-nav .button { flex: 1; }
    }
  </style>
</head>
<body>
  <header class="app-header">
    <div class="title-block">
      <h1>Blind Semantic Review</h1>
      <div class="experiment">__EXPERIMENT_ID__</div>
    </div>
    <div class="header-actions">
      <div class="progress" id="progressText">0 / 20 candidates complete</div>
      <button class="button __TRANSLATION_ONLY_CLASS__" id="languageButton" type="button">View English original</button>
      <button class="button" id="importButton" type="button">Import draft</button>
      <button class="button" id="draftButton" type="button">Download draft</button>
      <button class="button primary" id="exportButton" type="button" disabled>Export final</button>
      <input class="hidden" id="importFile" type="file" accept="application/json,.json">
    </div>
  </header>

  <div class="shell">
    <aside class="sidebar">
      <p class="sidebar-label">Tasks</p>
      <nav class="task-list" id="taskList" aria-label="Review tasks"></nav>
      <div class="packet-hash">Packet SHA-256<br>__PACKET_SHA256__</div>
    </aside>

    <main>
      <div class="task-context">
        <h2 id="question"></h2>
        <p class="review-purpose">Review A/B reports for obligation coverage, claim support, and citation match. Do not guess which system variant produced a candidate.</p>
        <p class="context" id="context"></p>
        <div class="rubric-line" id="acceptance"></div>
        <div class="translation-note __TRANSLATION_ONLY_CLASS__">中文内容为机器辅助翻译，不作为独立评分依据；如有歧义，请以英文原文为准。翻译模型：__TRANSLATOR_MODEL__。</div>
        <details class="review-guidance" open>
          <summary>Review criteria</summary>
          <ul id="rubricList"></ul>
          <div class="shortcut-label">Forbidden shortcuts</div>
          <ul id="shortcutList"></ul>
        </details>
        <div class="obligation-grid" id="obligationSummary"></div>
      </div>

      <div class="candidate-tabs" id="candidateTabs" role="tablist"></div>

      <div class="review-grid">
        <div>
          <section>
            <h3>Candidate report</h3>
            <pre class="report" id="report"></pre>
          </section>
          <section>
            <h3>Evidence supplied to this candidate</h3>
            <div class="evidence-list" id="evidenceList"></div>
          </section>
        </div>

        <div>
          <section>
            <h3>Obligation coverage</h3>
            <div class="check-group" id="coverageChecks"></div>
          </section>
          <section>
            <h3>Claim and citation review</h3>
            <div class="claim-list" id="claimList"></div>
          </section>
          <section>
            <h3>Reviewer notes</h3>
            <textarea id="notes" placeholder="Record the concrete reason for omissions, mismatches, or ambiguity."></textarea>
          </section>
          <div class="candidate-footer">
            <div class="candidate-state" id="candidateState">Claim classifications incomplete</div>
            <div class="footer-nav">
              <button class="button" id="previousButton" type="button">Previous</button>
              <button class="button primary" id="nextButton" type="button">Next</button>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>

  <div class="notice" id="notice" role="status"></div>

  <script>
    const packet = __PACKET_JSON__;
    const packetSha256 = "__PACKET_SHA256__";
    const locale = "__LOCALE__";
    const translationMap = __TRANSLATION_JSON__;
    const storageKey = `deepresearch-review:${packetSha256}`;
    const candidateKeys = packet.tasks.flatMap(task => task.candidates.map(candidate => `${task.task_id}::${candidate.candidate_id}`));
    let annotations = loadLocal();
    let taskIndex = 0;
    let candidateIndex = 0;
    let showOriginal = false;

    const byId = id => document.getElementById(id);
    const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[char]);
    const currentTask = () => packet.tasks[taskIndex];
    const currentCandidate = () => currentTask().candidates[candidateIndex];
    const currentKey = () => `${currentTask().task_id}::${currentCandidate().candidate_id}`;
    const contentText = value => locale === "zh-CN" && !showOriginal ? (translationMap[value] || value) : value;

    function emptyAnnotation(task, candidate) {
      return {
        task_id: task.task_id,
        candidate_id: candidate.candidate_id,
        covered_obligation_ids: [],
        supported_claim_ids: [],
        unsupported_claim_ids: [],
        citation_supported_claim_ids: [],
        citation_mismatched_claim_ids: [],
        irrelevant_claim_ids: [],
        conflict_handled_obligation_ids: [],
        notes: []
      };
    }

    function loadLocal() {
      try { return JSON.parse(localStorage.getItem(storageKey)) || {}; }
      catch { return {}; }
    }

    function saveLocal() {
      localStorage.setItem(storageKey, JSON.stringify(annotations));
      updateProgress();
    }

    function getAnnotation(task = currentTask(), candidate = currentCandidate()) {
      const key = `${task.task_id}::${candidate.candidate_id}`;
      if (!annotations[key]) annotations[key] = emptyAnnotation(task, candidate);
      return annotations[key];
    }

    function setMembership(field, id, checked) {
      const annotation = getAnnotation();
      const values = new Set(annotation[field]);
      checked ? values.add(id) : values.delete(id);
      annotation[field] = [...values];
      saveLocal();
    }

    function setExclusive(positiveField, negativeField, id, value) {
      const annotation = getAnnotation();
      annotation[positiveField] = annotation[positiveField].filter(item => item !== id);
      annotation[negativeField] = annotation[negativeField].filter(item => item !== id);
      annotation[value === "positive" ? positiveField : negativeField].push(id);
      saveLocal();
      updateCandidateState();
    }

    function isComplete(task, candidate) {
      const annotation = annotations[`${task.task_id}::${candidate.candidate_id}`];
      if (!annotation) return false;
      const classifiedClaims = new Set([...annotation.supported_claim_ids, ...annotation.unsupported_claim_ids]);
      const citedClaims = new Set(candidate.citations.map(item => item.claim_id));
      const classifiedCitations = new Set([...annotation.citation_supported_claim_ids, ...annotation.citation_mismatched_claim_ids]);
      return candidate.claims.every(claim => classifiedClaims.has(claim.id)) && [...citedClaims].every(id => classifiedCitations.has(id));
    }

    function taskStatus(task) {
      const complete = task.candidates.filter(candidate => isComplete(task, candidate)).length;
      return complete === task.candidates.length ? "done" : complete > 0 ? "partial" : "pending";
    }

    function render() {
      renderTaskList();
      const task = currentTask();
      const candidate = currentCandidate();
      const annotation = getAnnotation(task, candidate);
      byId("question").textContent = contentText(task.question);
      byId("context").textContent = `Decision context: ${contentText(task.decision_context)}`;
      byId("acceptance").textContent = task.acceptance_notes.map(contentText).join(" ");
      byId("rubricList").innerHTML = packet.rubric.map(item => `<li>${escapeHtml(contentText(item))}</li>`).join("");
      byId("shortcutList").innerHTML = task.forbidden_shortcuts.map(item => `<li>${escapeHtml(contentText(item))}</li>`).join("");
      byId("obligationSummary").innerHTML = task.obligations.map(item => `
        <div class="obligation"><span class="obligation-id">${escapeHtml(item.id)}</span>${escapeHtml(contentText(item.description))}</div>
      `).join("");
      renderCandidateTabs(task);
      byId("report").textContent = contentText(candidate.report);
      byId("evidenceList").innerHTML = candidate.evidence.map(item => `
        <div class="evidence-item">
          <div class="evidence-head"><span class="evidence-id">${escapeHtml(item.id)}</span><span class="evidence-title">${escapeHtml(contentText(item.title))}</span><a class="source-link" href="${escapeHtml(item.url)}" target="_blank" rel="noopener noreferrer">Open source</a></div>
          <p class="evidence-excerpt">${escapeHtml(contentText(item.excerpt))}</p>
        </div>
      `).join("");
      renderCoverage(task, annotation);
      renderClaims(candidate, annotation);
      byId("notes").value = annotation.notes.join("\n");
      byId("languageButton").textContent = showOriginal ? "View Chinese translation" : "View English original";
      updateCandidateState();
      updateProgress();
    }

    function renderTaskList() {
      byId("taskList").innerHTML = packet.tasks.map((task, index) => `
        <button class="task-button ${index === taskIndex ? "active" : ""}" type="button" data-task="${index}" title="${escapeHtml(contentText(task.question))}">
          <span class="task-number">${String(index + 1).padStart(2, "0")}</span>
          <span class="task-name">${escapeHtml(task.task_id)}</span>
          <span class="task-status ${taskStatus(task)}" aria-label="${taskStatus(task)}"></span>
        </button>
      `).join("");
      document.querySelectorAll("[data-task]").forEach(button => button.addEventListener("click", () => {
        taskIndex = Number(button.dataset.task);
        candidateIndex = 0;
        render();
        window.scrollTo({top: 0, behavior: "smooth"});
      }));
    }

    function renderCandidateTabs(task) {
      byId("candidateTabs").innerHTML = task.candidates.map((candidate, index) => `
        <button class="candidate-tab ${index === candidateIndex ? "active" : ""}" type="button" role="tab" data-candidate="${index}">
          Candidate ${escapeHtml(candidate.candidate_id)}${isComplete(task, candidate) ? " - Complete" : ""}
        </button>
      `).join("");
      document.querySelectorAll("[data-candidate]").forEach(button => button.addEventListener("click", () => {
        candidateIndex = Number(button.dataset.candidate);
        render();
      }));
    }

    function renderCoverage(task, annotation) {
      byId("coverageChecks").innerHTML = task.obligations.map(obligation => {
        const hasConflict = obligation.counter_evidence_ids.length > 0;
        return `
          <div class="check-row">
            <input type="checkbox" id="covered-${escapeHtml(obligation.id)}" data-covered="${escapeHtml(obligation.id)}" ${annotation.covered_obligation_ids.includes(obligation.id) ? "checked" : ""}>
            <div class="check-detail">
              <label for="covered-${escapeHtml(obligation.id)}"><span class="check-id">${escapeHtml(obligation.id)}</span> ${escapeHtml(contentText(obligation.description))}</label>
              ${hasConflict ? `<div class="conflict-note"><label><input type="checkbox" data-conflict="${escapeHtml(obligation.id)}" ${annotation.conflict_handled_obligation_ids.includes(obligation.id) ? "checked" : ""}> Counter-evidence acknowledged and reconciled</label></div>` : ""}
            </div>
          </div>
        `;
      }).join("");
      document.querySelectorAll("[data-covered]").forEach(input => input.addEventListener("change", () => setMembership("covered_obligation_ids", input.dataset.covered, input.checked)));
      document.querySelectorAll("[data-conflict]").forEach(input => input.addEventListener("change", () => setMembership("conflict_handled_obligation_ids", input.dataset.conflict, input.checked)));
    }

    function renderClaims(candidate, annotation) {
      const citedClaimIds = new Set(candidate.citations.map(item => item.claim_id));
      byId("claimList").innerHTML = candidate.claims.map(claim => {
        const support = annotation.supported_claim_ids.includes(claim.id) ? "positive" : annotation.unsupported_claim_ids.includes(claim.id) ? "negative" : "";
        const citation = annotation.citation_supported_claim_ids.includes(claim.id) ? "positive" : annotation.citation_mismatched_claim_ids.includes(claim.id) ? "negative" : "";
        const complete = support && (!citedClaimIds.has(claim.id) || citation);
        return `
          <div class="claim-item ${complete ? "" : "incomplete"}" data-claim-card="${escapeHtml(claim.id)}">
            <div class="claim-head"><span class="claim-id">${escapeHtml(claim.id)}</span><span class="claim-evidence">Evidence: ${escapeHtml(claim.evidence_ids.join(", "))}</span></div>
            <div class="claim-text">${escapeHtml(contentText(claim.text))}</div>
            <div class="field-label">Claim support</div>
            <div class="choice-row">
              <label class="choice"><input type="radio" name="support-${escapeHtml(claim.id)}" data-support="${escapeHtml(claim.id)}" value="positive" ${support === "positive" ? "checked" : ""}> Supported</label>
              <label class="choice"><input type="radio" name="support-${escapeHtml(claim.id)}" data-support="${escapeHtml(claim.id)}" value="negative" ${support === "negative" ? "checked" : ""}> Unsupported</label>
            </div>
            ${citedClaimIds.has(claim.id) ? `
              <div class="field-label">Citation match</div>
              <div class="choice-row">
                <label class="choice"><input type="radio" name="citation-${escapeHtml(claim.id)}" data-citation="${escapeHtml(claim.id)}" value="positive" ${citation === "positive" ? "checked" : ""}> Supports exact claim</label>
                <label class="choice"><input type="radio" name="citation-${escapeHtml(claim.id)}" data-citation="${escapeHtml(claim.id)}" value="negative" ${citation === "negative" ? "checked" : ""}> Mismatch</label>
              </div>
            ` : ""}
            <div class="irrelevant"><label class="choice"><input type="checkbox" data-irrelevant="${escapeHtml(claim.id)}" ${annotation.irrelevant_claim_ids.includes(claim.id) ? "checked" : ""}> Decision-irrelevant claim</label></div>
          </div>
        `;
      }).join("");
      document.querySelectorAll("[data-support]").forEach(input => input.addEventListener("change", () => {
        setExclusive("supported_claim_ids", "unsupported_claim_ids", input.dataset.support, input.value);
        renderClaims(currentCandidate(), getAnnotation());
      }));
      document.querySelectorAll("[data-citation]").forEach(input => input.addEventListener("change", () => {
        setExclusive("citation_supported_claim_ids", "citation_mismatched_claim_ids", input.dataset.citation, input.value);
        renderClaims(currentCandidate(), getAnnotation());
      }));
      document.querySelectorAll("[data-irrelevant]").forEach(input => input.addEventListener("change", () => setMembership("irrelevant_claim_ids", input.dataset.irrelevant, input.checked)));
    }

    function updateCandidateState() {
      const complete = isComplete(currentTask(), currentCandidate());
      byId("candidateState").textContent = complete ? "Candidate classification complete" : "Claim classifications incomplete";
      byId("candidateState").className = `candidate-state ${complete ? "done" : ""}`;
    }

    function updateProgress() {
      const complete = packet.tasks.flatMap(task => task.candidates.map(candidate => isComplete(task, candidate))).filter(Boolean).length;
      byId("progressText").textContent = `${complete} / ${candidateKeys.length} candidates complete`;
      byId("exportButton").disabled = complete !== candidateKeys.length;
      renderTaskList();
    }

    function move(delta) {
      const flat = packet.tasks.flatMap((task, tIndex) => task.candidates.map((candidate, cIndex) => ({tIndex, cIndex})));
      const current = flat.findIndex(item => item.tIndex === taskIndex && item.cIndex === candidateIndex);
      const next = Math.max(0, Math.min(flat.length - 1, current + delta));
      taskIndex = flat[next].tIndex;
      candidateIndex = flat[next].cIndex;
      render();
      window.scrollTo({top: 0, behavior: "smooth"});
    }

    function submission() {
      return {
        experiment_id: packet.experiment_id,
        reviewer_type: "human",
        annotations: packet.tasks.flatMap(task => task.candidates.map(candidate => annotations[`${task.task_id}::${candidate.candidate_id}`] || emptyAnnotation(task, candidate)))
      };
    }

    function download(name) {
      const blob = new Blob([JSON.stringify(submission(), null, 2)], {type: "application/json"});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = name;
      link.click();
      URL.revokeObjectURL(link.href);
    }

    function showNotice(message, error = false) {
      const notice = byId("notice");
      notice.textContent = message;
      notice.className = `notice show ${error ? "error" : ""}`;
      window.setTimeout(() => notice.className = "notice", 3200);
    }

    byId("notes").addEventListener("input", event => {
      getAnnotation().notes = event.target.value.split("\n").map(item => item.trim()).filter(Boolean);
      saveLocal();
    });
    byId("previousButton").addEventListener("click", () => move(-1));
    byId("nextButton").addEventListener("click", () => move(1));
    byId("languageButton").addEventListener("click", () => {
      showOriginal = !showOriginal;
      render();
    });
    byId("draftButton").addEventListener("click", () => download("review_annotations.draft.json"));
    byId("exportButton").addEventListener("click", () => download("review_annotations.final.json"));
    byId("importButton").addEventListener("click", () => byId("importFile").click());
    byId("importFile").addEventListener("change", async event => {
      const file = event.target.files[0];
      if (!file) return;
      try {
        const imported = JSON.parse(await file.text());
        if (imported.experiment_id !== packet.experiment_id || imported.reviewer_type !== "human" || !Array.isArray(imported.annotations)) throw new Error("Draft does not match this human review packet.");
        const allowed = new Set(candidateKeys);
        const next = {};
        for (const annotation of imported.annotations) {
          const key = `${annotation.task_id}::${annotation.candidate_id}`;
          if (!allowed.has(key)) throw new Error(`Unknown candidate ${key}.`);
          next[key] = annotation;
        }
        annotations = next;
        saveLocal();
        render();
        showNotice("Draft imported.");
      } catch (error) {
        showNotice(error.message, true);
      } finally {
        event.target.value = "";
      }
    });

    render();
  </script>
</body>
</html>
"""
