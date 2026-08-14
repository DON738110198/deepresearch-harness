from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_plus import load_pi_browsecomp_run
from .dense_depth_probe import DenseDepthProbeRegistration, ProbeSourceArtifact
from .pi_browsecomp import PiSmokeSummary
from .wide_selector_probe import WideSelectorProbeResult


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BandwidthProblem(StrictContract):
    wide_selector_probe_path: str = Field(min_length=1)
    wide_selector_probe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    wide_selector_registration_path: str = Field(min_length=1)
    wide_selector_registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    best_fixed_top5_delta_pp: float
    dense_top20_pool_delta_pp: float
    diagnosis: str = Field(min_length=1)


class BandwidthMechanism(StrictContract):
    name: Literal["Evidence Bandwidth Exchange"]
    description: str = Field(min_length=1)
    result_count: int = Field(ge=6, le=20)
    minimum_snippet_tokens_per_result: int = Field(ge=1)
    allocation: Literal["equal_waterfill_rank_tiebreak"]
    payload_serialization: Literal["json_indent_2_utf8"]
    score_token_proxy: float = Field(allow_inf_nan=False)


class BandwidthContract(StrictContract):
    provider_calls: Literal[0]
    source_queries: Literal["frozen_agent_search_calls"]
    snippet_tokenizer: str = Field(min_length=1)
    snippet_tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_total_snippet_token_budgets: list[int] = Field(min_length=1)
    baseline: Literal["stored_bm25_top5_512_tokens_per_snippet"]
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    sealed_holdout_access: Literal["forbidden"]

    @model_validator(mode="after")
    def budgets_are_unique_and_ordered(self) -> "BandwidthContract":
        if self.candidate_total_snippet_token_budgets != sorted(
            set(self.candidate_total_snippet_token_budgets)
        ):
            raise ValueError("bandwidth budgets must be unique and increasing")
        return self


class BandwidthSelectionRule(StrictContract):
    maximum_aggregate_payload_token_ratio: float = Field(gt=0)
    maximum_each_trial_payload_token_ratio: float = Field(gt=0)
    selection: Literal["largest_registered_budget_passing_both_ratio_limits"]
    if_none_pass: Literal["do_not_run_live_wide_payload"]
    if_one_passes: Literal["freeze_selected_budget_before_live_generation"]


class EvidenceBandwidthRegistration(StrictContract):
    schema_version: Literal["browsecomp-plus-evidence-bandwidth-calibration-v0"]
    status: Literal["preregistered_not_run"]
    registered_at: str
    problem: BandwidthProblem
    mechanism: BandwidthMechanism
    fixed_contract: BandwidthContract
    selection_rule: BandwidthSelectionRule
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def minimum_allocation_fits_every_budget(self) -> "EvidenceBandwidthRegistration":
        minimum = (
            self.mechanism.result_count
            * self.mechanism.minimum_snippet_tokens_per_result
        )
        if any(
            budget < minimum
            for budget in self.fixed_contract.candidate_total_snippet_token_budgets
        ):
            raise ValueError("bandwidth budget cannot fund the minimum allocation")
        return self


class BandwidthTrialMetric(StrictContract):
    trial_id: str = Field(min_length=1)
    search_calls: int = Field(gt=0)
    baseline_payload_tokens: int = Field(gt=0)
    candidate_payload_tokens: int = Field(gt=0)
    payload_token_ratio: float = Field(gt=0)


class BandwidthBudgetMetric(StrictContract):
    total_snippet_token_budget: int = Field(gt=0)
    baseline_payload_tokens: int = Field(gt=0)
    candidate_payload_tokens: int = Field(gt=0)
    aggregate_payload_token_ratio: float = Field(gt=0)
    maximum_trial_payload_token_ratio: float = Field(gt=0)
    passed: bool
    trials: list[BandwidthTrialMetric] = Field(min_length=1)


class BandwidthRuntimeSnapshot(StrictContract):
    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    document_index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    unique_documents_tokenized: int = Field(gt=0)
    load_latency_ms: int = Field(ge=0)
    calibration_latency_ms: int = Field(ge=0)


class EvidenceBandwidthCalibrationResult(StrictContract):
    schema_version: Literal[
        "browsecomp-plus-evidence-bandwidth-calibration-result-v0"
    ] = "browsecomp-plus-evidence-bandwidth-calibration-result-v0"
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    registration: ProbeSourceArtifact
    wide_selector_probe: ProbeSourceArtifact
    wide_selector_registration: ProbeSourceArtifact
    runtime: BandwidthRuntimeSnapshot
    result_count: int = Field(gt=5)
    minimum_snippet_tokens_per_result: int = Field(gt=0)
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    search_calls: int = Field(gt=0)
    budgets: list[BandwidthBudgetMetric] = Field(min_length=1)
    decision: Literal[
        "freeze_selected_budget_before_live_generation",
        "do_not_run_live_wide_payload",
    ]
    selected_total_snippet_token_budget: int | None = Field(default=None, gt=0)
    claim_boundary: str = Field(min_length=1)


def calibrate_evidence_bandwidth(
    *,
    registration_path: Path,
    tokenizer_dir: Path,
    document_index_path: Path,
    output_path: Path,
) -> EvidenceBandwidthCalibrationResult:
    repository_root = _find_repository_root(registration_path.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("evidence bandwidth calibration output already exists")
    registration = EvidenceBandwidthRegistration.model_validate_json(
        registration_path.read_text(encoding="utf-8")
    )
    wide_path = _resolve_repository_file(
        registration.problem.wide_selector_probe_path,
        repository_root,
        required_parent="runs",
    )
    wide_registration_path = _resolve_repository_file(
        registration.problem.wide_selector_registration_path,
        repository_root,
        required_parent="benchmarks",
    )
    _require_hash(
        wide_path,
        registration.problem.wide_selector_probe_sha256,
        "wide selector probe",
    )
    _require_hash(
        wide_registration_path,
        registration.problem.wide_selector_registration_sha256,
        "wide selector registration",
    )
    wide = WideSelectorProbeResult.model_validate_json(
        wide_path.read_text(encoding="utf-8")
    )
    if wide.decision != "reject_fixed_top5_selectors":
        raise ValueError("bandwidth calibration requires rejected top-5 selectors")
    depth_registration_path = (
        repository_root / wide.depth_registration.path
    ).resolve()
    _require_hash(
        depth_registration_path,
        wide.depth_registration.sha256,
        "dense depth registration",
    )
    depth_registration = DenseDepthProbeRegistration.model_validate_json(
        depth_registration_path.read_text(encoding="utf-8")
    )
    if (
        wide.trial_count != registration.fixed_contract.trial_count
        or wide.queries_per_trial != registration.fixed_contract.queries_per_trial
    ):
        raise ValueError("bandwidth calibration source grid changed")

    started = time.perf_counter()
    try:
        from pyserini.search.lucene import LuceneSearcher
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            'install calibration dependencies with pip install -e ".[browsecomp-plus]"'
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir.resolve(), local_files_only=True, trust_remote_code=False
    )
    documents = LuceneSearcher(str(document_index_path.resolve()))
    load_latency_ms = round((time.perf_counter() - started) * 1000)

    ranking_by_query = {ranking.query: ranking for ranking in wide.rankings}
    doc_token_cache: dict[str, list[int]] = {}
    baseline_by_trial: dict[str, list[int]] = {}
    candidate_by_budget_trial: dict[int, dict[str, list[int]]] = {
        budget: {}
        for budget in registration.fixed_contract.candidate_total_snippet_token_budgets
    }
    calibration_started = time.perf_counter()
    for source in depth_registration.sources:
        summary_path = _validated_source_path(
            source.summary_path,
            source.summary_sha256,
            repository_root,
        )
        summary = PiSmokeSummary.model_validate_json(
            summary_path.read_text(encoding="utf-8")
        )
        baseline_tokens: list[int] = []
        candidate_tokens = {
            budget: []
            for budget in registration.fixed_contract.candidate_total_snippet_token_budgets
        }
        for item in summary.items:
            if item.run_sha256 is None:
                raise ValueError("bandwidth calibration source run is missing")
            run_path = summary_path.parent / _safe_id(item.query_id) / "run.json"
            _require_hash(run_path, item.run_sha256, "bandwidth source run")
            run = load_pi_browsecomp_run(run_path)
            for call in run.search_calls:
                ranking = ranking_by_query.get(call.query)
                if ranking is None:
                    raise ValueError("bandwidth calibration ranking is missing")
                baseline_payload = [
                    result.model_dump(mode="json") for result in call.results
                ]
                baseline_tokens.append(
                    _serialized_token_count(baseline_payload, tokenizer)
                )
                dense_docids = ranking.dense_docids[
                    : registration.mechanism.result_count
                ]
                if len(dense_docids) != registration.mechanism.result_count:
                    raise ValueError("bandwidth calibration ranking is too shallow")
                tokenized_docs = [
                    _document_tokens(
                        docid=docid,
                        documents=documents,
                        tokenizer=tokenizer,
                        cache=doc_token_cache,
                    )
                    for docid in dense_docids
                ]
                for budget in candidate_tokens:
                    caps = allocate_waterfill_caps(
                        [len(tokens) for tokens in tokenized_docs],
                        total_budget=budget,
                        minimum_per_result=(
                            registration.mechanism.minimum_snippet_tokens_per_result
                        ),
                    )
                    payload = [
                        {
                            "docid": docid,
                            "score": registration.mechanism.score_token_proxy,
                            "snippet": tokenizer.decode(
                                tokens[:cap], skip_special_tokens=True
                            ),
                        }
                        for docid, tokens, cap in zip(
                            dense_docids, tokenized_docs, caps, strict=True
                        )
                    ]
                    candidate_tokens[budget].append(
                        _serialized_token_count(payload, tokenizer)
                    )
        baseline_by_trial[source.trial_id] = baseline_tokens
        for budget, values in candidate_tokens.items():
            candidate_by_budget_trial[budget][source.trial_id] = values

    metrics = []
    for budget in registration.fixed_contract.candidate_total_snippet_token_budgets:
        trial_metrics = []
        for source in depth_registration.sources:
            baseline_total = sum(baseline_by_trial[source.trial_id])
            candidate_total = sum(
                candidate_by_budget_trial[budget][source.trial_id]
            )
            trial_metrics.append(
                BandwidthTrialMetric(
                    trial_id=source.trial_id,
                    search_calls=len(baseline_by_trial[source.trial_id]),
                    baseline_payload_tokens=baseline_total,
                    candidate_payload_tokens=candidate_total,
                    payload_token_ratio=round(
                        candidate_total / baseline_total, 9
                    ),
                )
            )
        baseline_total = sum(row.baseline_payload_tokens for row in trial_metrics)
        candidate_total = sum(row.candidate_payload_tokens for row in trial_metrics)
        aggregate_ratio = round(candidate_total / baseline_total, 9)
        maximum_trial_ratio = max(row.payload_token_ratio for row in trial_metrics)
        metrics.append(
            BandwidthBudgetMetric(
                total_snippet_token_budget=budget,
                baseline_payload_tokens=baseline_total,
                candidate_payload_tokens=candidate_total,
                aggregate_payload_token_ratio=aggregate_ratio,
                maximum_trial_payload_token_ratio=maximum_trial_ratio,
                passed=(
                    aggregate_ratio
                    <= registration.selection_rule.maximum_aggregate_payload_token_ratio
                    and maximum_trial_ratio
                    <= registration.selection_rule.maximum_each_trial_payload_token_ratio
                ),
                trials=trial_metrics,
            )
        )
    passing = [row for row in metrics if row.passed]
    selected_budget = (
        max(row.total_snippet_token_budget for row in passing) if passing else None
    )
    result = EvidenceBandwidthCalibrationResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=_source(registration_path, repository_root),
        wide_selector_probe=_source(wide_path, repository_root),
        wide_selector_registration=_source(
            wide_registration_path, repository_root
        ),
        runtime=BandwidthRuntimeSnapshot(
            tokenizer_name=registration.fixed_contract.snippet_tokenizer,
            tokenizer_revision=(
                registration.fixed_contract.snippet_tokenizer_revision
            ),
            document_index_revision=(
                wide.bm25_runtime.index_revision
            ),
            unique_documents_tokenized=len(doc_token_cache),
            load_latency_ms=load_latency_ms,
            calibration_latency_ms=round(
                (time.perf_counter() - calibration_started) * 1000
            ),
        ),
        result_count=registration.mechanism.result_count,
        minimum_snippet_tokens_per_result=(
            registration.mechanism.minimum_snippet_tokens_per_result
        ),
        trial_count=registration.fixed_contract.trial_count,
        queries_per_trial=registration.fixed_contract.queries_per_trial,
        search_calls=sum(len(values) for values in baseline_by_trial.values()),
        budgets=metrics,
        decision=(
            registration.selection_rule.if_one_passes
            if selected_budget is not None
            else registration.selection_rule.if_none_pass
        ),
        selected_total_snippet_token_budget=selected_budget,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return result


def allocate_waterfill_caps(
    lengths: Sequence[int], *, total_budget: int, minimum_per_result: int
) -> list[int]:
    if not lengths or any(length < 0 for length in lengths):
        raise ValueError("waterfill lengths must be non-negative and non-empty")
    if minimum_per_result < 1:
        raise ValueError("waterfill minimum must be positive")
    minimum_required = minimum_per_result * len(lengths)
    if total_budget < minimum_required:
        raise ValueError("waterfill budget cannot fund every result")
    caps = [min(length, minimum_per_result) for length in lengths]
    remaining = total_budget - sum(caps)
    while remaining > 0:
        progressed = False
        for index, length in enumerate(lengths):
            if caps[index] < length:
                caps[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return caps


def _document_tokens(*, docid: str, documents: object, tokenizer: object, cache: dict[str, list[int]]) -> list[int]:
    cached = cache.get(docid)
    if cached is not None:
        return cached
    document = documents.doc(docid)  # type: ignore[attr-defined]
    if document is None:
        raise ValueError(f"bandwidth document store lacks {docid}")
    raw = json.loads(document.raw())
    tokens = tokenizer.encode(  # type: ignore[attr-defined]
        raw["contents"], add_special_tokens=False
    )
    cache[docid] = tokens
    return tokens


def _serialized_token_count(payload: object, tokenizer: object) -> int:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return len(tokenizer.encode(text, add_special_tokens=False))  # type: ignore[attr-defined]


def _validated_source_path(
    value: str, expected_hash: str, repository_root: Path
) -> Path:
    path = _resolve_repository_file(value, repository_root, required_parent="runs")
    _require_hash(path, expected_hash, value)
    return path


def _resolve_repository_file(
    value: str, repository_root: Path, *, required_parent: str
) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("bandwidth source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to((repository_root / required_parent).resolve()):
        raise ValueError(f"bandwidth source leaves {required_parent}/")
    if not resolved.is_file():
        raise ValueError(f"bandwidth source is missing: {value}")
    return resolved


def _source(path: Path, repository_root: Path) -> ProbeSourceArtifact:
    resolved = path.resolve()
    return ProbeSourceArtifact(
        path=resolved.relative_to(repository_root.resolve()).as_posix(),
        sha256=_file_sha256(resolved),
    )


def _require_hash(path: Path, expected_hash: str, label: str) -> None:
    if _file_sha256(path) != expected_hash:
        raise ValueError(f"bandwidth source hash changed: {label}")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("bandwidth artifacts must stay under ignored runs/")


def _find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "runs").is_dir():
            return candidate
    raise ValueError("could not locate repository root")


def _safe_id(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    if not safe or safe in {".", ".."}:
        raise ValueError("query ID cannot be represented safely")
    return safe
