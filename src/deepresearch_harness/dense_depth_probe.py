from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import load_pi_browsecomp_run, normalized_text_file_sha256
from .pi_browsecomp import PiSmokeSummary
from .retrieval_replay import (
    DenseRuntimeSnapshot,
    QwenDenseReplaySearcher,
    RankedHit,
    RetrievalReplaySummary,
    collect_frozen_search_queries,
    load_retriever_candidates,
)


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DepthProbeProblem(StrictContract):
    decision_path: str = Field(min_length=1)
    decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_pool_probe_path: str = Field(min_length=1)
    retrieval_pool_probe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    live_dense_evidence_recall_delta_pp: float
    frozen_query_dense_top5_delta_pp: float
    frozen_query_rrf_top5_delta_pp: float
    bm25_dense_union_pool_delta_pp: float
    diagnosis: str = Field(min_length=1)


class DepthProbeContract(StrictContract):
    provider_calls: Literal[0]
    source_queries: Literal["frozen_agent_search_calls"]
    model: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    retriever_manifest: str = Field(min_length=1)
    retriever_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    depths: list[int] = Field(min_length=2)
    baseline: Literal["stored_bm25_top5"]
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    sealed_holdout_access: Literal["forbidden"]

    @model_validator(mode="after")
    def depths_are_frozen_and_ordered(self) -> "DepthProbeContract":
        if self.depths != sorted(set(self.depths)):
            raise ValueError("dense probe depths must be unique and increasing")
        if self.depths[0] != 5 or self.depths[-1] > 100:
            raise ValueError("dense probe must reproduce top-5 and stay at depth <= 100")
        return self


class DepthProbeSource(StrictContract):
    trial_id: str = Field(min_length=1)
    summary_path: str = Field(min_length=1)
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gold_path: str = Field(min_length=1)
    gold_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top5_replay_path: str = Field(min_length=1)
    top5_replay_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DepthProbeDecisionRule(StrictContract):
    top5_reproduction_tolerance_pp: float = Field(ge=0)
    minimum_depth20_delta_pp: float
    minimum_depth50_delta_pp: float
    if_depth20_passes: Literal[
        "select_candidate_generation_plus_reranking_as_next_layer"
    ]
    if_only_depth50_passes: Literal[
        "test_depth_efficiency_and_reranking_before_live_generation"
    ]
    if_depth50_fails: Literal[
        "reject_depth_expansion_and_diagnose_query_or_corpus_mismatch"
    ]


class DenseDepthProbeRegistration(StrictContract):
    schema_version: Literal["browsecomp-plus-dense-depth-probe-v0"]
    status: Literal["preregistered_not_run"]
    registered_at: str
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    problem: DepthProbeProblem
    hypothesis: str = Field(min_length=1)
    fixed_contract: DepthProbeContract
    sources: list[DepthProbeSource] = Field(min_length=1)
    decision_rule: DepthProbeDecisionRule
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def source_grid_matches_contract(self) -> "DenseDepthProbeRegistration":
        if len(self.sources) != self.fixed_contract.trial_count:
            raise ValueError("dense probe trial count differs from source grid")
        trial_ids = [source.trial_id for source in self.sources]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("dense probe trial IDs must be unique")
        required_depths = {20, 50}
        if not required_depths.issubset(self.fixed_contract.depths):
            raise ValueError("dense probe decision requires depths 20 and 50")
        return self


class ProbeSourceArtifact(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DepthObservation(StrictContract):
    depth: int = Field(gt=0)
    retrieved_unique: int = Field(ge=0)
    evidence_recall: float = Field(ge=0, le=1)
    gold_recall: float = Field(ge=0, le=1)


class DepthProbeQueryRow(StrictContract):
    trial_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    search_calls: int = Field(ge=0)
    baseline: DepthObservation
    dense: list[DepthObservation] = Field(min_length=2)
    first_evidence_rescue_depth: int | None = Field(default=None, gt=5)
    no_relevant_doc_at_max_depth: bool


class DepthProbeMetrics(StrictContract):
    depth: int = Field(gt=0)
    evidence_recall_percent: float = Field(ge=0, le=100)
    gold_recall_percent: float = Field(ge=0, le=100)
    evidence_delta_vs_bm25_pp: float
    zero_evidence_recall_queries: int = Field(ge=0)
    no_relevant_doc_queries: int = Field(ge=0)


class DepthProbeTrial(StrictContract):
    trial_id: str = Field(min_length=1)
    query_count: int = Field(gt=0)
    search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    baseline_gold_recall_percent: float = Field(ge=0, le=100)
    depth_metrics: list[DepthProbeMetrics] = Field(min_length=2)
    top5_exact_reproduction: Literal[True]
    rows: list[DepthProbeQueryRow] = Field(min_length=1)


DepthProbeDecision = Literal[
    "select_candidate_generation_plus_reranking_as_next_layer",
    "test_depth_efficiency_and_reranking_before_live_generation",
    "reject_depth_expansion_and_diagnose_query_or_corpus_mismatch",
]


class DenseDepthProbeResult(StrictContract):
    schema_version: Literal["browsecomp-plus-dense-depth-probe-result-v0"] = (
        "browsecomp-plus-dense-depth-probe-result-v0"
    )
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    candidate_id: str = Field(min_length=1)
    registration: ProbeSourceArtifact
    target_manifest: ProbeSourceArtifact
    retriever_manifest: ProbeSourceArtifact
    supporting_sources: dict[str, ProbeSourceArtifact]
    runtime: DenseRuntimeSnapshot
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    total_search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    baseline_gold_recall_percent: float = Field(ge=0, le=100)
    depth_metrics: list[DepthProbeMetrics] = Field(min_length=2)
    top5_exact_reproduction: Literal[True]
    observations_rescued_after_top5: dict[str, int]
    persistent_no_relevant_doc_query_ids: list[str]
    decision: DepthProbeDecision
    next_action: str = Field(min_length=1)
    trials: list[DepthProbeTrial] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


def run_dense_depth_probe(
    *,
    registration_path: Path,
    target_manifest_path: Path,
    model_dir: Path,
    index_root: Path,
    output_path: Path,
    batch_size: int = 8,
) -> DenseDepthProbeResult:
    repository_root = _find_repository_root(registration_path.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("dense depth probe output already exists")

    registration = DenseDepthProbeRegistration.model_validate_json(
        registration_path.read_text(encoding="utf-8")
    )
    if registration.target_manifest_sha256 != normalized_text_file_sha256(
        target_manifest_path
    ):
        raise ValueError("dense depth probe targets another benchmark manifest")
    retriever_manifest_path = _resolve_repository_file(
        registration.fixed_contract.retriever_manifest,
        repository_root,
        required_parent="benchmarks",
    )
    if registration.fixed_contract.retriever_manifest_sha256 != (
        normalized_text_file_sha256(retriever_manifest_path)
    ):
        raise ValueError("dense depth probe retriever manifest hash changed")

    retriever_manifest = load_retriever_candidates(
        retriever_manifest_path, target_manifest_path=target_manifest_path
    )
    candidates = [
        candidate
        for candidate in retriever_manifest.candidates
        if candidate.model.name == registration.fixed_contract.model
        and candidate.model.revision == registration.fixed_contract.model_revision
    ]
    if len(candidates) != 1:
        raise ValueError("dense depth probe model does not identify one candidate")
    source_candidate = candidates[0]
    if source_candidate.top_k != registration.fixed_contract.depths[0]:
        raise ValueError("dense depth probe no longer starts from frozen top-5")

    supporting_sources = _validate_supporting_sources(
        registration=registration,
        repository_root=repository_root,
    )
    resolved_sources = [
        _ResolvedSource(
            contract=source,
            summary_path=_resolve_repository_file(
                source.summary_path, repository_root, required_parent="runs"
            ),
            gold_path=_resolve_repository_file(
                source.gold_path, repository_root, required_parent="runs"
            ),
            replay_path=_resolve_repository_file(
                source.top5_replay_path, repository_root, required_parent="runs"
            ),
        )
        for source in registration.sources
    ]
    frozen_queries: list[str] = []
    for source in resolved_sources:
        frozen_queries.extend(collect_frozen_search_queries(source.summary_path.parent))
    unique_queries = list(dict.fromkeys(frozen_queries))

    max_depth = registration.fixed_contract.depths[-1]
    search_candidate = source_candidate.model_copy(update={"top_k": max_depth})
    searcher = QwenDenseReplaySearcher(
        candidate=search_candidate,
        model_dir=model_dir,
        index_root=index_root,
        batch_size=batch_size,
    )
    candidate_results = searcher.search_many(unique_queries)
    if set(candidate_results) != set(unique_queries):
        raise ValueError("dense depth probe did not return every frozen query")

    trials = [
        _score_trial(
            source=source,
            candidate_results=candidate_results,
            depths=registration.fixed_contract.depths,
            tolerance_pp=(
                registration.decision_rule.top5_reproduction_tolerance_pp
            ),
        )
        for source in resolved_sources
    ]
    if any(trial.query_count != registration.fixed_contract.queries_per_trial for trial in trials):
        raise ValueError("dense depth probe query count differs from registration")

    aggregate_baseline_evidence = _trial_mean(
        trials, "baseline_evidence_recall_percent"
    )
    aggregate_baseline_gold = _trial_mean(trials, "baseline_gold_recall_percent")
    depth_metrics = []
    for depth in registration.fixed_contract.depths:
        rows = [_metric_at_depth(trial, depth) for trial in trials]
        evidence = round(mean(row.evidence_recall_percent for row in rows), 6)
        depth_metrics.append(
            DepthProbeMetrics(
                depth=depth,
                evidence_recall_percent=evidence,
                gold_recall_percent=round(
                    mean(row.gold_recall_percent for row in rows), 6
                ),
                evidence_delta_vs_bm25_pp=round(
                    evidence - aggregate_baseline_evidence, 6
                ),
                zero_evidence_recall_queries=sum(
                    row.zero_evidence_recall_queries for row in rows
                ),
                no_relevant_doc_queries=sum(
                    row.no_relevant_doc_queries for row in rows
                ),
            )
        )
    top5_delta = _metric_at_depth_list(depth_metrics, 5).evidence_delta_vs_bm25_pp
    if abs(top5_delta - registration.problem.frozen_query_dense_top5_delta_pp) > (
        registration.decision_rule.top5_reproduction_tolerance_pp
    ):
        raise ValueError("dense top-5 aggregate no longer reproduces preregistration")

    decision = choose_depth_probe_decision(
        depth20_delta_pp=_metric_at_depth_list(
            depth_metrics, 20
        ).evidence_delta_vs_bm25_pp,
        depth50_delta_pp=_metric_at_depth_list(
            depth_metrics, 50
        ).evidence_delta_vs_bm25_pp,
        rule=registration.decision_rule,
    )
    rescue_counts: dict[str, int] = {
        str(depth): sum(
            row.first_evidence_rescue_depth == depth
            for trial in trials
            for row in trial.rows
        )
        for depth in registration.fixed_contract.depths[1:]
    }
    rows_by_query: dict[str, list[DepthProbeQueryRow]] = defaultdict(list)
    for trial in trials:
        for row in trial.rows:
            rows_by_query[row.query_id].append(row)
    persistent = sorted(
        query_id
        for query_id, rows in rows_by_query.items()
        if len(rows) == registration.fixed_contract.trial_count
        and all(row.no_relevant_doc_at_max_depth for row in rows)
    )
    result = DenseDepthProbeResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        candidate_id=source_candidate.candidate_id,
        registration=_source_artifact(registration_path, repository_root),
        target_manifest=_source_artifact(target_manifest_path, repository_root),
        retriever_manifest=_source_artifact(
            retriever_manifest_path, repository_root, normalized_text=True
        ),
        supporting_sources=supporting_sources,
        runtime=searcher.runtime_snapshot,
        trial_count=len(trials),
        queries_per_trial=registration.fixed_contract.queries_per_trial,
        total_search_calls=sum(trial.search_calls for trial in trials),
        unique_search_queries=len(unique_queries),
        baseline_evidence_recall_percent=aggregate_baseline_evidence,
        baseline_gold_recall_percent=aggregate_baseline_gold,
        depth_metrics=depth_metrics,
        top5_exact_reproduction=True,
        observations_rescued_after_top5=rescue_counts,
        persistent_no_relevant_doc_query_ids=persistent,
        decision=decision,
        next_action=decision,
        trials=trials,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return result


def choose_depth_probe_decision(
    *,
    depth20_delta_pp: float,
    depth50_delta_pp: float,
    rule: DepthProbeDecisionRule,
) -> DepthProbeDecision:
    if depth20_delta_pp >= rule.minimum_depth20_delta_pp:
        return rule.if_depth20_passes
    if depth50_delta_pp >= rule.minimum_depth50_delta_pp:
        return rule.if_only_depth50_passes
    return rule.if_depth50_fails


class _ResolvedSource:
    def __init__(
        self,
        *,
        contract: DepthProbeSource,
        summary_path: Path,
        gold_path: Path,
        replay_path: Path,
    ) -> None:
        self.contract = contract
        self.summary_path = summary_path
        self.gold_path = gold_path
        self.replay_path = replay_path


def _score_trial(
    *,
    source: _ResolvedSource,
    candidate_results: Mapping[str, Sequence[RankedHit]],
    depths: Sequence[int],
    tolerance_pp: float,
) -> DepthProbeTrial:
    summary = PiSmokeSummary.model_validate_json(
        source.summary_path.read_text(encoding="utf-8")
    )
    gold = DevelopmentGoldSlice.model_validate_json(
        source.gold_path.read_text(encoding="utf-8")
    )
    replay = RetrievalReplaySummary.model_validate_json(
        source.replay_path.read_text(encoding="utf-8")
    )
    if gold.source_summary_sha256 != _file_sha256(source.summary_path):
        raise ValueError("dense depth probe gold targets another summary")
    if replay.source_summary_sha256 != _file_sha256(source.summary_path):
        raise ValueError("dense depth probe replay targets another summary")
    if replay.gold_slice_sha256 != _file_sha256(source.gold_path):
        raise ValueError("dense depth probe replay targets another gold slice")

    gold_by_id = {row.query_id: row for row in gold.rows}
    replay_by_id = {row.query_id: row for row in replay.rows}
    rows: list[DepthProbeQueryRow] = []
    for item in summary.items:
        if item.run_sha256 is None:
            raise ValueError("dense depth probe source run is missing")
        run_path = source.summary_path.parent / _safe_id(item.query_id) / "run.json"
        if _file_sha256(run_path) != item.run_sha256:
            raise ValueError("dense depth probe source run hash changed")
        run = load_pi_browsecomp_run(run_path)
        reference = gold_by_id.get(item.query_id)
        replay_row = replay_by_id.get(item.query_id)
        if reference is None or replay_row is None:
            raise ValueError("dense depth probe query lacks gold or replay")
        if len(run.search_calls) != len(replay_row.calls):
            raise ValueError("dense depth probe source calls differ from top-5 replay")

        baseline_docids: set[str] = set()
        dense_docids = {depth: set() for depth in depths}
        for call, replay_call in zip(run.search_calls, replay_row.calls, strict=True):
            hits = candidate_results.get(call.query)
            if hits is None or len(hits) < depths[-1]:
                raise ValueError("dense depth probe candidate depth is incomplete")
            top5 = [hit.docid for hit in hits[:5]]
            if call.query != replay_call.query or top5 != replay_call.candidate_docids:
                raise ValueError("dense depth probe failed exact top-5 reproduction")
            baseline_docids.update(result.docid for result in call.results)
            for depth in depths:
                dense_docids[depth].update(hit.docid for hit in hits[:depth])

        evidence = set(reference.evidence_docids)
        gold_docids = set(reference.gold_docids)
        baseline = _observation(5, baseline_docids, evidence, gold_docids)
        dense = [
            _observation(depth, dense_docids[depth], evidence, gold_docids)
            for depth in depths
        ]
        first_rescue = next(
            (
                observation.depth
                for observation in dense[1:]
                if dense[0].evidence_recall == 0
                and observation.evidence_recall > 0
            ),
            None,
        )
        rows.append(
            DepthProbeQueryRow(
                trial_id=source.contract.trial_id,
                query_id=item.query_id,
                search_calls=len(run.search_calls),
                baseline=baseline,
                dense=dense,
                first_evidence_rescue_depth=first_rescue,
                no_relevant_doc_at_max_depth=(
                    dense[-1].evidence_recall == 0 and dense[-1].gold_recall == 0
                ),
            )
        )

    baseline_evidence = _row_percent(rows, variant="baseline", metric="evidence")
    baseline_gold = _row_percent(rows, variant="baseline", metric="gold")
    depth_metrics = [_depth_metrics(rows, depth, baseline_evidence) for depth in depths]
    top5 = _metric_at_depth_list(depth_metrics, 5)
    if abs(top5.evidence_recall_percent - float(replay.candidate_evidence_recall_percent)) > tolerance_pp:
        raise ValueError("dense depth probe top-5 evidence recall changed")
    if abs(baseline_evidence - float(replay.baseline_evidence_recall_percent)) > tolerance_pp:
        raise ValueError("dense depth probe BM25 evidence recall changed")

    return DepthProbeTrial(
        trial_id=source.contract.trial_id,
        query_count=len(rows),
        search_calls=sum(row.search_calls for row in rows),
        unique_search_queries=len(collect_frozen_search_queries(source.summary_path.parent)),
        baseline_evidence_recall_percent=baseline_evidence,
        baseline_gold_recall_percent=baseline_gold,
        depth_metrics=depth_metrics,
        top5_exact_reproduction=True,
        rows=rows,
    )


def _depth_metrics(
    rows: Sequence[DepthProbeQueryRow], depth: int, baseline_evidence: float
) -> DepthProbeMetrics:
    observations = [_dense_observation(row, depth) for row in rows]
    evidence = round(mean(row.evidence_recall for row in observations) * 100, 2)
    return DepthProbeMetrics(
        depth=depth,
        evidence_recall_percent=evidence,
        gold_recall_percent=round(mean(row.gold_recall for row in observations) * 100, 2),
        evidence_delta_vs_bm25_pp=round(evidence - baseline_evidence, 2),
        zero_evidence_recall_queries=sum(row.evidence_recall == 0 for row in observations),
        no_relevant_doc_queries=sum(
            row.evidence_recall == 0 and row.gold_recall == 0 for row in observations
        ),
    )


def _observation(
    depth: int,
    retrieved: set[str],
    evidence: set[str],
    gold_docids: set[str],
) -> DepthObservation:
    return DepthObservation(
        depth=depth,
        retrieved_unique=len(retrieved),
        evidence_recall=_recall(retrieved, evidence),
        gold_recall=_recall(retrieved, gold_docids),
    )


def _recall(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def _row_percent(
    rows: Sequence[DepthProbeQueryRow], *, variant: str, metric: str
) -> float:
    if variant != "baseline":
        raise ValueError("only the stored baseline is supported")
    field = f"{metric}_recall"
    return round(mean(getattr(row.baseline, field) for row in rows) * 100, 2)


def _dense_observation(row: DepthProbeQueryRow, depth: int) -> DepthObservation:
    matches = [observation for observation in row.dense if observation.depth == depth]
    if len(matches) != 1:
        raise ValueError(f"dense depth probe lacks depth {depth}")
    return matches[0]


def _metric_at_depth(trial: DepthProbeTrial, depth: int) -> DepthProbeMetrics:
    return _metric_at_depth_list(trial.depth_metrics, depth)


def _metric_at_depth_list(
    metrics: Sequence[DepthProbeMetrics], depth: int
) -> DepthProbeMetrics:
    matches = [metric for metric in metrics if metric.depth == depth]
    if len(matches) != 1:
        raise ValueError(f"dense depth probe lacks aggregate depth {depth}")
    return matches[0]


def _trial_mean(trials: Sequence[DepthProbeTrial], field: str) -> float:
    return round(mean(float(getattr(trial, field)) for trial in trials), 6)


def _validate_supporting_sources(
    *,
    registration: DenseDepthProbeRegistration,
    repository_root: Path,
) -> dict[str, ProbeSourceArtifact]:
    records: dict[str, ProbeSourceArtifact] = {}
    problem_sources = (
        (
            "dense_confirmation_decision",
            registration.problem.decision_path,
            registration.problem.decision_sha256,
        ),
        (
            "retrieval_pool_probe",
            registration.problem.retrieval_pool_probe_path,
            registration.problem.retrieval_pool_probe_sha256,
        ),
    )
    for name, value, expected_hash in problem_sources:
        path = _resolve_repository_file(value, repository_root, required_parent="runs")
        if _file_sha256(path) != expected_hash:
            raise ValueError(f"dense depth probe supporting source changed: {name}")
        records[name] = _source_artifact(path, repository_root)
    for source in registration.sources:
        for kind, value, expected_hash in (
            ("summary", source.summary_path, source.summary_sha256),
            ("gold", source.gold_path, source.gold_sha256),
            ("top5_replay", source.top5_replay_path, source.top5_replay_sha256),
        ):
            name = f"{source.trial_id}_{kind}"
            path = _resolve_repository_file(value, repository_root, required_parent="runs")
            if _file_sha256(path) != expected_hash:
                raise ValueError(f"dense depth probe source changed: {name}")
            records[name] = _source_artifact(path, repository_root)
    return records


def _resolve_repository_file(
    value: str, repository_root: Path, *, required_parent: str
) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("dense depth probe source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    parent = (repository_root / required_parent).resolve()
    if not resolved.is_relative_to(parent) or not resolved.is_file():
        raise ValueError(f"dense depth probe source is invalid: {value}")
    return resolved


def _source_artifact(
    path: Path,
    repository_root: Path,
    *,
    normalized_text: bool = False,
) -> ProbeSourceArtifact:
    resolved = path.resolve()
    return ProbeSourceArtifact(
        path=resolved.relative_to(repository_root.resolve()).as_posix(),
        sha256=(
            normalized_text_file_sha256(resolved)
            if normalized_text
            else _file_sha256(resolved)
        ),
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("dense depth probe artifacts must stay under ignored runs/")


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
