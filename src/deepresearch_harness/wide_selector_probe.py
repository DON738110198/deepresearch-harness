from __future__ import annotations

import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .browsecomp_evaluation import DevelopmentGoldSlice
from .browsecomp_plus import (
    load_browsecomp_plus_target,
    load_pi_browsecomp_run,
    normalized_text_file_sha256,
)
from .dense_depth_probe import (
    DenseDepthProbeRegistration,
    DenseDepthProbeResult,
    ProbeSourceArtifact,
)
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


class WideSelectorProblem(StrictContract):
    dense_depth_probe_path: str = Field(min_length=1)
    dense_depth_probe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_depth_registration_path: str = Field(min_length=1)
    dense_depth_registration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_dense_top5_delta_pp: float
    observed_dense_top20_pool_delta_pp: float
    diagnosis: str = Field(min_length=1)


class WideSelectorContract(StrictContract):
    provider_calls: Literal[0]
    source_queries: Literal["frozen_agent_search_calls"]
    target_manifest: str = Field(min_length=1)
    target_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retriever_manifest: str = Field(min_length=1)
    retriever_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dense_model: str = Field(min_length=1)
    dense_model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    bm25_index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_depth: int = Field(ge=5, le=100)
    output_depth: int = Field(ge=1, le=20)
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    sealed_holdout_access: Literal["forbidden"]


class SelectorSpec(StrictContract):
    selector_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    kind: Literal[
        "weighted_reciprocal_rank_fusion", "fixed_dense_rank_portfolio"
    ]
    bm25_weight: float | None = Field(default=None, gt=0)
    dense_weight: float | None = Field(default=None, gt=0)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    dense_ranks: list[int] | None = None

    @model_validator(mode="after")
    def parameters_match_kind(self) -> "SelectorSpec":
        if self.kind == "weighted_reciprocal_rank_fusion":
            if (
                self.bm25_weight is None
                or self.dense_weight is None
                or self.rrf_k is None
                or self.dense_ranks is not None
            ):
                raise ValueError("weighted RRF selector parameters are incomplete")
        elif (
            self.dense_ranks is None
            or self.bm25_weight is not None
            or self.dense_weight is not None
            or self.rrf_k is not None
        ):
            raise ValueError("dense portfolio selector parameters are incomplete")
        return self


class WideSelectorDecisionRule(StrictContract):
    minimum_evidence_recall_delta_pp: float
    minimum_query_wins_minus_losses: int
    selection_metric: Literal["evidence_recall_percent"]
    tie_break_priority: list[str] = Field(min_length=1)
    if_any_selector_passes: Literal[
        "preregister_fresh_slice_live_confirmation_for_selected_fixed_top5_selector"
    ]
    if_all_selectors_fail: Literal[
        "preregister_token_matched_wide_result_payload_experiment"
    ]


class WideSelectorRegistration(StrictContract):
    schema_version: Literal["browsecomp-plus-wide-selector-probe-v0"]
    status: Literal["preregistered_not_run"]
    registered_at: str
    problem: WideSelectorProblem
    hypothesis: str = Field(min_length=1)
    fixed_contract: WideSelectorContract
    selectors: list[SelectorSpec] = Field(min_length=1)
    decision_rule: WideSelectorDecisionRule
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def selector_grid_is_frozen(self) -> "WideSelectorRegistration":
        ids = [selector.selector_id for selector in self.selectors]
        if len(ids) != len(set(ids)):
            raise ValueError("wide selector IDs must be unique")
        if set(ids) != set(self.decision_rule.tie_break_priority):
            raise ValueError("wide selector priority must cover the frozen grid")
        for selector in self.selectors:
            if selector.dense_ranks is not None:
                if (
                    len(selector.dense_ranks) != self.fixed_contract.output_depth
                    or selector.dense_ranks != sorted(set(selector.dense_ranks))
                    or selector.dense_ranks[-1]
                    > self.fixed_contract.candidate_depth
                ):
                    raise ValueError("dense portfolio ranks differ from output contract")
        return self


class SelectorRanking(StrictContract):
    query: str = Field(min_length=1)
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bm25_docids: list[str]
    dense_docids: list[str]
    selected_docids: dict[str, list[str]]


class SelectorObservation(StrictContract):
    selector_id: str = Field(min_length=1)
    retrieved_unique: int = Field(ge=0)
    evidence_recall: float = Field(ge=0, le=1)
    gold_recall: float = Field(ge=0, le=1)


class SelectorQueryRow(StrictContract):
    trial_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    search_calls: int = Field(ge=0)
    baseline_evidence_recall: float = Field(ge=0, le=1)
    baseline_gold_recall: float = Field(ge=0, le=1)
    selectors: list[SelectorObservation] = Field(min_length=1)


class SelectorTrialMetrics(StrictContract):
    selector_id: str = Field(min_length=1)
    evidence_recall_percent: float = Field(ge=0, le=100)
    gold_recall_percent: float = Field(ge=0, le=100)
    evidence_delta_vs_bm25_pp: float
    query_wins: int = Field(ge=0)
    query_losses: int = Field(ge=0)
    query_ties: int = Field(ge=0)
    no_relevant_doc_queries: int = Field(ge=0)


class WideSelectorTrial(StrictContract):
    trial_id: str = Field(min_length=1)
    query_count: int = Field(gt=0)
    search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    baseline_gold_recall_percent: float = Field(ge=0, le=100)
    selector_metrics: list[SelectorTrialMetrics] = Field(min_length=1)
    bm25_top5_exact_reproduction: Literal[True]
    dense_top5_exact_reproduction: Literal[True]
    rows: list[SelectorQueryRow] = Field(min_length=1)


class SelectorAggregate(StrictContract):
    selector_id: str = Field(min_length=1)
    evidence_recall_percent: float = Field(ge=0, le=100)
    gold_recall_percent: float = Field(ge=0, le=100)
    evidence_delta_vs_bm25_pp: float
    query_wins: int = Field(ge=0)
    query_losses: int = Field(ge=0)
    query_ties: int = Field(ge=0)
    query_wins_minus_losses: int
    no_relevant_doc_queries: int = Field(ge=0)
    passed: bool


class BM25RuntimeSnapshot(StrictContract):
    index_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    candidate_depth: int = Field(gt=0)
    load_latency_ms: int = Field(ge=0)
    search_latency_ms: int = Field(ge=0)


class WideSelectorProbeResult(StrictContract):
    schema_version: Literal["browsecomp-plus-wide-selector-probe-result-v0"] = (
        "browsecomp-plus-wide-selector-probe-result-v0"
    )
    created_at: str
    status: Literal["diagnostic_not_official"] = "diagnostic_not_official"
    provider_calls: Literal[0] = 0
    sealed_holdout_accessed: Literal[False] = False
    registration: ProbeSourceArtifact
    depth_probe: ProbeSourceArtifact
    depth_registration: ProbeSourceArtifact
    target_manifest: ProbeSourceArtifact
    retriever_manifest: ProbeSourceArtifact
    dense_runtime: DenseRuntimeSnapshot
    bm25_runtime: BM25RuntimeSnapshot
    trial_count: int = Field(gt=0)
    queries_per_trial: int = Field(gt=0)
    total_search_calls: int = Field(ge=0)
    unique_search_queries: int = Field(ge=0)
    baseline_evidence_recall_percent: float = Field(ge=0, le=100)
    baseline_gold_recall_percent: float = Field(ge=0, le=100)
    selector_metrics: list[SelectorAggregate] = Field(min_length=1)
    bm25_top5_exact_reproduction: Literal[True]
    dense_top5_exact_reproduction: Literal[True]
    decision: Literal["select_fixed_top5_selector", "reject_fixed_top5_selectors"]
    selected_selector_id: str | None = None
    next_action: str = Field(min_length=1)
    rankings: list[SelectorRanking] = Field(min_length=1)
    trials: list[WideSelectorTrial] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)


def run_wide_selector_probe(
    *,
    registration_path: Path,
    model_dir: Path,
    dense_index_root: Path,
    bm25_index_path: Path,
    output_path: Path,
    batch_size: int = 16,
) -> WideSelectorProbeResult:
    repository_root = _find_repository_root(registration_path.resolve())
    _require_under_runs(output_path, repository_root)
    if output_path.exists():
        raise ValueError("wide selector probe output already exists")
    registration = WideSelectorRegistration.model_validate_json(
        registration_path.read_text(encoding="utf-8")
    )
    target_manifest_path = _resolve_repository_file(
        registration.fixed_contract.target_manifest,
        repository_root,
        required_parent="benchmarks",
    )
    retriever_manifest_path = _resolve_repository_file(
        registration.fixed_contract.retriever_manifest,
        repository_root,
        required_parent="benchmarks",
    )
    if registration.fixed_contract.target_manifest_sha256 != (
        normalized_text_file_sha256(target_manifest_path)
    ):
        raise ValueError("wide selector target manifest changed")
    if registration.fixed_contract.retriever_manifest_sha256 != (
        normalized_text_file_sha256(retriever_manifest_path)
    ):
        raise ValueError("wide selector retriever manifest changed")
    target = load_browsecomp_plus_target(target_manifest_path)
    if (
        target.benchmark.index_dataset.revision
        != registration.fixed_contract.bm25_index_revision
    ):
        raise ValueError("wide selector BM25 index revision changed")

    depth_probe_path = _resolve_repository_file(
        registration.problem.dense_depth_probe_path,
        repository_root,
        required_parent="runs",
    )
    depth_registration_path = _resolve_repository_file(
        registration.problem.dense_depth_registration_path,
        repository_root,
        required_parent="benchmarks",
    )
    _require_hash(
        depth_probe_path,
        registration.problem.dense_depth_probe_sha256,
        "dense depth probe",
    )
    _require_hash(
        depth_registration_path,
        registration.problem.dense_depth_registration_sha256,
        "dense depth registration",
    )
    depth_probe = DenseDepthProbeResult.model_validate_json(
        depth_probe_path.read_text(encoding="utf-8")
    )
    depth_registration = DenseDepthProbeRegistration.model_validate_json(
        depth_registration_path.read_text(encoding="utf-8")
    )
    if (
        depth_probe.trial_count != registration.fixed_contract.trial_count
        or depth_probe.queries_per_trial
        != registration.fixed_contract.queries_per_trial
        or depth_probe.registration.sha256
        != registration.problem.dense_depth_registration_sha256
    ):
        raise ValueError("wide selector source grid differs from dense depth probe")

    retrievers = load_retriever_candidates(
        retriever_manifest_path, target_manifest_path=target_manifest_path
    )
    matches = [
        candidate
        for candidate in retrievers.candidates
        if candidate.model.name == registration.fixed_contract.dense_model
        and candidate.model.revision
        == registration.fixed_contract.dense_model_revision
    ]
    if len(matches) != 1:
        raise ValueError("wide selector dense model does not identify one candidate")
    dense_candidate = matches[0].model_copy(
        update={"top_k": registration.fixed_contract.candidate_depth}
    )

    resolved_sources = []
    frozen_queries: list[str] = []
    for source in depth_registration.sources:
        summary_path = _validated_source_path(
            source.summary_path,
            source.summary_sha256,
            repository_root,
        )
        gold_path = _validated_source_path(
            source.gold_path,
            source.gold_sha256,
            repository_root,
        )
        replay_path = _validated_source_path(
            source.top5_replay_path,
            source.top5_replay_sha256,
            repository_root,
        )
        resolved_sources.append((source.trial_id, summary_path, gold_path, replay_path))
        frozen_queries.extend(collect_frozen_search_queries(summary_path.parent))
    unique_queries = list(dict.fromkeys(frozen_queries))

    dense_searcher = QwenDenseReplaySearcher(
        candidate=dense_candidate,
        model_dir=model_dir,
        index_root=dense_index_root,
        batch_size=batch_size,
    )
    dense_results = dense_searcher.search_many(unique_queries)
    bm25_results, bm25_runtime = _search_bm25(
        queries=unique_queries,
        index_path=bm25_index_path,
        depth=registration.fixed_contract.candidate_depth,
        index_revision=registration.fixed_contract.bm25_index_revision,
    )
    rankings = [
        _ranking(
            query=query,
            bm25=bm25_results[query],
            dense=dense_results[query],
            selectors=registration.selectors,
            output_depth=registration.fixed_contract.output_depth,
        )
        for query in unique_queries
    ]
    ranking_by_query = {row.query: row for row in rankings}

    trials = [
        _score_trial(
            trial_id=trial_id,
            summary_path=summary_path,
            gold_path=gold_path,
            replay_path=replay_path,
            ranking_by_query=ranking_by_query,
            selectors=registration.selectors,
        )
        for trial_id, summary_path, gold_path, replay_path in resolved_sources
    ]
    baseline_evidence = _trial_mean(trials, "baseline_evidence_recall_percent")
    baseline_gold = _trial_mean(trials, "baseline_gold_recall_percent")
    selector_metrics = [
        _aggregate_selector(
            selector_id=selector.selector_id,
            trials=trials,
            baseline_evidence=baseline_evidence,
            minimum_delta=registration.decision_rule.minimum_evidence_recall_delta_pp,
            minimum_wins_minus_losses=(
                registration.decision_rule.minimum_query_wins_minus_losses
            ),
        )
        for selector in registration.selectors
    ]
    passing = [metric for metric in selector_metrics if metric.passed]
    priority = {
        selector_id: index
        for index, selector_id in enumerate(
            registration.decision_rule.tie_break_priority
        )
    }
    selected = (
        sorted(
            passing,
            key=lambda row: (-row.evidence_recall_percent, priority[row.selector_id]),
        )[0]
        if passing
        else None
    )
    result = WideSelectorProbeResult(
        created_at=datetime.now(timezone.utc).isoformat(),
        registration=_source(registration_path, repository_root),
        depth_probe=_source(depth_probe_path, repository_root),
        depth_registration=_source(depth_registration_path, repository_root),
        target_manifest=_source(
            target_manifest_path, repository_root, normalized_text=True
        ),
        retriever_manifest=_source(
            retriever_manifest_path, repository_root, normalized_text=True
        ),
        dense_runtime=dense_searcher.runtime_snapshot,
        bm25_runtime=bm25_runtime,
        trial_count=len(trials),
        queries_per_trial=registration.fixed_contract.queries_per_trial,
        total_search_calls=sum(trial.search_calls for trial in trials),
        unique_search_queries=len(unique_queries),
        baseline_evidence_recall_percent=baseline_evidence,
        baseline_gold_recall_percent=baseline_gold,
        selector_metrics=selector_metrics,
        bm25_top5_exact_reproduction=True,
        dense_top5_exact_reproduction=True,
        decision=(
            "select_fixed_top5_selector"
            if selected is not None
            else "reject_fixed_top5_selectors"
        ),
        selected_selector_id=(selected.selector_id if selected is not None else None),
        next_action=(
            registration.decision_rule.if_any_selector_passes
            if selected is not None
            else registration.decision_rule.if_all_selectors_fail
        ),
        rankings=rankings,
        trials=trials,
        claim_boundary=registration.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return result


def weighted_reciprocal_rank_fusion(
    bm25_docids: Sequence[str],
    dense_docids: Sequence[str],
    *,
    bm25_weight: float,
    dense_weight: float,
    k: int,
    depth: int,
) -> list[str]:
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    source_order: dict[str, int] = {}
    for source_index, (ranking, weight) in enumerate(
        ((bm25_docids, bm25_weight), (dense_docids, dense_weight))
    ):
        for rank, docid in enumerate(ranking, start=1):
            scores[docid] = scores.get(docid, 0.0) + weight / (k + rank)
            best_rank[docid] = min(best_rank.get(docid, rank), rank)
            source_order.setdefault(docid, source_index)
    ordered = sorted(
        scores,
        key=lambda docid: (
            -scores[docid],
            best_rank[docid],
            source_order[docid],
            docid,
        ),
    )
    return ordered[:depth]


def apply_selector(
    selector: SelectorSpec,
    *,
    bm25_docids: Sequence[str],
    dense_docids: Sequence[str],
    output_depth: int,
) -> list[str]:
    if selector.kind == "weighted_reciprocal_rank_fusion":
        assert selector.bm25_weight is not None
        assert selector.dense_weight is not None
        assert selector.rrf_k is not None
        return weighted_reciprocal_rank_fusion(
            bm25_docids,
            dense_docids,
            bm25_weight=selector.bm25_weight,
            dense_weight=selector.dense_weight,
            k=selector.rrf_k,
            depth=output_depth,
        )
    assert selector.dense_ranks is not None
    selected = [dense_docids[rank - 1] for rank in selector.dense_ranks]
    if len(selected) != output_depth:
        raise ValueError("dense portfolio does not fill the output depth")
    return selected


def _search_bm25(
    *, queries: Sequence[str], index_path: Path, depth: int, index_revision: str
) -> tuple[dict[str, list[str]], BM25RuntimeSnapshot]:
    started = time.perf_counter()
    try:
        from pyserini.search.lucene import LuceneSearcher
    except ImportError as error:
        raise RuntimeError(
            'install BM25 probe dependencies with pip install -e ".[browsecomp-plus]"'
        ) from error
    searcher = LuceneSearcher(str(index_path.resolve()))
    load_latency_ms = round((time.perf_counter() - started) * 1000)
    search_started = time.perf_counter()
    output = {
        query: [hit.docid for hit in searcher.search(query, depth)] for query in queries
    }
    if any(len(docids) < depth for docids in output.values()):
        raise ValueError("wide selector BM25 candidate depth is incomplete")
    return output, BM25RuntimeSnapshot(
        index_revision=index_revision,
        candidate_depth=depth,
        load_latency_ms=load_latency_ms,
        search_latency_ms=round((time.perf_counter() - search_started) * 1000),
    )


def _ranking(
    *,
    query: str,
    bm25: Sequence[str],
    dense: Sequence[RankedHit],
    selectors: Sequence[SelectorSpec],
    output_depth: int,
) -> SelectorRanking:
    dense_docids = [hit.docid for hit in dense]
    return SelectorRanking(
        query=query,
        query_sha256=sha256(query.encode("utf-8")).hexdigest(),
        bm25_docids=list(bm25),
        dense_docids=dense_docids,
        selected_docids={
            selector.selector_id: apply_selector(
                selector,
                bm25_docids=bm25,
                dense_docids=dense_docids,
                output_depth=output_depth,
            )
            for selector in selectors
        },
    )


def _score_trial(
    *,
    trial_id: str,
    summary_path: Path,
    gold_path: Path,
    replay_path: Path,
    ranking_by_query: Mapping[str, SelectorRanking],
    selectors: Sequence[SelectorSpec],
) -> WideSelectorTrial:
    summary = PiSmokeSummary.model_validate_json(summary_path.read_text(encoding="utf-8"))
    gold = DevelopmentGoldSlice.model_validate_json(gold_path.read_text(encoding="utf-8"))
    replay = RetrievalReplaySummary.model_validate_json(
        replay_path.read_text(encoding="utf-8")
    )
    if gold.source_summary_sha256 != _file_sha256(summary_path):
        raise ValueError("wide selector gold targets another summary")
    if replay.source_summary_sha256 != _file_sha256(summary_path):
        raise ValueError("wide selector replay targets another summary")
    gold_by_id = {row.query_id: row for row in gold.rows}
    replay_by_id = {row.query_id: row for row in replay.rows}
    rows: list[SelectorQueryRow] = []
    for item in summary.items:
        if item.run_sha256 is None:
            raise ValueError("wide selector source run is missing")
        run_path = summary_path.parent / _safe_id(item.query_id) / "run.json"
        if _file_sha256(run_path) != item.run_sha256:
            raise ValueError("wide selector source run hash changed")
        run = load_pi_browsecomp_run(run_path)
        reference = gold_by_id.get(item.query_id)
        replay_row = replay_by_id.get(item.query_id)
        if reference is None or replay_row is None:
            raise ValueError("wide selector query lacks gold or top-5 replay")
        if len(run.search_calls) != len(replay_row.calls):
            raise ValueError("wide selector source call count changed")
        baseline_docids: set[str] = set()
        selected = {selector.selector_id: set() for selector in selectors}
        for call, replay_call in zip(run.search_calls, replay_row.calls, strict=True):
            ranking = ranking_by_query.get(call.query)
            if ranking is None:
                raise ValueError("wide selector ranking is missing")
            stored_bm25 = [result.docid for result in call.results]
            if stored_bm25 != ranking.bm25_docids[:5]:
                raise ValueError("wide selector failed exact BM25 top-5 reproduction")
            if replay_call.candidate_docids != ranking.dense_docids[:5]:
                raise ValueError("wide selector failed exact dense top-5 reproduction")
            baseline_docids.update(stored_bm25)
            for selector in selectors:
                selected[selector.selector_id].update(
                    ranking.selected_docids[selector.selector_id]
                )
        evidence = set(reference.evidence_docids)
        gold_docids = set(reference.gold_docids)
        rows.append(
            SelectorQueryRow(
                trial_id=trial_id,
                query_id=item.query_id,
                search_calls=len(run.search_calls),
                baseline_evidence_recall=_recall(baseline_docids, evidence),
                baseline_gold_recall=_recall(baseline_docids, gold_docids),
                selectors=[
                    SelectorObservation(
                        selector_id=selector.selector_id,
                        retrieved_unique=len(selected[selector.selector_id]),
                        evidence_recall=_recall(
                            selected[selector.selector_id], evidence
                        ),
                        gold_recall=_recall(
                            selected[selector.selector_id], gold_docids
                        ),
                    )
                    for selector in selectors
                ],
            )
        )
    baseline_evidence = _percent(row.baseline_evidence_recall for row in rows)
    return WideSelectorTrial(
        trial_id=trial_id,
        query_count=len(rows),
        search_calls=sum(row.search_calls for row in rows),
        unique_search_queries=len(collect_frozen_search_queries(summary_path.parent)),
        baseline_evidence_recall_percent=baseline_evidence,
        baseline_gold_recall_percent=_percent(
            row.baseline_gold_recall for row in rows
        ),
        selector_metrics=[
            _trial_selector_metrics(
                rows=rows,
                selector_id=selector.selector_id,
                baseline_evidence=baseline_evidence,
            )
            for selector in selectors
        ],
        bm25_top5_exact_reproduction=True,
        dense_top5_exact_reproduction=True,
        rows=rows,
    )


def _trial_selector_metrics(
    *, rows: Sequence[SelectorQueryRow], selector_id: str, baseline_evidence: float
) -> SelectorTrialMetrics:
    observations = [_selector_observation(row, selector_id) for row in rows]
    evidence = _percent(row.evidence_recall for row in observations)
    wins, losses, ties = _paired_counts(rows, selector_id)
    return SelectorTrialMetrics(
        selector_id=selector_id,
        evidence_recall_percent=evidence,
        gold_recall_percent=_percent(row.gold_recall for row in observations),
        evidence_delta_vs_bm25_pp=round(evidence - baseline_evidence, 2),
        query_wins=wins,
        query_losses=losses,
        query_ties=ties,
        no_relevant_doc_queries=sum(
            row.evidence_recall == 0 and row.gold_recall == 0
            for row in observations
        ),
    )


def _aggregate_selector(
    *,
    selector_id: str,
    trials: Sequence[WideSelectorTrial],
    baseline_evidence: float,
    minimum_delta: float,
    minimum_wins_minus_losses: int,
) -> SelectorAggregate:
    rows = [_trial_selector_metric(trial, selector_id) for trial in trials]
    evidence = round(mean(row.evidence_recall_percent for row in rows), 6)
    wins = sum(row.query_wins for row in rows)
    losses = sum(row.query_losses for row in rows)
    ties = sum(row.query_ties for row in rows)
    delta = round(evidence - baseline_evidence, 6)
    wins_minus_losses = wins - losses
    return SelectorAggregate(
        selector_id=selector_id,
        evidence_recall_percent=evidence,
        gold_recall_percent=round(mean(row.gold_recall_percent for row in rows), 6),
        evidence_delta_vs_bm25_pp=delta,
        query_wins=wins,
        query_losses=losses,
        query_ties=ties,
        query_wins_minus_losses=wins_minus_losses,
        no_relevant_doc_queries=sum(row.no_relevant_doc_queries for row in rows),
        passed=(
            delta >= minimum_delta
            and wins_minus_losses >= minimum_wins_minus_losses
        ),
    )


def _selector_observation(
    row: SelectorQueryRow, selector_id: str
) -> SelectorObservation:
    matches = [item for item in row.selectors if item.selector_id == selector_id]
    if len(matches) != 1:
        raise ValueError(f"wide selector row lacks {selector_id}")
    return matches[0]


def _trial_selector_metric(
    trial: WideSelectorTrial, selector_id: str
) -> SelectorTrialMetrics:
    matches = [
        item for item in trial.selector_metrics if item.selector_id == selector_id
    ]
    if len(matches) != 1:
        raise ValueError(f"wide selector trial lacks {selector_id}")
    return matches[0]


def _paired_counts(
    rows: Sequence[SelectorQueryRow], selector_id: str
) -> tuple[int, int, int]:
    wins = losses = ties = 0
    for row in rows:
        candidate = _selector_observation(row, selector_id).evidence_recall
        if candidate > row.baseline_evidence_recall:
            wins += 1
        elif candidate < row.baseline_evidence_recall:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def _percent(values: Sequence[float] | object) -> float:
    materialized = list(values)  # type: ignore[arg-type]
    return round(mean(materialized) * 100, 2)


def _trial_mean(trials: Sequence[WideSelectorTrial], field: str) -> float:
    return round(mean(float(getattr(trial, field)) for trial in trials), 6)


def _recall(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


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
        raise ValueError("wide selector source path must be repository-relative")
    resolved = (repository_root / path).resolve()
    if not resolved.is_relative_to((repository_root / required_parent).resolve()):
        raise ValueError(f"wide selector source leaves {required_parent}/")
    if not resolved.is_file():
        raise ValueError(f"wide selector source is missing: {value}")
    return resolved


def _source(
    path: Path, repository_root: Path, *, normalized_text: bool = False
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


def _require_hash(path: Path, expected_hash: str, label: str) -> None:
    if _file_sha256(path) != expected_hash:
        raise ValueError(f"wide selector source hash changed: {label}")


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require_under_runs(path: Path, repository_root: Path) -> None:
    if not path.resolve().is_relative_to((repository_root / "runs").resolve()):
        raise ValueError("wide selector artifacts must stay under ignored runs/")


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
