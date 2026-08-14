from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from statistics import mean

from deepresearch_harness.browsecomp_evaluation import DevelopmentGoldSlice
from deepresearch_harness.browsecomp_repeats import RepeatExperimentManifest
from deepresearch_harness.retrieval_replay import RetrievalReplaySummary


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate BM25, dense, RRF, and doubled-pool retrieval diagnostics "
            "from frozen replay artifacts without making provider calls."
        )
    )
    parser.add_argument("--repeat-experiment", type=Path, required=True)
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.replay) != 3:
        parser.error("exactly three --replay artifacts are required")
    if args.output.exists():
        parser.error("--output must not already exist")

    experiment_path = args.repeat_experiment.resolve()
    repository_root = find_repository_root(experiment_path)
    experiment = RepeatExperimentManifest.model_validate_json(
        experiment_path.read_text(encoding="utf-8")
    )
    replay_by_source = {}
    replay_paths = {}
    for path_arg in args.replay:
        path = path_arg.resolve()
        replay = RetrievalReplaySummary.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if replay.source_summary_sha256 in replay_by_source:
            raise ValueError("retrieval pool probe received duplicate source summaries")
        replay_by_source[replay.source_summary_sha256] = replay
        replay_paths[replay.source_summary_sha256] = path

    trials = []
    query_values: dict[str, dict[str, list[float]]] = {}
    for pair in experiment.pairs:
        summary_path = resolve_repository_file(
            pair.baseline.summary_path, repository_root
        )
        gold_path = resolve_repository_file(
            pair.baseline.gold_slice_path, repository_root
        )
        summary_hash = sha256_file(summary_path)
        replay = replay_by_source.get(summary_hash)
        if replay is None:
            raise ValueError(f"missing replay for {pair.trial_id}")
        if replay.gold_slice_sha256 != sha256_file(gold_path):
            raise ValueError("retrieval replay changes the frozen gold slice")
        gold = DevelopmentGoldSlice.model_validate_json(
            gold_path.read_text(encoding="utf-8")
        )
        gold_by_id = {row.query_id: row for row in gold.rows}
        union_evidence_values = []
        union_gold_values = []
        union_zero_recall = 0
        for row in replay.rows:
            reference = gold_by_id.get(row.query_id)
            if reference is None:
                raise ValueError("retrieval replay query is missing frozen gold")
            union_docids = {
                docid
                for call in row.calls
                for docid in [*call.baseline_docids, *call.candidate_docids]
            }
            evidence_recall = recall(union_docids, set(reference.evidence_docids))
            gold_recall = recall(union_docids, set(reference.gold_docids))
            union_evidence_values.append(evidence_recall)
            union_gold_values.append(gold_recall)
            if evidence_recall == 0 and gold_recall == 0:
                union_zero_recall += 1
            values = query_values.setdefault(
                row.query_id,
                {"bm25": [], "dense": [], "rrf": [], "union_pool": []},
            )
            values["bm25"].append(row.baseline.evidence_recall or 0.0)
            values["dense"].append(row.candidate.evidence_recall or 0.0)
            values["rrf"].append(row.fused.evidence_recall or 0.0)
            values["union_pool"].append(evidence_recall)
        replay_path = replay_paths[summary_hash]
        trials.append(
            {
                "trial_id": pair.trial_id,
                "execution_order": pair.execution_order,
                "replay_path": replay_path.relative_to(repository_root).as_posix(),
                "replay_sha256": sha256_file(replay_path),
                "bm25_evidence_recall_percent": replay.baseline_evidence_recall_percent,
                "dense_evidence_recall_percent": replay.candidate_evidence_recall_percent,
                "rrf_evidence_recall_percent": replay.fused_evidence_recall_percent,
                "union_pool_evidence_recall_percent": percent(
                    union_evidence_values
                ),
                "union_pool_gold_recall_percent": percent(union_gold_values),
                "union_pool_zero_recall_queries": union_zero_recall,
            }
        )

    query_rows = []
    for query_id, values in sorted(query_values.items()):
        averages = {key: mean(rows) * 100 for key, rows in values.items()}
        query_rows.append(
            {
                "query_id": query_id,
                **{f"{key}_evidence_recall_percent": round(value, 6) for key, value in averages.items()},
                "dense_minus_bm25_pp": round(
                    averages["dense"] - averages["bm25"], 6
                ),
                "rrf_minus_bm25_pp": round(
                    averages["rrf"] - averages["bm25"], 6
                ),
                "union_pool_minus_bm25_pp": round(
                    averages["union_pool"] - averages["bm25"], 6
                ),
            }
        )
    aggregate = {}
    for field_name in (
        "bm25_evidence_recall_percent",
        "dense_evidence_recall_percent",
        "rrf_evidence_recall_percent",
        "union_pool_evidence_recall_percent",
        "union_pool_gold_recall_percent",
    ):
        aggregate[f"{field_name}_mean"] = round(
            mean(float(row[field_name]) for row in trials), 6
        )
    aggregate["dense_minus_bm25_pp"] = round(
        aggregate["dense_evidence_recall_percent_mean"]
        - aggregate["bm25_evidence_recall_percent_mean"],
        6,
    )
    aggregate["rrf_minus_bm25_pp"] = round(
        aggregate["rrf_evidence_recall_percent_mean"]
        - aggregate["bm25_evidence_recall_percent_mean"],
        6,
    )
    aggregate["union_pool_minus_bm25_pp"] = round(
        aggregate["union_pool_evidence_recall_percent_mean"]
        - aggregate["bm25_evidence_recall_percent_mean"],
        6,
    )
    payload = {
        "schema_version": "browsecomp-plus-retrieval-pool-probe-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_not_official",
        "provider_calls": 0,
        "repeat_experiment_sha256": sha256_file(experiment_path),
        "trial_count": len(trials),
        "queries_per_trial": len(query_rows),
        "fixed_per_call_depth": 5,
        "union_pool_max_per_call_depth": 10,
        "aggregate": aggregate,
        "trials": trials,
        "queries": query_rows,
        "claim_boundary": (
            "The union pool doubles the per-call candidate pool and is an offline "
            "upper-bound diagnostic, not a deployable fixed-top-5 result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"output={args.output}")
    print("provider_calls=0")
    print(json.dumps(aggregate, sort_keys=True))
    return 0


def recall(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def percent(values: list[float]) -> float:
    return round(mean(values) * 100, 6)


def resolve_repository_file(value: str, repository_root: Path) -> Path:
    path = (repository_root / value).resolve()
    if not path.is_relative_to((repository_root / "runs").resolve()) or not path.is_file():
        raise ValueError("retrieval pool probe source is invalid")
    return path


def find_repository_root(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "runs"
        ).is_dir():
            return candidate
    raise ValueError("could not locate repository root")


if __name__ == "__main__":
    raise SystemExit(main())
