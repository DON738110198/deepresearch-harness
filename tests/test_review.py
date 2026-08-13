import json
from pathlib import Path

import pytest

from deepresearch_harness.batch import run_experiment_batch
from deepresearch_harness.contracts import HarnessConfig
from deepresearch_harness.providers import FakeProvider
from deepresearch_harness.review import (
    BlindReviewPacket,
    BlindReviewSubmission,
    CandidateAnnotation,
    ReviewerType,
    prepare_blind_review,
    score_blind_review,
    validate_blind_submission,
)
from deepresearch_harness.review_workspace import render_review_workspace, validate_review_submission_file


ROOT = Path(__file__).resolve().parents[1]


def test_review_packet_hides_variants_and_writes_separate_key(tmp_path: Path) -> None:
    config = HarnessConfig.model_validate({"provider": {"kind": "fake"}, "run": {"max_evidence": 6}})
    summary = run_experiment_batch(
        manifest_path=ROOT / "experiments" / "pilot_v0" / "token_matched.json",
        config=config,
        output_root=tmp_path / "batch",
        provider=FakeProvider(),
        enforce_provider_match=False,
    )
    summary_path = Path(summary.output_dir) / "summary.json"

    packet_path, template_path, key_path = prepare_blind_review(
        summary_path=summary_path,
        suite_path=ROOT / "benchmarks" / "pilot_v0" / "tasks.json",
        output_dir=tmp_path / "review",
    )

    packet_text = packet_path.read_text(encoding="utf-8")
    assert "b0_search_write" not in packet_text
    assert "b1_plan_search_ledger_write" not in packet_text
    assert '"query"' not in packet_text
    assert '"candidate_id": "A"' in packet_text
    template = BlindReviewSubmission.model_validate_json(template_path.read_text(encoding="utf-8"))
    assert len(template.annotations) == 20
    key_text = key_path.read_text(encoding="utf-8")
    assert "b0_search_write" in key_text
    assert "b1_plan_search_ledger_write" in key_text

    packet = BlindReviewPacket.model_validate_json(packet_text)
    annotations = []
    for task in packet.tasks:
        for candidate in task.candidates:
            claim_ids = [claim.id for claim in candidate.claims]
            annotations.append(
                CandidateAnnotation(
                    task_id=task.task_id,
                    candidate_id=candidate.candidate_id,
                    covered_obligation_ids=[item.id for item in task.obligations],
                    supported_claim_ids=claim_ids,
                    citation_supported_claim_ids=sorted({item.claim_id for item in candidate.citations}),
                    conflict_handled_obligation_ids=[
                        item.id for item in task.obligations if item.counter_evidence_ids
                    ],
                )
            )
    submission = BlindReviewSubmission(
        experiment_id=packet.experiment_id,
        reviewer_type=ReviewerType.AI_ASSISTED,
        annotations=annotations,
    )
    annotations_path = tmp_path / "review" / "annotations.json"
    annotations_path.write_text(submission.model_dump_json(indent=2), encoding="utf-8")
    result = score_blind_review(
        packet_path=packet_path,
        annotations_path=annotations_path,
        answer_key_path=key_path,
        output_path=tmp_path / "review" / "scores.json",
    )
    assert result.result_status == "calibration_only"
    assert set(result.aggregates) == {"b0_search_write", "b1_plan_search_ledger_write"}
    assert all(item.score.annotation_status == "ai_assisted_calibration" for item in result.candidates)

    workspace_path = render_review_workspace(
        packet_path=packet_path,
        output_path=tmp_path / "review" / "review_workspace.html",
    )
    workspace = workspace_path.read_text(encoding="utf-8")
    assert "Blind Semantic Review" in workspace
    assert "b0_search_write" not in workspace
    assert "b1_plan_search_ledger_write" not in workspace
    assert "answer_key" not in workspace
    assert 'reviewer_type: "human"' in workspace
    assert "Open source" in workspace
    validated_count, annotation_hash, reviewer_type = validate_review_submission_file(
        packet_path=packet_path,
        annotations_path=annotations_path,
    )
    assert validated_count == 20
    assert len(annotation_hash) == 64
    assert reviewer_type == "ai_assisted"

    incomplete = submission.model_copy(deep=True)
    incomplete.annotations[0].supported_claim_ids.pop()
    with pytest.raises(ValueError, match="classify every claim"):
        validate_blind_submission(packet, incomplete)
