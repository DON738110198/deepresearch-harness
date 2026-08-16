from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .counter_hypothesis_packet import BridgePacket, CounterProbeResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(StrictContract):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CandidateReplaySpec(StrictContract):
    schema_version: Literal["counter-candidate-replay-spec-v0"] = (
        "counter-candidate-replay-spec-v0"
    )
    status: Literal["registered_before_local_replay"]
    registered_at: str = Field(min_length=1)
    source_result: ArtifactReference
    gold_slice: ArtifactReference
    query_ids: tuple[str, ...] = Field(min_length=1)
    search_url: str = Field(pattern=r"^http://127\.0\.0\.1:\d+/search$")
    retriever_id: str = Field(min_length=1)
    candidates_per_case: Literal[3] = 3
    max_search_results: int = Field(ge=1, le=100)
    query_transformation: Literal[
        "replace_selected_bridge_in_preserved_query_v0"
    ] = "replace_selected_bridge_in_preserved_query_v0"
    frozen_artifacts: tuple[ArtifactReference, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "CandidateReplaySpec":
        if len(self.query_ids) != len(set(self.query_ids)):
            raise ValueError("candidate replay IDs must be unique")
        return self


class CandidateReplayItem(StrictContract):
    query_id: str
    candidate_index: int = Field(ge=0, le=2)
    was_selected: bool
    bridge_type: str
    bridge_term: str
    query: str
    returned_docids: tuple[str, ...]
    gold_hits: tuple[str, ...]
    gold_hit_ranks: tuple[int, ...]
    search_latency_ms: int = Field(ge=0)


class CandidateReplayResult(StrictContract):
    schema_version: Literal["counter-candidate-replay-v0"] = (
        "counter-candidate-replay-v0"
    )
    created_at: str
    status: Literal["post_generation_diagnostic_not_effectiveness"] = (
        "post_generation_diagnostic_not_effectiveness"
    )
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_count: int = Field(ge=0)
    search_calls: int = Field(ge=0)
    source_selected_gold_hit_cases: int = Field(ge=0)
    replay_selected_gold_hit_cases: int = Field(ge=0)
    any_candidate_gold_hit_cases: int = Field(ge=0)
    unselected_rescue_cases: int = Field(ge=0)
    diagnosis: Literal[
        "selection_bottleneck_present",
        "candidate_generation_bottleneck",
    ]
    items: tuple[CandidateReplayItem, ...]
    claim_boundary: str


def _validate_artifacts(root: Path, artifacts: tuple[ArtifactReference, ...]) -> None:
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"registered artifact is missing or escapes root: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            raise ValueError(f"registered artifact hash changed: {artifact.path}")


def load_candidate_replay_spec(path: Path) -> CandidateReplaySpec:
    spec = CandidateReplaySpec.model_validate_json(path.read_text(encoding="utf-8"))
    root = path.resolve().parents[2]
    _validate_artifacts(
        root,
        (spec.source_result, spec.gold_slice, *spec.frozen_artifacts),
    )
    return spec


def build_replay_query(packet: BridgePacket, candidate_index: int) -> str:
    if candidate_index == packet.selected:
        return packet.query
    selected_tokens = set(_tokens(packet.bridges[packet.selected].term))
    remaining = [token for token in _tokens(packet.query) if token not in selected_tokens]
    combined = [*_tokens(packet.bridges[candidate_index].term), *remaining]
    deduplicated = tuple(dict.fromkeys(combined))
    if not 3 <= len(deduplicated) <= 18:
        raise ValueError("replayed candidate query must contain 3 to 18 terms")
    return " ".join(deduplicated)


def _packet_from_item(item: object) -> BridgePacket:
    packet = getattr(item, "packet")
    if packet is not None:
        return packet
    raw_text = getattr(item, "raw_completion_text")
    if not isinstance(raw_text, str):
        raise ValueError(f"source item has no replayable BridgePacket: {getattr(item, 'query_id')}")
    return BridgePacket.model_validate_json(raw_text)


def _search(
    search_url: str, run_id: str, query: str, timeout_seconds: int
) -> tuple[tuple[str, ...], int]:
    request = Request(
        search_url,
        data=json.dumps({"run_id": run_id, "query": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as error:
        raise RuntimeError(f"candidate replay retrieval failed: {error}") from error
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("candidate replay response has no results array")
    return (
        tuple(str(item["docid"]) for item in results),
        round((time.perf_counter() - started) * 1000),
    )


def run_candidate_replay(
    *, spec_path: Path, output_path: Path, timeout_seconds: int = 60
) -> CandidateReplayResult:
    spec = load_candidate_replay_spec(spec_path)
    root = spec_path.resolve().parents[2]
    if not output_path.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("candidate replay output must stay under ignored runs/")
    if output_path.exists():
        raise ValueError("candidate replay output already exists")
    source = CounterProbeResult.model_validate_json(
        (root / spec.source_result.path).read_text(encoding="utf-8")
    )
    source_by_id = {item.query_id: item for item in source.items}
    if tuple(source_by_id) != spec.query_ids:
        raise ValueError("candidate replay source IDs differ from registration")
    gold = json.loads((root / spec.gold_slice.path).read_text(encoding="utf-8"))
    gold_by_id = {
        str(row["query_id"]): tuple(str(item) for item in row["gold_docids"])
        for row in gold["rows"]
    }
    items: list[CandidateReplayItem] = []
    source_selected_cases = sum(
        bool(source_by_id[query_id].candidate_gold_hits) for query_id in spec.query_ids
    )
    for query_id in spec.query_ids:
        source_item = source_by_id[query_id]
        packet = _packet_from_item(source_item)
        if len(packet.bridges) != spec.candidates_per_case:
            raise ValueError("source BridgePacket changed candidate count")
        gold_docids = gold_by_id[query_id]
        for index, bridge in enumerate(packet.bridges):
            query = build_replay_query(packet, index)
            docids, latency = _search(
                spec.search_url,
                f"counter-candidate-replay-{query_id}-{index}",
                query,
                timeout_seconds,
            )
            if len(docids) > spec.max_search_results:
                raise ValueError("candidate replay retriever exceeded the result cap")
            hits = tuple(item for item in docids if item in gold_docids)
            ranks = tuple(
                rank + 1 for rank, item in enumerate(docids) if item in gold_docids
            )
            items.append(
                CandidateReplayItem(
                    query_id=query_id,
                    candidate_index=index,
                    was_selected=index == packet.selected,
                    bridge_type=bridge.type,
                    bridge_term=bridge.term,
                    query=query,
                    returned_docids=docids,
                    gold_hits=hits,
                    gold_hit_ranks=ranks,
                    search_latency_ms=latency,
                )
            )
    selected_hits = {
        item.query_id for item in items if item.was_selected and item.gold_hits
    }
    any_hits = {item.query_id for item in items if item.gold_hits}
    unselected_rescues = {
        item.query_id for item in items if not item.was_selected and item.gold_hits
    }
    result = CandidateReplayResult(
        created_at=_utc_now(),
        spec_sha256=_sha256_file(spec_path),
        query_count=len(spec.query_ids),
        search_calls=len(items),
        source_selected_gold_hit_cases=source_selected_cases,
        replay_selected_gold_hit_cases=len(selected_hits),
        any_candidate_gold_hit_cases=len(any_hits),
        unselected_rescue_cases=len(unselected_rescues),
        diagnosis=(
            "selection_bottleneck_present"
            if unselected_rescues
            else "candidate_generation_bottleneck"
        ),
        items=tuple(items),
        claim_boundary=spec.claim_boundary,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
