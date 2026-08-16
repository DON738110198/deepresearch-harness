from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceTokenizer(Protocol):
    def encode(self, text: str) -> Sequence[object]: ...

    def decode(self, tokens: Sequence[object]) -> str: ...


class StrictContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProgressiveDisclosurePolicy(StrictContract):
    anchor_count: int = Field(default=5, ge=1, le=10)
    dense_lead_count: int = Field(default=15, ge=1, le=20)
    anchor_token_cap: int = Field(default=512, ge=16, le=1024)
    lead_token_cap: int = Field(default=24, ge=4, le=128)
    open_token_cap: int = Field(default=512, ge=16, le=2048)
    maximum_open_calls: int = Field(default=8, ge=1, le=32)
    total_evidence_ingress_token_budget: int = Field(ge=512, le=100_000)
    open_evidence_ingress_token_budget: int = Field(default=0, ge=0, le=65_536)
    anchor_open_policy: Literal["assume_full", "reopen_with_obligation"] = (
        "assume_full"
    )
    open_content_policy: Literal["head_v0", "answer_obligation_window_v0"] = (
        "head_v0"
    )

    @model_validator(mode="after")
    def open_budget_is_funded_and_usable(self) -> "ProgressiveDisclosurePolicy":
        if (
            self.open_evidence_ingress_token_budget
            > self.total_evidence_ingress_token_budget
        ):
            raise ValueError("open evidence budget exceeds the total ingress budget")
        if self.open_evidence_ingress_token_budget > (
            self.maximum_open_calls * self.open_token_cap
        ):
            raise ValueError("open evidence budget exceeds the maximum usable amount")
        if (
            self.anchor_open_policy == "reopen_with_obligation"
            and self.open_content_policy != "answer_obligation_window_v0"
        ):
            raise ValueError(
                "reopening anchor previews requires obligation-window opening"
            )
        return self


class EvidenceCandidate(StrictContract):
    docid: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    text: str = Field(min_length=1)


class DisclosedEvidence(StrictContract):
    docid: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    channel: Literal["bm25_anchor", "dense_lead"]
    content: str = Field(min_length=1)
    ingress_tokens: int = Field(gt=0)


class DisclosureSearchResult(StrictContract):
    search_index: int = Field(ge=1)
    results: list[DisclosedEvidence]
    anchors_returned: int = Field(ge=0)
    leads_returned: int = Field(ge=0)
    within_channel_duplicate_slots: int = Field(ge=0)
    prior_context_duplicate_slots: int = Field(ge=0)
    cross_channel_duplicate_slots: int = Field(ge=0)
    new_ingress_tokens: int = Field(ge=0)
    cumulative_ingress_tokens: int = Field(ge=0)
    remaining_ingress_tokens: int = Field(ge=0)
    remaining_search_ingress_tokens: int = Field(ge=0)
    remaining_open_ingress_tokens: int = Field(ge=0)
    ingress_budget_exhausted: bool

    @model_validator(mode="after")
    def counts_match_results(self) -> "DisclosureSearchResult":
        if self.anchors_returned + self.leads_returned != len(self.results):
            raise ValueError("disclosure channel counts do not match results")
        if self.new_ingress_tokens != sum(row.ingress_tokens for row in self.results):
            raise ValueError("disclosure token count does not match results")
        return self


class OpenEvidenceResult(StrictContract):
    docid: str = Field(min_length=1)
    outcome: Literal[
        "opened",
        "already_opened",
        "not_disclosed",
        "open_limit_reached",
        "ingress_budget_exhausted",
        "missing_document",
    ]
    content: str | None = None
    ingress_tokens: int = Field(ge=0)
    cumulative_ingress_tokens: int = Field(ge=0)
    remaining_ingress_tokens: int = Field(ge=0)
    remaining_search_ingress_tokens: int = Field(ge=0)
    remaining_open_ingress_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def content_matches_outcome(self) -> "OpenEvidenceResult":
        if self.outcome == "opened":
            if not self.content or self.ingress_tokens <= 0:
                raise ValueError("opened evidence requires content and tokens")
        elif self.content is not None or self.ingress_tokens != 0:
            raise ValueError("non-open outcome cannot ingest evidence")
        return self


class DisclosureStateSnapshot(StrictContract):
    run_id: str = Field(min_length=1)
    search_calls: int = Field(ge=0)
    open_attempts: int = Field(ge=0)
    successful_open_calls: int = Field(ge=0)
    seen_docids: tuple[str, ...]
    eligible_open_docids: tuple[str, ...]
    opened_docids: tuple[str, ...]
    cumulative_ingress_tokens: int = Field(ge=0)
    search_ingress_tokens: int = Field(ge=0)
    open_ingress_tokens: int = Field(ge=0)
    remaining_ingress_tokens: int = Field(ge=0)
    remaining_search_ingress_tokens: int = Field(ge=0)
    remaining_open_ingress_tokens: int = Field(ge=0)


class ProgressiveDisclosureSession:
    def __init__(
        self,
        *,
        run_id: str,
        policy: ProgressiveDisclosurePolicy,
        tokenizer: EvidenceTokenizer,
        document_loader: Callable[[str], str | None],
        obligation_document_loader: Callable[[str, str], str | None] | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be blank")
        self._run_id = run_id
        self._policy = policy
        self._tokenizer = tokenizer
        self._document_loader = document_loader
        self._obligation_document_loader = obligation_document_loader
        if (
            policy.open_content_policy == "answer_obligation_window_v0"
            and obligation_document_loader is None
        ):
            raise ValueError(
                "obligation-window opening requires an obligation document loader"
            )
        self._seen_docids: set[str] = set()
        self._eligible_open_docids: set[str] = set()
        self._opened_docids: set[str] = set()
        self._search_calls = 0
        self._open_attempts = 0
        self._successful_open_calls = 0
        self._ingress_tokens = 0
        self._search_ingress_tokens = 0
        self._open_ingress_tokens = 0

    def search(
        self,
        *,
        bm25_candidates: Sequence[EvidenceCandidate],
        dense_candidates: Sequence[EvidenceCandidate],
    ) -> DisclosureSearchResult:
        self._search_calls += 1
        bm25_unique, bm25_duplicates = _deduplicate_candidates(bm25_candidates)
        dense_unique, dense_duplicates = _deduplicate_candidates(dense_candidates)
        prior_duplicates = sum(
            row.docid in self._seen_docids for row in (*bm25_unique, *dense_unique)
        )

        anchors = [
            row for row in bm25_unique if row.docid not in self._seen_docids
        ][: self._policy.anchor_count]
        anchor_docids = {row.docid for row in anchors}
        dense_after_prior = [
            row for row in dense_unique if row.docid not in self._seen_docids
        ]
        cross_channel_duplicates = sum(
            row.docid in anchor_docids for row in dense_after_prior
        )
        leads = [
            row for row in dense_after_prior if row.docid not in anchor_docids
        ][: self._policy.dense_lead_count]

        disclosed: list[DisclosedEvidence] = []
        for candidate in anchors:
            item = self._disclose_candidate(
                candidate, channel="bm25_anchor", token_cap=self._policy.anchor_token_cap
            )
            if item is None:
                break
            disclosed.append(item)
            self._seen_docids.add(candidate.docid)
            if self._policy.anchor_open_policy == "assume_full":
                self._opened_docids.add(candidate.docid)
            else:
                self._eligible_open_docids.add(candidate.docid)
        for candidate in leads:
            item = self._disclose_candidate(
                candidate, channel="dense_lead", token_cap=self._policy.lead_token_cap
            )
            if item is None:
                break
            disclosed.append(item)
            self._seen_docids.add(candidate.docid)
            self._eligible_open_docids.add(candidate.docid)

        new_tokens = sum(row.ingress_tokens for row in disclosed)
        return DisclosureSearchResult(
            search_index=self._search_calls,
            results=disclosed,
            anchors_returned=sum(row.channel == "bm25_anchor" for row in disclosed),
            leads_returned=sum(row.channel == "dense_lead" for row in disclosed),
            within_channel_duplicate_slots=bm25_duplicates + dense_duplicates,
            prior_context_duplicate_slots=prior_duplicates,
            cross_channel_duplicate_slots=cross_channel_duplicates,
            new_ingress_tokens=new_tokens,
            cumulative_ingress_tokens=self._ingress_tokens,
            remaining_ingress_tokens=self.remaining_ingress_tokens,
            remaining_search_ingress_tokens=self.remaining_search_ingress_tokens,
            remaining_open_ingress_tokens=self.remaining_open_ingress_tokens,
            ingress_budget_exhausted=self.remaining_search_ingress_tokens == 0,
        )

    def open_evidence(
        self, docid: str, *, obligation_query: str | None = None
    ) -> OpenEvidenceResult:
        if not docid.strip():
            raise ValueError("docid must not be blank")
        if self._policy.open_content_policy == "answer_obligation_window_v0":
            if obligation_query is None or not obligation_query.strip():
                raise ValueError(
                    "obligation_query is required for obligation-window opening"
                )
        self._open_attempts += 1
        if docid in self._opened_docids:
            return self._open_result(docid, "already_opened")
        if docid not in self._eligible_open_docids:
            return self._open_result(docid, "not_disclosed")
        if self._successful_open_calls >= self._policy.maximum_open_calls:
            return self._open_result(docid, "open_limit_reached")
        if self.remaining_open_ingress_tokens == 0:
            return self._open_result(docid, "ingress_budget_exhausted")
        text = (
            self._obligation_document_loader(docid, obligation_query.strip())
            if self._policy.open_content_policy == "answer_obligation_window_v0"
            and self._obligation_document_loader is not None
            and obligation_query is not None
            else self._document_loader(docid)
        )
        if text is None or not text.strip():
            return self._open_result(docid, "missing_document")
        content, token_count = self._truncate(
            text,
            min(self._policy.open_token_cap, self.remaining_open_ingress_tokens),
        )
        if token_count == 0:
            return self._open_result(docid, "ingress_budget_exhausted")
        self._ingress_tokens += token_count
        self._open_ingress_tokens += token_count
        self._successful_open_calls += 1
        self._opened_docids.add(docid)
        self._eligible_open_docids.remove(docid)
        return OpenEvidenceResult(
            docid=docid,
            outcome="opened",
            content=content,
            ingress_tokens=token_count,
            cumulative_ingress_tokens=self._ingress_tokens,
            remaining_ingress_tokens=self.remaining_ingress_tokens,
            remaining_search_ingress_tokens=self.remaining_search_ingress_tokens,
            remaining_open_ingress_tokens=self.remaining_open_ingress_tokens,
        )

    @property
    def remaining_ingress_tokens(self) -> int:
        return max(
            self._policy.total_evidence_ingress_token_budget - self._ingress_tokens,
            0,
        )

    @property
    def remaining_open_ingress_tokens(self) -> int:
        return min(
            max(
                self._policy.open_evidence_ingress_token_budget
                - self._open_ingress_tokens,
                0,
            ),
            self.remaining_ingress_tokens,
        )

    @property
    def remaining_search_ingress_tokens(self) -> int:
        return max(
            self.remaining_ingress_tokens - self.remaining_open_ingress_tokens,
            0,
        )

    def snapshot(self) -> DisclosureStateSnapshot:
        return DisclosureStateSnapshot(
            run_id=self._run_id,
            search_calls=self._search_calls,
            open_attempts=self._open_attempts,
            successful_open_calls=self._successful_open_calls,
            seen_docids=tuple(sorted(self._seen_docids)),
            eligible_open_docids=tuple(sorted(self._eligible_open_docids)),
            opened_docids=tuple(sorted(self._opened_docids)),
            cumulative_ingress_tokens=self._ingress_tokens,
            search_ingress_tokens=self._search_ingress_tokens,
            open_ingress_tokens=self._open_ingress_tokens,
            remaining_ingress_tokens=self.remaining_ingress_tokens,
            remaining_search_ingress_tokens=self.remaining_search_ingress_tokens,
            remaining_open_ingress_tokens=self.remaining_open_ingress_tokens,
        )

    def _disclose_candidate(
        self,
        candidate: EvidenceCandidate,
        *,
        channel: Literal["bm25_anchor", "dense_lead"],
        token_cap: int,
    ) -> DisclosedEvidence | None:
        available = min(token_cap, self.remaining_search_ingress_tokens)
        if available <= 0:
            return None
        content, token_count = self._truncate(candidate.text, available)
        if token_count == 0:
            return None
        self._ingress_tokens += token_count
        self._search_ingress_tokens += token_count
        return DisclosedEvidence(
            docid=candidate.docid,
            score=candidate.score,
            channel=channel,
            content=content,
            ingress_tokens=token_count,
        )

    def _truncate(self, text: str, token_cap: int) -> tuple[str, int]:
        tokens = list(self._tokenizer.encode(text))
        selected = tokens[:token_cap]
        if not selected:
            return "", 0
        return self._tokenizer.decode(selected), len(selected)

    def _open_result(
        self,
        docid: str,
        outcome: Literal[
            "already_opened",
            "not_disclosed",
            "open_limit_reached",
            "ingress_budget_exhausted",
            "missing_document",
        ],
    ) -> OpenEvidenceResult:
        return OpenEvidenceResult(
            docid=docid,
            outcome=outcome,
            ingress_tokens=0,
            cumulative_ingress_tokens=self._ingress_tokens,
            remaining_ingress_tokens=self.remaining_ingress_tokens,
            remaining_search_ingress_tokens=self.remaining_search_ingress_tokens,
            remaining_open_ingress_tokens=self.remaining_open_ingress_tokens,
        )


def _deduplicate_candidates(
    candidates: Sequence[EvidenceCandidate],
) -> tuple[list[EvidenceCandidate], int]:
    seen: set[str] = set()
    unique: list[EvidenceCandidate] = []
    duplicate_slots = 0
    for candidate in candidates:
        if candidate.docid in seen:
            duplicate_slots += 1
            continue
        seen.add(candidate.docid)
        unique.append(candidate)
    return unique, duplicate_slots


def format_bm25_anchor(contents: str) -> str:
    return f"[BM25 anchor: full evidence]\n{contents}"


def format_bm25_anchor_preview(docid: str, contents: str) -> str:
    return (
        "[BM25 anchor preview; content is truncated. Use open_evidence with "
        f"docid={docid} and an answer-critical obligation before relying on it.]\n"
        f"{contents}"
    )


def format_dense_lead(docid: str, contents: str) -> str:
    return (
        "[Dense lead: preview only. Use open_evidence with this "
        f"docid before relying on it.]\n{contents}"
    )


def format_opened_evidence(docid: str, contents: str) -> str:
    return f"[Opened dense evidence: docid={docid}]\n{contents}"


def format_opened_obligation_span(
    docid: str, contents: str, *, start_character: int, end_character: int
) -> str:
    return (
        f"[Opened obligation span: docid={docid}; "
        f"characters={start_character}:{end_character}]\n{contents}"
    )
