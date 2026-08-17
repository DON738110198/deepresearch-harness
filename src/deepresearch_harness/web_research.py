from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import socket
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from typing import Callable, Protocol
from urllib.parse import ParseResult, parse_qs, quote, unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import BaseModel, Field, HttpUrl

from .contracts import Evidence, SearchConfig, TraceEvent


class SearchHit(BaseModel):
    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = ""


@dataclass(frozen=True)
class HttpResponse:
    url: str
    content_type: str
    body: bytes


class HttpClient(Protocol):
    def get(self, url: str, *, accept: str) -> HttpResponse: ...


class JsonPostHttpClient(HttpClient, Protocol):
    def post_json(
        self,
        url: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> HttpResponse: ...


class SearchBudgetExhausted(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchAttempt:
    index: int
    logical_query_index: int
    query_sha256: str
    provider: str
    outcome: str
    result_count: int | None
    latency_ms: int
    estimated_cost_usd: float
    error_type: str | None = None


class SearchCallBudget:
    """Counts every networked search attempt before it is issued."""

    def __init__(self, max_attempts: int) -> None:
        self._max_attempts = max_attempts
        self._active_query = ""
        self._active_query_index = 0
        self.attempts: list[SearchAttempt] = []

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def reset(self) -> None:
        self.attempts = []
        self._active_query = ""
        self._active_query_index = 0

    def begin_logical_query(self, query: str, index: int) -> None:
        self._active_query = query
        self._active_query_index = index

    def search(
        self,
        provider: str,
        request: Callable[[], list[SearchHit]],
        *,
        estimated_cost_usd: float = 0.0,
    ) -> list[SearchHit]:
        if len(self.attempts) >= self._max_attempts:
            raise SearchBudgetExhausted(
                f"search attempt budget exhausted at {self._max_attempts} attempts"
            )
        started = time.perf_counter()
        index = len(self.attempts) + 1
        try:
            hits = request()
        except Exception as error:
            self.attempts.append(
                SearchAttempt(
                    index=index,
                    logical_query_index=self._active_query_index,
                    query_sha256=sha256(self._active_query.encode("utf-8")).hexdigest(),
                    provider=provider,
                    outcome="error",
                    result_count=None,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    estimated_cost_usd=estimated_cost_usd,
                    error_type=type(error).__name__,
                )
            )
            raise
        self.attempts.append(
            SearchAttempt(
                index=index,
                logical_query_index=self._active_query_index,
                query_sha256=sha256(self._active_query.encode("utf-8")).hexdigest(),
                provider=provider,
                outcome="ok",
                result_count=len(hits),
                latency_ms=int((time.perf_counter() - started) * 1000),
                estimated_cost_usd=estimated_cost_usd,
            )
        )
        return hits


class SafeHttpClient:
    def __init__(self, *, timeout_seconds: int, max_download_bytes: int, user_agent: str) -> None:
        self._timeout_seconds = timeout_seconds
        self._max_download_bytes = max_download_bytes
        self._user_agent = user_agent
        self._opener = build_opener(_SafeRedirectHandler())

    def get(self, url: str, *, accept: str) -> HttpResponse:
        _require_public_http_url(url)
        request = Request(
            url,
            headers={"Accept": accept, "User-Agent": self._user_agent},
            method="GET",
        )
        with self._opener.open(request, timeout=self._timeout_seconds) as response:
            final_url = response.geturl()
            _require_public_http_url(final_url)
            content_type = response.headers.get_content_type()
            body = response.read(self._max_download_bytes + 1)
        if len(body) > self._max_download_bytes:
            raise ValueError(f"response exceeded max_download_bytes={self._max_download_bytes}")
        return HttpResponse(url=final_url, content_type=content_type, body=body)

    def post_json(
        self,
        url: str,
        *,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> HttpResponse:
        _require_public_http_url(url)
        request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
            **headers,
        }
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        with self._opener.open(request, timeout=self._timeout_seconds) as response:
            final_url = response.geturl()
            _require_public_http_url(final_url)
            content_type = response.headers.get_content_type()
            body = response.read(self._max_download_bytes + 1)
        if len(body) > self._max_download_bytes:
            raise ValueError(f"response exceeded max_download_bytes={self._max_download_bytes}")
        return HttpResponse(url=final_url, content_type=content_type, body=body)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _require_public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class DuckDuckGoSearchProvider:
    """Best-effort no-key search adapter over DuckDuckGo's public HTML results."""

    name = "duckduckgo_html"

    def __init__(self, client: HttpClient, budget: SearchCallBudget | None = None) -> None:
        self._client = client
        self._budget = budget

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        url = f"https://lite.duckduckgo.com/lite/?q={quote(query)}"
        def request() -> list[SearchHit]:
            response = self._client.get(url, accept="text/html,application/xhtml+xml")
            parser = _DuckDuckGoParser()
            parser.feed(_decode_body(response.body, response.content_type))
            hits: list[SearchHit] = []
            seen: set[str] = set()
            for raw in parser.results:
                target = _duckduckgo_target(raw["url"])
                if not target or target in seen:
                    continue
                try:
                    _require_http_url_syntax(target)
                    hit = SearchHit(title=raw["title"], url=target, snippet=raw.get("snippet", ""))
                except ValueError:
                    continue
                hits.append(hit)
                seen.add(target)
                if len(hits) >= limit:
                    break
            return hits

        return self._search_with_budget(query, request)

    def _search_with_budget(
        self, query: str, request: Callable[[], list[SearchHit]]
    ) -> list[SearchHit]:
        if self._budget is None:
            return request()
        return self._budget.search(self.name, request)


class BingRssSearchProvider:
    """Fallback no-key adapter for Bing's RSS-formatted search results."""

    name = "bing_rss"

    def __init__(self, client: HttpClient, budget: SearchCallBudget | None = None) -> None:
        self._client = client
        self._budget = budget

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        url = f"https://www.bing.com/search?format=rss&setlang=en-US&q={quote(query)}"
        def request() -> list[SearchHit]:
            response = self._client.get(url, accept="application/rss+xml,application/xml,text/xml")
            root = ET.fromstring(response.body)
            hits: list[SearchHit] = []
            for item in root.findall("./channel/item"):
                title = _normalize_space(item.findtext("title") or "")
                target = _normalize_space(item.findtext("link") or "")
                snippet = _normalize_space(item.findtext("description") or "")
                if not title or not target:
                    continue
                try:
                    _require_http_url_syntax(target)
                    hits.append(SearchHit(title=title, url=target, snippet=snippet))
                except ValueError:
                    continue
                if len(hits) >= limit:
                    break
            return hits

        if self._budget is None:
            return request()
        return self._budget.search(self.name, request)


class GitHubRepositorySearchProvider:
    """Uses GitHub's public repository search for repository-shaped queries."""

    name = "github_repository_api"
    _QUERY_MARKERS = ("repository", "github", "repo")
    _REMOVED_TERMS = {
        "architecture",
        "benchmark",
        "benchmarks",
        "core",
        "documentation",
        "evaluation",
        "execution",
        "github",
        "methodology",
        "metrics",
        "official",
        "repo",
        "repository",
        "workflow",
    }

    def __init__(self, client: HttpClient, budget: SearchCallBudget | None = None) -> None:
        self._client = client
        self._budget = budget

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        if not any(marker in query.casefold() for marker in self._QUERY_MARKERS):
            return []
        project_name = self._project_name(query)
        if not project_name:
            return []
        search_terms = [f"{project_name} in:name"]
        hyphenated = _camel_to_hyphen(project_name)
        if " " not in project_name and hyphenated.casefold() != project_name.casefold():
            search_terms.append(f"{hyphenated} in:name")

        candidates: dict[str, SearchHit] = {}
        last_error: Exception | None = None
        for term in search_terms:
            url = (
                "https://api.github.com/search/repositories?"
                f"q={quote(term)}&per_page={min(max(limit, 3), 10)}"
            )

            def request() -> list[SearchHit]:
                response = self._client.get(url, accept="application/vnd.github+json")
                payload = json.loads(_decode_body(response.body, response.content_type))
                hits: list[SearchHit] = []
                for item in payload.get("items", []):
                    if not isinstance(item, dict) or not isinstance(item.get("html_url"), str):
                        continue
                    try:
                        hits.append(
                            SearchHit(
                                title=str(
                                    item.get("full_name")
                                    or item.get("name")
                                    or "GitHub repository"
                                ),
                                url=str(item["html_url"]),
                                snippet=str(item.get("description") or ""),
                            )
                        )
                    except ValueError:
                        continue
                return hits

            try:
                hits = request() if self._budget is None else self._budget.search(self.name, request)
            except Exception as error:
                last_error = error
                continue
            for hit in hits:
                candidates[str(hit.url)] = hit
        if not candidates and last_error is not None:
            raise last_error

        target = _compact_name(project_name)
        return sorted(
            candidates.values(),
            key=lambda hit: (
                _compact_name(hit.title.rsplit("/", maxsplit=1)[-1]) != target,
                -_name_overlap(project_name, hit.title.rsplit("/", maxsplit=1)[-1]),
                hit.title.casefold(),
            ),
        )[:limit]

    @classmethod
    def _project_name(cls, query: str) -> str:
        source_marker = re.search(r"\b(?:official\s+)?(?:github\s+)?repo(?:sitory)?\b", query, flags=re.IGNORECASE)
        candidate = query[: source_marker.start()] if source_marker else query
        terms = re.findall(r"[A-Za-z0-9_.-]+", candidate)
        kept = [term for term in terms if term.casefold() not in cls._REMOVED_TERMS]
        return " ".join(kept[:4]).strip()


class NoKeyWebSearchProvider:
    name = "no_key_web_search"

    def __init__(self, client: HttpClient, budget: SearchCallBudget | None = None) -> None:
        self._providers = [
            GitHubRepositorySearchProvider(client, budget),
            DuckDuckGoSearchProvider(client, budget),
            BingRssSearchProvider(client, budget),
        ]
        self.last_backend = ""

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        last_error: Exception | None = None
        for provider in self._providers:
            try:
                hits = provider.search(query, limit=limit)
            except Exception as error:
                last_error = error
                continue
            if hits:
                self.last_backend = provider.name
                return hits
        self.last_backend = self._providers[-1].name
        if last_error is not None:
            raise last_error
        return []


class TavilySearchProvider:
    """Keyed Tavily /search adapter with a small, explicit request surface."""

    name = "tavily_search"
    last_backend = name
    _ENDPOINT = "https://api.tavily.com/search"

    def __init__(
        self, client: JsonPostHttpClient, config: SearchConfig, budget: SearchCallBudget
    ) -> None:
        if config.kind != "tavily" or not config.api_key_env:
            raise ValueError("TavilySearchProvider requires search.kind=tavily and api_key_env")
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing search API key in environment variable {config.api_key_env}")
        self._client = client
        self._api_key = api_key
        self._search_depth = config.tavily_search_depth
        self._budget = budget
        self._estimated_cost_usd = config.tavily_basic_credit_price_usd

    def search(self, query: str, *, limit: int) -> list[SearchHit]:
        def request() -> list[SearchHit]:
            response = self._client.post_json(
                self._ENDPOINT,
                payload={
                    "query": query,
                    "search_depth": self._search_depth,
                    "max_results": limit,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                },
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            if response.content_type != "application/json":
                raise ValueError("Tavily search response must be application/json")
            payload = json.loads(_decode_body(response.body, response.content_type))
            raw_results = payload.get("results", [])
            if not isinstance(raw_results, list):
                raise ValueError("Tavily search response results must be a list")

            hits: list[SearchHit] = []
            seen: set[str] = set()
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                target = item.get("url")
                if not isinstance(target, str) or target in seen:
                    continue
                try:
                    _require_http_url_syntax(target)
                    title = _normalize_space(str(item.get("title") or urlparse(target).netloc))
                    if not title:
                        continue
                    hits.append(
                        SearchHit(
                            title=title,
                            url=target,
                            snippet=_normalize_space(str(item.get("content") or "")),
                        )
                    )
                except ValueError:
                    continue
                seen.add(target)
                if len(hits) >= limit:
                    break
            return hits

        return self._budget.search(
            self.name, request, estimated_cost_usd=self._estimated_cost_usd
        )


class LiveWebCollector:
    """Searches public web results, fetches pages, and retains a per-request audit trace."""

    def __init__(self, config: SearchConfig, *, client: HttpClient | None = None) -> None:
        self._config = config
        self._client = client or SafeHttpClient(
            timeout_seconds=config.timeout_seconds,
            max_download_bytes=config.max_download_bytes,
            user_agent=config.user_agent,
        )
        self._search_budget = SearchCallBudget(config.max_search_calls)
        if config.kind == "duckduckgo":
            self._search = NoKeyWebSearchProvider(self._client, self._search_budget)
        elif config.kind == "tavily":
            if not hasattr(self._client, "post_json"):
                raise TypeError("tavily search requires an HTTP client with post_json")
            self._search = TavilySearchProvider(  # type: ignore[arg-type]
                self._client, config, self._search_budget
            )
        else:
            raise ValueError(f"live web collector does not support search kind {config.kind}")
        self.trace_events: list[TraceEvent] = []

    def collect(self, queries: list[str], max_evidence: int) -> list[Evidence]:
        self.trace_events = []
        self._search_budget.reset()
        ranked_by_query = [
            self._search_query(query, logical_query_index=index)
            for index, query in enumerate(queries, start=1)
        ]
        candidates = _round_robin_hits(ranked_by_query, max_evidence * 2)
        evidence: list[Evidence] = []
        seen_urls: set[str] = set()
        for hit, query in candidates:
            normalized_url = str(hit.url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            fetched = self._fetch_evidence(hit, query)
            if fetched is not None:
                evidence.append(fetched)
            if len(evidence) >= max_evidence:
                break
        return evidence

    def _search_query(
        self, query: str, *, logical_query_index: int
    ) -> list[tuple[SearchHit, str]]:
        self._search_budget.begin_logical_query(query, logical_query_index)
        before_attempts = len(self._search_budget.attempts)
        try:
            hits = self._search.search(query, limit=self._config.max_results_per_query)
            self._append_search_attempt_trace(before_attempts)
            return [(hit, query) for hit in hits]
        except Exception as error:
            self._append_search_attempt_trace(before_attempts)
            if len(self._search_budget.attempts) == before_attempts:
                self.trace_events.append(
                    TraceEvent(
                        stage="search_budget",
                        provider=self._search.last_backend or self._search.name,
                        model="html-v1",
                        latency_ms=0,
                        outcome="error",
                        detail=json.dumps(
                            {
                                "logical_query_index": logical_query_index,
                                "query_sha256": sha256(query.encode("utf-8")).hexdigest(),
                                "error_type": type(error).__name__,
                                "attempted_http_search": False,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            return []

    def _append_search_attempt_trace(self, start_index: int) -> None:
        for attempt in self._search_budget.attempts[start_index:]:
            self.trace_events.append(
                TraceEvent(
                    stage="search",
                    provider=attempt.provider,
                    model="tavily-basic-v1" if attempt.provider == "tavily_search" else "html-v1",
                    latency_ms=attempt.latency_ms,
                    outcome=attempt.outcome,
                    detail=json.dumps(
                        {
                            "logical_query_index": attempt.logical_query_index,
                            "http_search_attempt_index": attempt.index,
                            "query_sha256": attempt.query_sha256,
                            "results": attempt.result_count,
                            "budget_before": attempt.index - 1,
                            "budget_after": self._search_budget.max_attempts - attempt.index,
                            "estimated_search_cost_usd": attempt.estimated_cost_usd,
                            "error_type": attempt.error_type,
                        },
                        ensure_ascii=False,
                    ),
                )
            )

    def _fetch_evidence(self, hit: SearchHit, query: str) -> Evidence | None:
        started = time.perf_counter()
        url = str(hit.url)
        try:
            fetch_url = _content_url(url)
            response = self._client.get(fetch_url, accept="text/html,text/plain,application/xhtml+xml,text/markdown")
            if response.content_type not in {"text/html", "text/plain", "text/markdown", "application/xhtml+xml"}:
                raise ValueError(f"unsupported content type {response.content_type}")
            document = _extract_document(response)
            excerpt = _select_excerpt(document.blocks, query, self._config.max_excerpt_characters)
            if not excerpt:
                excerpt = hit.snippet.strip()[: self._config.max_excerpt_characters]
            if not excerpt:
                raise ValueError("page contained no readable text")
            title = document.title or hit.title
            self.trace_events.append(
                TraceEvent(
                    stage="fetch",
                    provider="public_http",
                    model="html-text-v1",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    outcome="ok",
                    detail=json.dumps(
                        {
                            "citation_url": url,
                            "fetched_url": response.url,
                            "content_type": response.content_type,
                            "characters": len(excerpt),
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            return Evidence(
                id=f"web-{sha256(url.encode('utf-8')).hexdigest()[:12]}",
                title=title,
                url=url,
                excerpt=excerpt,
                query=query,
            )
        except Exception as error:
            self.trace_events.append(
                TraceEvent(
                    stage="fetch",
                    provider="public_http",
                    model="html-text-v1",
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    outcome="error",
                    detail=json.dumps({"url": url, "error": str(error)}, ensure_ascii=False),
                )
            )
            return None


def live_collector_from_config(config: SearchConfig) -> LiveWebCollector:
    if config.kind == "local":
        raise ValueError("live research requires search.kind=duckduckgo or tavily")
    return LiveWebCollector(config)


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture: str | None = None
        self._capture_tag = ""
        self._href = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key: value or "" for key, value in attrs}
        classes = set(values.get("class", "").split())
        if tag == "a" and ({"result__a", "result-link"} & classes):
            self._capture = "title"
            self._capture_tag = tag
            self._href = html.unescape(values.get("href", ""))
            self._parts = []
        elif tag == "a" and "result__snippet" in classes:
            self._capture = "snippet"
            self._capture_tag = tag
            self._href = html.unescape(values.get("href", ""))
            self._parts = []
        elif tag == "td" and "result-snippet" in classes:
            self._capture = "snippet"
            self._capture_tag = tag
            self._href = self.results[-1]["url"] if self.results else ""
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._capture_tag or not self._capture:
            return
        text = _normalize_space(" ".join(self._parts))
        if self._capture == "title" and text and self._href:
            self.results.append({"title": text, "url": self._href, "snippet": ""})
        elif self._capture == "snippet" and text:
            target = _duckduckgo_target(self._href)
            for result in reversed(self.results):
                if _duckduckgo_target(result["url"]) == target:
                    result["snippet"] = text
                    break
        self._capture = None
        self._capture_tag = ""
        self._href = ""
        self._parts = []


@dataclass(frozen=True)
class ExtractedDocument:
    title: str
    blocks: list[str]


class _ReadableHtmlParser(HTMLParser):
    _BLOCK_TAGS = {"h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"}
    _IGNORED_TAGS = {"script", "style", "svg", "nav", "footer", "form", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.blocks: list[str] = []
        self._ignored_depth = 0
        self._in_title = False
        self._block_depth = 0
        self._parts: list[str] = []
        self._description = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            values = {key.casefold(): value or "" for key, value in attrs}
            if values.get("name", "").casefold() == "description":
                self._description = values.get("content", "")
        if tag in self._BLOCK_TAGS:
            if self._block_depth == 0:
                self._parts = []
            self._block_depth += 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self.title += data
        if self._block_depth:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._IGNORED_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS and self._block_depth:
            self._block_depth -= 1
            if self._block_depth == 0:
                block = _normalize_space(" ".join(self._parts))
                if len(block) >= 40 and block not in self.blocks:
                    self.blocks.append(block)
                self._parts = []

    def document(self) -> ExtractedDocument:
        blocks = list(self.blocks)
        description = _normalize_space(self._description)
        if description and description not in blocks:
            blocks.insert(0, description)
        return ExtractedDocument(title=_normalize_space(self.title), blocks=blocks)


def _extract_document(response: HttpResponse) -> ExtractedDocument:
    text = _decode_body(response.body, response.content_type)
    if response.content_type in {"text/plain", "text/markdown"}:
        blocks = [_normalize_space(block) for block in re.split(r"\n\s*\n", text)]
        return ExtractedDocument(title="", blocks=[block for block in blocks if len(block) >= 40])
    parser = _ReadableHtmlParser()
    parser.feed(text)
    return parser.document()


def _select_excerpt(blocks: list[str], query: str, limit: int) -> str:
    terms = {term.casefold() for term in re.findall(r"[\w.-]{2,}", query)}
    document_frequency = {
        term: sum(term in block.casefold() for block in blocks)
        for term in terms
    }
    ranked = sorted(
        enumerate(blocks),
        key=lambda item: (
            -sum(
                1.0 / document_frequency[term]
                for term in terms
                if document_frequency[term] and term in item[1].casefold()
            ),
            item[0],
        ),
    )
    selected: list[str] = []
    characters = 0
    for _, block in ranked:
        remaining = limit - characters
        if remaining <= 0:
            break
        piece = block[:remaining].strip()
        if piece:
            selected.append(piece)
            characters += len(piece) + 2
        if len(selected) >= 6:
            break
    return "\n\n".join(selected)[:limit].strip()


def _round_robin_hits(
    ranked_by_query: list[list[tuple[SearchHit, str]]],
    limit: int,
) -> list[tuple[SearchHit, str]]:
    selected: list[tuple[SearchHit, str]] = []
    positions = [0] * len(ranked_by_query)
    while len(selected) < limit:
        added = False
        for index, rows in enumerate(ranked_by_query):
            if positions[index] >= len(rows):
                continue
            selected.append(rows[positions[index]])
            positions[index] += 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def _duckduckgo_target(url: str) -> str | None:
    absolute = urljoin("https://duckduckgo.com", html.unescape(url))
    parsed = urlparse(absolute)
    if parsed.hostname and parsed.hostname.casefold().endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else None
    return absolute


def _content_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname != "github.com":
        return url
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 2:
        owner, repository = parts
        return f"https://raw.githubusercontent.com/{owner}/{repository}/HEAD/README.md"
    if len(parts) >= 5 and parts[2] == "blob":
        owner, repository, _, revision, *path = parts
        return f"https://raw.githubusercontent.com/{owner}/{repository}/{revision}/{'/'.join(path)}"
    return url


def _camel_to_hyphen(value: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", value)
    return re.sub(r"[\s_]+", "-", words).strip("-")


def _compact_name(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _name_overlap(left: str, right: str) -> int:
    left_terms = set(re.findall(r"[a-z0-9]+", _camel_to_hyphen(left).casefold()))
    right_terms = set(re.findall(r"[a-z0-9]+", _camel_to_hyphen(right).casefold()))
    return len(left_terms & right_terms)


def _require_public_http_url(url: str) -> None:
    parsed = _require_http_url_syntax(url)
    hostname = parsed.hostname.casefold()
    if hostname == "localhost" or hostname.endswith(".local"):
        raise ValueError("local network URLs are not allowed")
    addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    if not addresses:
        raise ValueError("URL hostname did not resolve")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private, loopback, link-local, and reserved URLs are not allowed")


def _require_http_url_syntax(url: str) -> ParseResult:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("only public http(s) URLs without embedded credentials are allowed")
    return parsed


def _decode_body(body: bytes, content_type: str) -> str:
    del content_type
    for encoding in ("utf-8", "gb18030", "latin-1"):
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
