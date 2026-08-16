from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class SearchServiceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchServiceHealth:
    health_url: str
    retriever_id: str


def health_url_for_search(search_url: str) -> str:
    parsed = urlsplit(search_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("search URL must be absolute HTTP(S)")
    if parsed.query or parsed.fragment:
        raise ValueError("search URL cannot contain query or fragment components")
    path = parsed.path.rstrip("/")
    if not path.endswith("/search"):
        raise ValueError("search URL path must end with /search")
    health_path = f"{path[:-len('/search')]}/health" or "/health"
    return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def require_search_service_health(
    search_url: str,
    *,
    expected_retriever_id: str,
    timeout_seconds: float = 5.0,
) -> SearchServiceHealth:
    health_url = health_url_for_search(search_url)
    try:
        with urlopen(
            Request(health_url, headers={"Accept": "application/json"}),
            timeout=timeout_seconds,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as error:
        raise SearchServiceUnavailable(
            f"search_service_unhealthy:{health_url}:{type(error).__name__}"
        ) from error
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise SearchServiceUnavailable(
            f"search_service_unhealthy:{health_url}:invalid_status"
        )
    retriever_id = payload.get("retriever_id")
    if retriever_id != expected_retriever_id:
        raise SearchServiceUnavailable(
            "search_service_unhealthy:retriever_id_mismatch:"
            f"{retriever_id!r}!={expected_retriever_id!r}"
        )
    return SearchServiceHealth(
        health_url=health_url,
        retriever_id=expected_retriever_id,
    )
