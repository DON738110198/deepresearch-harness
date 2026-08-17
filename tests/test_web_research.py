import json
from urllib.parse import quote

import pytest
from pydantic import ValidationError

from deepresearch_harness.contracts import SearchConfig
from deepresearch_harness.web_research import (
    GitHubRepositorySearchProvider,
    HttpResponse,
    LiveWebCollector,
    SafeHttpClient,
    SearchBudgetExhausted,
    SearchCallBudget,
    TavilySearchProvider,
)


class FixtureHttpClient:
    def __init__(self) -> None:
        search_url = "https://lite.duckduckgo.com/lite/?q=" + quote("deep research evaluation")
        self.responses = {
            search_url: HttpResponse(
                url=search_url,
                content_type="text/html",
                body=b"""
                <html><body>
                  <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">First result</a>
                  <td class="result-snippet">Rubric based evaluation.</td>
                  <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo">Second result</a>
                  <td class="result-snippet">Citation support checks.</td>
                </body></html>
                """,
            ),
            "https://example.com/one": HttpResponse(
                url="https://example.com/one",
                content_type="text/html",
                body=b"""
                <html><head><title>Evaluation One</title></head><body>
                  <nav><p>This navigation text must be ignored even though it is long.</p></nav>
                  <h1>Deep research evaluation</h1>
                  <p>Rubric based evaluation checks whether a report covers task-specific requirements with evidence.</p>
                </body></html>
                """,
            ),
            "https://example.org/two": HttpResponse(
                url="https://example.org/two",
                content_type="text/html",
                body=b"""
                <html><head><title>Evaluation Two</title></head><body>
                  <p>Citation evaluation verifies whether a linked source supports the exact factual claim.</p>
                </body></html>
                """,
            ),
        }

    def get(self, url: str, *, accept: str) -> HttpResponse:
        assert accept
        return self.responses[url]


def test_live_collector_searches_fetches_and_traces_without_an_api_key() -> None:
    collector = LiveWebCollector(
        SearchConfig(
            kind="duckduckgo",
            max_results_per_query=2,
            max_excerpt_characters=400,
        ),
        client=FixtureHttpClient(),
    )

    evidence = collector.collect(["deep research evaluation"], max_evidence=2)

    assert [item.title for item in evidence] == ["Evaluation One", "Evaluation Two"]
    assert all(item.query == "deep research evaluation" for item in evidence)
    assert "Rubric based evaluation" in evidence[0].excerpt
    assert "navigation text" not in evidence[0].excerpt
    assert [event.stage for event in collector.trace_events] == ["search", "fetch", "fetch"]
    assert all(event.outcome == "ok" for event in collector.trace_events)
    assert "example.com/one" in collector.trace_events[1].detail


def test_safe_http_client_rejects_local_network_targets_before_fetch() -> None:
    client = SafeHttpClient(
        timeout_seconds=1,
        max_download_bytes=16_384,
        user_agent="test-agent",
    )

    with pytest.raises(ValueError, match="not allowed"):
        client.get("http://127.0.0.1/private", accept="text/plain")

    with pytest.raises(ValueError, match="not allowed"):
        client.post_json(
            "http://127.0.0.1/private",
            payload={"query": "private"},
            headers={"Authorization": "Bearer test-only-key"},
        )


def test_tavily_config_requires_the_fixed_environment_variable_name() -> None:
    with pytest.raises(ValidationError, match="TAVILY_API_KEY"):
        SearchConfig(kind="tavily", api_key_env="OTHER_KEY")


def test_tavily_search_uses_small_fixed_request_and_redacts_key_from_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-only-tavily-key")

    class TavilyFixtureHttpClient:
        def __init__(self) -> None:
            self.post_calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

        def post_json(
            self,
            url: str,
            *,
            payload: dict[str, object],
            headers: dict[str, str],
        ) -> HttpResponse:
            self.post_calls.append((url, payload, headers))
            return HttpResponse(
                url=url,
                content_type="application/json",
                body=b'''{"answer":"must-not-enter-evidence","results":[
                  {"title":"Evaluation One","url":"https://example.com/one",
                   "content":"search snippet","raw_content":"must-not-enter-evidence"}
                ]}''',
            )

        def get(self, url: str, *, accept: str) -> HttpResponse:
            assert url == "https://example.com/one"
            assert accept
            return HttpResponse(
                url=url,
                content_type="text/html",
                body=b"<html><head><title>Evaluation One</title></head><body><p>Direct source evidence supports the evaluation claim.</p></body></html>",
            )

    client = TavilyFixtureHttpClient()
    collector = LiveWebCollector(
        SearchConfig(kind="tavily", api_key_env="TAVILY_API_KEY", max_search_calls=5),
        client=client,
    )
    evidence = collector.collect(["deep research evaluation"], max_evidence=1)

    assert len(evidence) == 1
    assert "must-not-enter-evidence" not in evidence[0].excerpt
    endpoint, payload, headers = client.post_calls[0]
    assert endpoint == "https://api.tavily.com/search"
    assert payload == {
        "query": "deep research evaluation",
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }
    assert headers["Authorization"] == "Bearer test-only-tavily-key"
    search_event = collector.trace_events[0]
    assert search_event.provider == "tavily_search"
    assert search_event.model == "tavily-basic-v1"
    trace = json.loads(search_event.detail)
    assert trace["http_search_attempt_index"] == 1
    assert trace["estimated_search_cost_usd"] == pytest.approx(0.008)
    assert "test-only-tavily-key" not in "\n".join(event.detail for event in collector.trace_events)
    assert "must-not-enter-evidence" not in "\n".join(event.detail for event in collector.trace_events)


def test_tavily_attempt_budget_blocks_sixth_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-only-tavily-key")

    class CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self, url: str, *, payload: dict[str, object], headers: dict[str, str]
        ) -> HttpResponse:
            assert url == "https://api.tavily.com/search"
            assert payload["include_answer"] is False
            assert headers["Authorization"] == "Bearer test-only-tavily-key"
            self.calls += 1
            return HttpResponse(url=url, content_type="application/json", body=b'{"results": []}')

        def get(self, url: str, *, accept: str) -> HttpResponse:
            raise AssertionError("Tavily empty results must not fetch documents")

    client = CountingClient()
    budget = SearchCallBudget(max_attempts=5)
    provider = TavilySearchProvider(
        client,
        SearchConfig(kind="tavily", api_key_env="TAVILY_API_KEY"),
        budget,
    )
    for index in range(1, 6):
        budget.begin_logical_query(f"query {index}", index)
        assert provider.search(f"query {index}", limit=5) == []
    budget.begin_logical_query("query 6", 6)
    with pytest.raises(SearchBudgetExhausted, match="exhausted"):
        provider.search("query 6", limit=5)

    assert client.calls == 5
    assert len(budget.attempts) == 5


def test_live_collector_records_blocked_logical_query_without_exceeding_search_attempt_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-only-tavily-key")

    class EmptyTavilyClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self, url: str, *, payload: dict[str, object], headers: dict[str, str]
        ) -> HttpResponse:
            self.calls += 1
            return HttpResponse(url=url, content_type="application/json", body=b'{"results": []}')

        def get(self, url: str, *, accept: str) -> HttpResponse:
            raise AssertionError("empty Tavily results must not fetch documents")

    client = EmptyTavilyClient()
    collector = LiveWebCollector(
        SearchConfig(kind="tavily", api_key_env="TAVILY_API_KEY", max_search_calls=5),
        client=client,
    )
    assert collector.collect([f"query {index}" for index in range(6)], max_evidence=1) == []

    assert client.calls == 5
    assert sum(event.stage == "search" for event in collector.trace_events) == 5
    budget_events = [event for event in collector.trace_events if event.stage == "search_budget"]
    assert len(budget_events) == 1
    assert json.loads(budget_events[0].detail)["attempted_http_search"] is False


def test_tavily_failed_request_counts_once_and_redacts_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-only-tavily-key")

    class FailingClient:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self, url: str, *, payload: dict[str, object], headers: dict[str, str]
        ) -> HttpResponse:
            self.calls += 1
            raise RuntimeError("HTTP 429 test-only-tavily-key")

        def get(self, url: str, *, accept: str) -> HttpResponse:
            raise AssertionError("no document fetch expected")

    budget = SearchCallBudget(max_attempts=5)
    provider = TavilySearchProvider(
        FailingClient(),
        SearchConfig(kind="tavily", api_key_env="TAVILY_API_KEY"),
        budget,
    )
    budget.begin_logical_query("retry me", 1)
    with pytest.raises(RuntimeError, match="429"):
        provider.search("retry me", limit=5)

    assert provider._client.calls == 1
    assert budget.attempts[0].outcome == "error"
    assert budget.attempts[0].error_type == "RuntimeError"
    assert "test-only-tavily-key" not in repr(budget.attempts[0])


def test_tavily_missing_key_fails_before_a_network_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    class UnusedClient:
        def get(self, url: str, *, accept: str) -> HttpResponse:
            raise AssertionError("no network call expected")

        def post_json(
            self, url: str, *, payload: dict[str, object], headers: dict[str, str]
        ) -> HttpResponse:
            raise AssertionError("no network call expected")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        LiveWebCollector(
            SearchConfig(kind="tavily", api_key_env="TAVILY_API_KEY"),
            client=UnusedClient(),
        )


def test_repository_queries_prefer_exact_github_repository_name() -> None:
    query = "Open Deep Research official repository workflow benchmark"
    search_url = (
        "https://api.github.com/search/repositories?q="
        + quote("Open Deep Research in:name")
        + "&per_page=3"
    )

    class GitHubFixtureClient:
        def get(self, url: str, *, accept: str) -> HttpResponse:
            assert url == search_url
            assert accept == "application/vnd.github+json"
            return HttpResponse(
                url=url,
                content_type="application/json",
                body=b"""
                {"items": [
                  {"name": "open-deep-research-notes", "full_name": "demo/open-deep-research-notes",
                   "html_url": "https://github.com/demo/open-deep-research-notes", "description": "Notes", "stargazers_count": 50},
                  {"name": "open_deep_research", "full_name": "langchain-ai/open_deep_research",
                   "html_url": "https://github.com/langchain-ai/open_deep_research", "description": "Official project", "stargazers_count": 10}
                ]}
                """,
            )

    hits = GitHubRepositorySearchProvider(GitHubFixtureClient()).search(query, limit=3)

    assert str(hits[0].url).rstrip("/") == "https://github.com/langchain-ai/open_deep_research"
    assert hits[0].title == "langchain-ai/open_deep_research"


def test_camel_case_repository_query_tries_hyphenated_name() -> None:
    first_url = (
        "https://api.github.com/search/repositories?q="
        + quote("DeerFlow in:name")
        + "&per_page=3"
    )
    second_url = (
        "https://api.github.com/search/repositories?q="
        + quote("Deer-Flow in:name")
        + "&per_page=3"
    )

    class CamelCaseFixtureClient:
        def get(self, url: str, *, accept: str) -> HttpResponse:
            assert accept == "application/vnd.github+json"
            if url == first_url:
                body = b'{"items": []}'
            elif url == second_url:
                body = (
                    b'{"items": [{"name": "deer-flow", "full_name": "bytedance/deer-flow", '
                    b'"html_url": "https://github.com/bytedance/deer-flow", "description": "Official", '
                    b'"stargazers_count": 100}]}'
                )
            else:
                raise AssertionError(url)
            return HttpResponse(url=url, content_type="application/json", body=body)

    hits = GitHubRepositorySearchProvider(CamelCaseFixtureClient()).search(
        "DeerFlow official repository workflow benchmark",
        limit=3,
    )

    assert hits[0].title == "bytedance/deer-flow"
