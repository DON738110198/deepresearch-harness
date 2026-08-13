from urllib.parse import quote

import pytest

from deepresearch_harness.contracts import SearchConfig
from deepresearch_harness.web_research import (
    GitHubRepositorySearchProvider,
    HttpResponse,
    LiveWebCollector,
    SafeHttpClient,
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
