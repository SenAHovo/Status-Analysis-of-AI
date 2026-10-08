import asyncio
from typing import ClassVar

import pytest

from ai_status_report.a2a import apps
from ai_status_report.mcp.client import TavilyRateLimitedError
from ai_status_report.schemas.search import SearchLimits, SearchPlan, SearchQuery


def test_unconfigured_search_provider_fails_before_planning(monkeypatch):
    class Settings:
        tavily_api_key = ""
        tavily_mcp_url = "http://unused"
        deepseek = None
        timeout = 5.0
        web_search_max_output_tokens = 65536

    monkeypatch.setattr(apps, "load_search_settings", lambda root: Settings())
    application = apps.network_search_app()
    dispatcher = application.routes[-1].endpoint.__self__
    executor = dispatcher.request_handler.agent_executor

    with pytest.raises(RuntimeError, match="no_search_provider_configured"):
        asyncio.run(executor.search({"query": "test"}))


def test_search_agent_closes_deepseek_client_when_planning_fails(monkeypatch, tmp_path):
    class Settings:
        tavily_api_key = ""
        tavily_mcp_url = "http://unused"
        timeout = 5.0

        class deepseek:
            model = "deepseek-v4-flash"

    class Http:
        closed = False

        def close(self):
            self.closed = True

    clients = []

    class FakeDeepSeekClient:
        def __init__(self, *_args, **_kwargs):
            self.http = Http()
            clients.append(self)

    class FailingPlanner:
        def __init__(self, *_args, **_kwargs):
            pass

        def plan(self, _task):
            raise RuntimeError("planner_failed")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(apps, "load_search_settings", lambda root: Settings())
    monkeypatch.setattr(apps, "new_run_ledger", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(apps, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(apps, "SearchPlanner", FailingPlanner)
    application = apps.network_search_app()
    dispatcher = application.routes[-1].endpoint.__self__
    executor = dispatcher.request_handler.agent_executor

    with pytest.raises(RuntimeError, match="planner_failed"):
        asyncio.run(executor.search({"run_id": "run-1", "query": "test"}))

    assert len(clients) == 1
    assert clients[0].http.closed is True


def test_search_agent_stops_remaining_queries_after_tavily_rate_limit(monkeypatch, tmp_path):
    class Settings:
        tavily_api_key = "test-key"
        tavily_mcp_url = "https://example.invalid/mcp/"
        deepseek = None
        timeout = 5.0

    class Ledger:
        def __init__(self):
            self.calls = 0

        def meter(self, _kind):
            self.calls += 1

    ledger = Ledger()

    class Planner:
        def __init__(self, *_args, **_kwargs):
            pass

        def plan(self, _task):
            return SearchPlan(
                original_query="test",
                queries=[SearchQuery(query="first"), SearchQuery(query="second")],
                providers=["tavily"],
                evidence_requirement="retrievable",
                limits=SearchLimits(max_queries=2),
            )

    class Tavily:
        calls: ClassVar[list[str]] = []

        def __init__(self, *_args, **_kwargs):
            pass

        async def call(self, name, _arguments, *, on_attempt, on_rate_limit):
            self.calls.append(name)
            on_attempt()
            await on_rate_limit(TavilyRateLimitedError(name, 5), 2, 5)
            raise TavilyRateLimitedError(name, 5)

    progress = []

    async def collect_progress(code):
        progress.append(code)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(apps, "load_search_settings", lambda root: Settings())
    monkeypatch.setattr(apps, "new_run_ledger", lambda *_args, **_kwargs: ledger)
    monkeypatch.setattr(apps, "TavilyMCPClient", Tavily)
    monkeypatch.setattr(apps, "SearchPlanner", Planner)
    monkeypatch.setattr(apps, "validate_plan", lambda plan, _task, **_kwargs: plan)
    application = apps.network_search_app()
    dispatcher = application.routes[-1].endpoint.__self__
    executor = dispatcher.request_handler.agent_executor

    report = asyncio.run(
        executor.search({"run_id": "run-rate-limited", "query": "test"}, progress=collect_progress)
    )

    assert '"status":"failed"' in report
    assert '"incomplete_reason":"tavily_rate_limited"' in report
    assert Tavily.calls == ["tavily_search"]
    assert ledger.calls == 1
    assert progress == ["tavily_rate_limited_wait:5:2"]
