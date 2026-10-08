import asyncio

import pytest

from ai_status_report.mcp import client as mcp_client
from ai_status_report.mcp.client import (
    TavilyMCPClient,
    TavilyRateLimitedError,
    TavilyRequestPacer,
    _rate_limit_metadata,
)


def test_request_pacer_defers_the_next_shared_call(monkeypatch):
    pacer = TavilyRequestPacer(1)
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    async def run():
        await pacer.wait_turn()
        await pacer.defer(4)
        await pacer.wait_turn()

    asyncio.run(run())

    assert sleeps
    assert sleeps[-1] >= 3


def test_rate_limit_metadata_uses_structured_status_without_error_text():
    assert _rate_limit_metadata({"status": 429, "retry_after": 17}) == (True, 17)
    assert _rate_limit_metadata({"nested": [{"status": "429"}]}) == (True, None)
    assert _rate_limit_metadata({"error": "excessive requests"}) == (False, None)


def test_tavily_client_retries_only_the_rate_limited_tool_call(monkeypatch):
    client = TavilyMCPClient(
        "https://example.invalid/mcp/",
        "test-key",
        pacer=TavilyRequestPacer(0),
        fallback_retry_after_seconds=0,
    )
    attempts = []
    retries = []

    async def fake_call_once(name, arguments):
        attempts.append((name, arguments))
        if len(attempts) == 1:
            raise TavilyRateLimitedError(name, None)
        return {"results": []}

    monkeypatch.setattr(client, "_call_once", fake_call_once)

    async def run():
        return await client.call(
            "tavily_search",
            {"query": "test"},
            on_attempt=lambda: None,
            on_rate_limit=lambda error, attempt, delay: retries.append(
                (str(error), attempt, delay)
            ),
        )

    assert asyncio.run(run()) == {"results": []}
    assert attempts == [
        ("tavily_search", {"query": "test"}),
        ("tavily_search", {"query": "test"}),
    ]
    assert retries == [("tavily_rate_limited", 2, 0)]


def test_tavily_client_stops_after_one_rate_limit_retry(monkeypatch):
    client = TavilyMCPClient(
        "https://example.invalid/mcp/",
        "test-key",
        pacer=TavilyRequestPacer(0),
        fallback_retry_after_seconds=0,
    )
    calls = []

    async def always_rate_limited(name, _arguments):
        calls.append(name)
        raise TavilyRateLimitedError(name, None)

    monkeypatch.setattr(client, "_call_once", always_rate_limited)

    with pytest.raises(TavilyRateLimitedError, match="tavily_rate_limited"):
        asyncio.run(client.call("tavily_extract", {"urls": []}))

    assert calls == ["tavily_extract", "tavily_extract"]
