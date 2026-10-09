import json

import httpx
import pytest

from ai_status_report.model.client import DeepSeekClient, ProviderClient, ProviderError
from ai_status_report.rag.glm import GLMClient
from ai_status_report.settings import Profile
from ai_status_report.smoke import message, safe_usage
from ai_status_report.token_budget.allocator import RUN, SEARCH_REQUESTS, BudgetLimits, TokenLedger

PROFILE = Profile("test", "https://api.deepseek.com", "synthetic-test-secret")


def test_chat_request_explicit_mode_and_budget():
    def handler(request):
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 128
        assert request.headers["Authorization"] == "Bearer synthetic-test-secret"
        return httpx.Response(200, json={"choices": []})

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        assert client.chat([{"role": "user", "content": "hello"}]) == {"choices": []}


def test_chat_accounts_provider_usage_in_real_ledger():
    ledger = TokenLedger("run-ledger", BudgetLimits({RUN: 10_000}))

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    with DeepSeekClient(
        PROFILE,
        5,
        transport=httpx.MockTransport(handler),
        ledger=ledger,
    ) as client:
        client.chat([{"role": "user", "content": "hello"}], max_tokens=32)

    assert ledger.actual[RUN] == 10
    assert ledger.remaining(RUN) == 9_990


def test_web_search_requests_structured_citations():
    def handler(request):
        body = json.loads(request.content)
        assert body["tool_choice"] == {"type": "web_search"}
        assert body["text"]["format"] == {"type": "json_object"}
        assert "sources" in body["instructions"]
        return httpx.Response(200, json={"status": "completed"})

    ledger = TokenLedger("web-search", BudgetLimits({RUN: 100_000, SEARCH_REQUESTS: 2}))
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler), ledger=ledger) as client:
        assert client.web_search("hello")["status"] == "completed"
    assert ledger.actual[SEARCH_REQUESTS] == 1
    assert ledger.unknown[RUN] > 0


@pytest.mark.parametrize("budget", [1023, 384001])
def test_web_search_rejects_budget_outside_provider_limits(budget):
    with (
        DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client,
        pytest.raises(ProviderError, match="^invalid_web_search_budget$"),
    ):
        client.web_search("hello", max_output_tokens=budget)


@pytest.mark.parametrize("status", [301, 400, 401, 429, 500])
def test_http_errors_redacted_and_not_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            status, text="synthetic-test-secret", headers={"Location": "https://evil.example"}
        )

    with (
        ProviderClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ProviderError, match=f"^http_{status}$"),
    ):
        client.post("/embeddings", {})
    assert len(calls) == 1


def test_timeout_is_sanitized():
    def handler(request):
        raise httpx.ReadTimeout("synthetic-test-secret", request=request)

    with (
        ProviderClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ProviderError, match="^timeout$"),
    ):
        client.post("/embeddings", {})


@pytest.mark.parametrize("done", [True, False])
def test_stream_completion_required(done):
    events = 'data: {"choices":[{"delta":{"content":"READY"},"finish_reason":"stop"}]}\n\n'
    if done:
        events += 'data: {"choices":[],"usage":{"total_tokens":9}}\n\ndata: [DONE]\n\n'
    ledger = TokenLedger("stream", BudgetLimits({RUN: 10_000}))
    with DeepSeekClient(
        PROFILE,
        5,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=events)),
        ledger=ledger,
    ) as client:
        if done:
            assert client.stream_probe([])["usage"]["total_tokens"] == 9
            assert ledger.actual[RUN] == 9
        else:
            with pytest.raises(ProviderError, match="stream_missing_done"):
                client.stream_probe([])
            assert ledger.unknown[RUN] > 0


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 1, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [True, 2.0]}],
    ],
)
def test_embedding_rejects_invalid_shape(rows):
    with (
        GLMClient(
            PROFILE,
            5,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": rows})),
        ) as client,
        pytest.raises(ProviderError),
    ):
        client.embed(["example"], 2)


def test_embedding_reorders_indices():
    rows = [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]
    with GLMClient(
        PROFILE,
        5,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": rows})),
    ) as client:
        assert client.embed(["a", "b"], 1)["data"][0]["embedding"] == [1.0]


def test_embedding_accounts_provider_usage():
    ledger = TokenLedger("embedding", BudgetLimits({RUN: 1000}))
    response = {
        "data": [{"index": 0, "embedding": [1.0]}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 0, "total_tokens": 7},
    }
    with GLMClient(
        PROFILE,
        5,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        ledger=ledger,
    ) as client:
        client.embed(["a"], 1)
    assert ledger.actual[RUN] == 7


def test_rerank_uses_official_payload_and_returns_scores():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/rerank"
        assert body == {
            "model": "rerank",
            "query": "查询",
            "documents": ["候选一", "候选二"],
            "top_n": 2,
            "return_documents": False,
        }
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]},
        )

    with GLMClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        assert client.rerank("查询", ["候选一", "候选二"], top_n=2)["results"][0]["index"] == 1


@pytest.mark.parametrize("query, documents", [("", ["a"]), ("q", []), ("q", ["a" * 4097])])
def test_rerank_rejects_input_bounds(query, documents):
    with (
        GLMClient(PROFILE, 5, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client,
        pytest.raises(ProviderError),
    ):
        client.rerank(query, documents)


def test_truncated_generation_rejected():
    with pytest.raises(ProviderError, match="unexpected_finish_reason"):
        message({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})


def test_usage_allowlist():
    assert safe_usage({"usage": {"total_tokens": 5, "other": "secret", "prompt_tokens": True}}) == {
        "total_tokens": 5
    }


def test_provider_rejects_removed_document_parsing_endpoint():
    with (
        GLMClient(PROFILE, 5, transport=httpx.MockTransport(lambda request: httpx.Response(200))) as client,
        pytest.raises(ProviderError, match="unapproved_api_path"),
    ):
        client.post("/layout_parsing", {})
