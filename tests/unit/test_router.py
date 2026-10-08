import json
import sqlite3

import httpx
import pytest

from ai_status_report.model.client import DeepSeekClient
from ai_status_report.model.router import DeepSeekRouter, ModelRouteError, ResponseCache
from ai_status_report.settings import Profile

PROFILE = Profile("deepseek-v4-flash", "https://api.deepseek.com", "test-secret")


def test_deterministic_route_uses_zero_requests(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        route = DeepSeekRouter(client, ResponseCache(tmp_path / "cache.sqlite")).route_intent(
            "分析人工智能现状"
        )
    assert route.source == "deterministic"
    assert route.intent.value == "report_request"
    assert calls == []


def test_status_question_uses_deterministic_research_route(tmp_path):
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        route = DeepSeekRouter(client, ResponseCache(tmp_path / "cache.sqlite")).route_intent(
            "最近大模型的发展趋势如何"
        )
    assert route.intent.value == "report_request"
    assert route.source == "deterministic"


def test_model_route_is_cached(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({
                    "intent": "report_request", "reason": "model_reason", "confidence": 0.91
                })}}]
            },
        )

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        router = DeepSeekRouter(client, ResponseCache(tmp_path / "cache.sqlite"))
        first = router.route_intent("帮我判断这个请求", allow_model=True)
        second = router.route_intent("帮我判断这个请求", allow_model=True)
    assert first.source == "model"
    assert second.source == "model_cache"
    assert len(calls) == 1


def test_model_failure_falls_back_without_secret_in_reason(tmp_path):
    def handler(request):
        return httpx.Response(500, text="test-secret")

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        route = DeepSeekRouter(client, ResponseCache(tmp_path / "cache.sqlite")).route_intent(
            "这个请求很模糊", allow_model=True
        )
    assert route.source == "deterministic_fallback"
    assert "test-secret" not in route.reason


def test_strict_model_failure_stops_without_deterministic_route(tmp_path):
    def handler(request):
        return httpx.Response(500, text="test-secret")

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        router = DeepSeekRouter(client, ResponseCache(tmp_path / "cache.sqlite"))
        with pytest.raises(ModelRouteError, match="^intent_model_failed$"):
            router.route_intent(
                "这个请求很模糊",
                allow_model=True,
                fallback_to_deterministic=False,
            )


def test_invalid_cached_route_is_deleted_and_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({
                    "intent": "report_request", "reason": "fresh", "confidence": 0.8
                })}}]
            },
        )

    cache_path = tmp_path / "cache.sqlite"
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        router = DeepSeekRouter(client, ResponseCache(cache_path))
        text = "帮我判断这个请求"
        messages = [
            {"role": "system", "content": (
                "你是项目入口路由器。只输出 JSON。将用户输入分类为 report_request、"
                "conversation、unsupported。必须包含 intent、reason、confidence。"
                "凡是询问人工智能、大模型、机器学习等领域的当前现状、行业变化、发展趋势、"
                "技术进展、市场格局、应用、影响、机会或挑战，都属于 report_request，"
                "即使用户没有使用‘报告’或‘分析’这些词。只有闲聊、概念解释、问候、"
                "能力询问才属于 conversation；天气、订票、邮件等范围外任务属于 unsupported。"
            )},
            {"role": "user", "content": text},
        ]
        options = {
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": router.max_output_tokens,
        }
        from ai_status_report.model.router import _cache_key

        key = _cache_key(PROFILE.model, messages, options)
        router.cache.put(key, {"intent": "report_request", "reason": "old", "confidence": 0.1})
        with sqlite3.connect(cache_path) as db:
            db.execute(
                "UPDATE responses SET payload = ? WHERE cache_key = ?",
                (json.dumps(["corrupted"]), key),
            )
        route = router.route_intent(text, allow_model=True)

    assert route.source == "model"
    assert len(calls) == 1


def test_invalid_cached_route_fails_at_strict_model_boundary(tmp_path):
    cache_path = tmp_path / "cache.sqlite"
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(lambda request: httpx.Response(500))) as client:
        router = DeepSeekRouter(client, ResponseCache(cache_path))
        text = "帮我判断这个请求"
        messages = [
            {"role": "system", "content": (
                "你是项目入口路由器。只输出 JSON。将用户输入分类为 report_request、"
                "conversation、unsupported。必须包含 intent、reason、confidence。"
                "凡是询问人工智能、大模型、机器学习等领域的当前现状、行业变化、发展趋势、"
                "技术进展、市场格局、应用、影响、机会或挑战，都属于 report_request，"
                "即使用户没有使用‘报告’或‘分析’这些词。只有闲聊、概念解释、问候、"
                "能力询问才属于 conversation；天气、订票、邮件等范围外任务属于 unsupported。"
            )},
            {"role": "user", "content": text},
        ]
        options = {
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": router.max_output_tokens,
        }
        from ai_status_report.model.router import _cache_key

        key = _cache_key(PROFILE.model, messages, options)
        router.cache.put(key, {"intent": "report_request", "reason": "old", "confidence": 0.1})
        with sqlite3.connect(cache_path) as db:
            db.execute(
                "UPDATE responses SET payload = ? WHERE cache_key = ?",
                (json.dumps({"intent": "invalid"}), key),
            )
        with pytest.raises(ModelRouteError, match="^intent_model_failed$"):
            router.route_intent(text, allow_model=True, fallback_to_deterministic=False)
