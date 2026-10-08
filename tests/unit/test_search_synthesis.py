import json

import httpx

from ai_status_report.model.client import DeepSeekClient
from ai_status_report.model.router import ResponseCache
from ai_status_report.schemas.search import ResearchReport, ResearchSource
from ai_status_report.search.synthesis import synthesize_report
from ai_status_report.settings import Profile

PROFILE = Profile("deepseek-v4-flash", "https://api.deepseek.com", "test-secret")


def _report() -> ResearchReport:
    return ResearchReport(
        query="人工智能支付协议现状",
        provider="tavily_mcp",
        summary="原始搜索摘要",
        sources=[
            ResearchSource(
                source_id="tavily-source-1",
                provider="tavily_mcp",
                title="Payment protocol",
                url="https://example.com/payment",
                snippet="The protocol defines authorization and settlement.",
                retrieved_at="2026-09-10T00:00:00Z",
                verification_status="retrieved",
            )
        ],
    )


def test_synthesis_generates_source_grounded_report_and_caches(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "summary": "协议重点集中在授权和结算。",
                                    "key_findings": [
                                        {
                                            "finding_id": "finding-1",
                                            "claim": "授权和结算是协议核心机制。",
                                            "explanation": "来源片段同时提到两项机制。",
                                            "source_ids": ["tavily-source-1"],
                                            "confidence": 0.85,
                                        }
                                    ],
                                    "gaps": ["缺少更多官方规范来源"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    report = _report()
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        cache = ResponseCache(tmp_path / "synthesis.sqlite")
        first = synthesize_report(client, cache, report)
        second = synthesize_report(client, cache, report)

    assert len(calls) == 1
    assert first.summary == "协议重点集中在授权和结算。"
    assert first.key_findings[0].source_ids == ["tavily-source-1"]
    assert second.key_findings[0].finding_id == "finding-1"


def test_synthesis_discards_findings_with_unknown_sources(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"summary":"ok","key_findings":[{"finding_id":"bad","claim":"unsupported","source_ids":["missing"],"confidence":0.9}]}'
                        }
                    }
                ]
            },
        )

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        result = synthesize_report(client, None, _report())

    assert result.key_findings == []
    assert "finding_without_sources" in result.gaps


def test_synthesis_skips_model_for_empty_sources(tmp_path):
    def handler(request):
        raise AssertionError("model should not be called")

    report = _report().model_copy(update={"sources": []})
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        result = synthesize_report(client, ResponseCache(tmp_path / "synthesis.sqlite"), report)

    assert result == report


def test_synthesis_preserves_safe_schema_error_code(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"summary": ["wrong"]}'}}]},
        )

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        result = synthesize_report(client, None, _report())

    assert "model_synthesis_fallback:schema_mismatch:summary" in result.gaps
    assert result.summary
