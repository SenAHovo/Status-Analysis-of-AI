import json

import httpx
import pytest

from ai_status_report.model.client import DeepSeekClient
from ai_status_report.model.router import ResponseCache
from ai_status_report.schemas.search import ResearchFinding, ResearchReport, ResearchSource
from ai_status_report.search.merge import merge_reports
from ai_status_report.search.plan_validator import SearchPlanError, validate_plan
from ai_status_report.search.planner import SearchPlanner, deterministic_plan
from ai_status_report.search.report_validation import bound_gaps
from ai_status_report.settings import Profile

PROFILE = Profile("deepseek-v4-flash", "https://api.deepseek.com", "test-secret")


def test_deterministic_plan_preserves_existing_task_fields():
    plan = deterministic_plan(
        {
            "query": "人工智能支付协议现状",
            "source_preference": "both",
            "critical": True,
            "max_results": 7,
            "max_output_tokens": 32768,
        }
    )
    assert plan.providers == ["tavily", "deepseek"]
    assert plan.limits.max_results_per_query == 7
    assert plan.limits.max_output_tokens == 32768
    assert plan.need_cross_validation is True


def test_model_search_plan_is_structured_and_cached(tmp_path):
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
                                    "provider": "tavily",
                                    "reason": "official protocol",
                                }
                            )
                        }
                    }
                ]
            },
        )

    task = {"query": "人工智能支付协议现状", "source_preference": "auto"}
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        planner = SearchPlanner(client, ResponseCache(tmp_path / "plan.sqlite"))
        first = planner.plan(task)
        second = planner.plan(task)
    assert first.source == "model"
    assert second.source == "model_cache"
    assert first.original_query == task["query"]
    assert first.providers == ["tavily"]
    assert len(calls) == 1


def test_retrievable_evidence_requirement_enters_model_cache_key(tmp_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"provider":"deepseek","reason":"fresh"}'}}]},
        )

    plain_task = {"query": "当前电力行业人工智能应用", "source_preference": "auto"}
    retrieval_task = plain_task | {"evidence_requirement": "retrievable"}
    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        planner = SearchPlanner(client, ResponseCache(tmp_path / "plan.sqlite"))
        plain = planner.plan(plain_task)
        retrieval = planner.plan(retrieval_task)
    assert plain.providers == ["deepseek"]
    assert retrieval.providers == ["deepseek"]
    assert retrieval.evidence_requirement == "retrievable"
    assert len(calls) == 2


def test_plan_validator_requires_tavily_for_retrievable_evidence():
    plan = deterministic_plan({"query": "q", "evidence_requirement": "retrievable"})
    validated = validate_plan(
        plan,
        {"query": "q", "source_preference": "auto", "evidence_requirement": "retrievable"},
        available_providers={"tavily", "deepseek"},
        configured_budget=16384,
    )
    assert validated.providers == ["tavily"]
    assert validated.evidence_requirement == "retrievable"

    with pytest.raises(SearchPlanError, match="retrievable_evidence_provider_not_configured"):
        validate_plan(
            plan,
            {"query": "q", "evidence_requirement": "retrievable"},
            available_providers={"deepseek"},
            configured_budget=16384,
        )


def test_model_plan_fallback_exposes_safe_reason_code(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"provider":"unknown"}'}}]},
        )

    with DeepSeekClient(PROFILE, 5, transport=httpx.MockTransport(handler)) as client:
        plan = SearchPlanner(client, ResponseCache(tmp_path / "plan.sqlite")).plan(
            {"query": "q", "source_preference": "auto"}
        )
    assert plan.source == "deterministic_fallback"
    assert plan.validation_warnings == ["model_plan_fallback:schema_mismatch:provider"]


def test_plan_validator_enforces_explicit_provider_and_available_budget():
    plan = deterministic_plan({"query": "q", "source_preference": "auto"})
    validated = validate_plan(
        plan,
        {"query": "q", "source_preference": "auto", "max_results": 20},
        available_providers={"deepseek"},
        configured_budget=16384,
    )
    assert validated.providers == ["deepseek"]
    assert validated.limits.max_results_per_query == 5
    assert validated.limits.max_output_tokens == 16384
    assert "provider_reduced_to_available" in validated.validation_warnings

    with pytest.raises(SearchPlanError, match="requested_provider_not_configured"):
        validate_plan(
            plan,
            {"query": "q", "source_preference": "tavily"},
            available_providers={"deepseek"},
            configured_budget=16384,
        )


def test_merge_reports_deduplicates_urls_and_retains_refs():
    source_one = ResearchSource(
        source_id="t1",
        provider="tavily_mcp",
        title="Same",
        url="https://example.com/a/?utm_source=test",
        snippet="content",
        retrieved_at="2026-09-10T00:00:00Z",
    )
    source_two = source_one.model_copy(
        update={"source_id": "d1", "provider": "deepseek_native_web_search"}
    )
    plan = deterministic_plan({"query": "q", "source_preference": "both"})
    report = merge_reports(
        [
            ResearchReport(
                query="q", provider="tavily_mcp", summary="tavily", sources=[source_one], raw_ref="raw-t"
            ),
            ResearchReport(
                query="q", provider="deepseek_native_web_search", summary="deepseek", sources=[source_two], raw_ref="raw-d"
            ),
        ],
        plan,
        raw_refs=["raw-t", "raw-d"],
    )
    assert report.provider == "multi_provider"
    assert report.providers == ["tavily_mcp", "deepseek_native_web_search"]
    assert len(report.sources) == 1
    assert report.raw_refs == ["raw-t", "raw-d"]


def test_merge_reports_keeps_only_source_grounded_findings():
    source = ResearchSource(
        source_id="source-1",
        provider="tavily_mcp",
        url="https://example.com/source",
        snippet="content",
        retrieved_at="2026-09-10T00:00:00Z",
    )
    findings = [
        ResearchFinding(
            finding_id="f-1",
            claim="有来源支持的结论",
            source_ids=["source-1"],
            confidence=0.8,
        ),
        ResearchFinding(
            finding_id="f-2",
            claim="没有来源的结论",
            source_ids=["missing"],
            confidence=0.9,
        ),
    ]
    plan = deterministic_plan({"query": "q", "source_preference": "tavily"})
    report = merge_reports(
        [ResearchReport(query="q", provider="tavily_mcp", sources=[source], key_findings=findings)],
        plan,
        raw_refs=[],
    )

    assert [finding.finding_id for finding in report.key_findings] == ["f-1"]
    assert "finding_without_sources" in report.gaps


def test_merge_reports_remaps_findings_after_url_deduplication():
    first = ResearchSource(
        source_id="source-1",
        provider="tavily_mcp",
        url="https://example.com/source",
        snippet="content",
        retrieved_at="2026-09-10T00:00:00Z",
    )
    duplicate = first.model_copy(update={"source_id": "source-2", "provider": "deepseek_native_web_search"})
    finding = ResearchFinding(
        finding_id="f-1",
        claim="重复来源仍可追溯",
        source_ids=["source-2"],
        confidence=0.7,
    )
    plan = deterministic_plan({"query": "q", "source_preference": "both"})
    report = merge_reports(
        [
            ResearchReport(query="q", provider="tavily_mcp", sources=[first]),
            ResearchReport(
                query="q",
                provider="deepseek_native_web_search",
                sources=[duplicate],
                key_findings=[finding],
            ),
        ],
        plan,
        raw_refs=[],
    )

    assert report.key_findings[0].source_ids == ["source-1"]


def test_bound_gaps_preserves_priority_and_contract_limit():
    gaps = [f"plan-warning-{index}" for index in range(20)]
    gaps.extend(["provider_partial", "finding_without_sources", "provider_partial"])

    bounded = bound_gaps(gaps)

    assert len(bounded) == 20
    assert bounded[:2] == ["provider_partial", "finding_without_sources"]


def test_merge_reports_compacts_sources_and_preserves_incomplete_reason():
    source = ResearchSource(
        source_id="source-long",
        provider="tavily_mcp",
        url="https://example.com/long",
        snippet="x" * 10000,
        retrieved_at="2026-09-10T00:00:00Z",
    )
    plan = deterministic_plan({"query": "q", "source_preference": "tavily"})
    report = merge_reports(
        [
            ResearchReport(
                query="q",
                provider="tavily_mcp",
                status="partial",
                incomplete_reason="empty_summary",
                sources=[source],
            )
        ],
        plan,
        raw_refs=["raw.json"],
    )

    assert len(report.sources[0].snippet) == 1200
    assert report.incomplete_reason == "empty_summary"


def test_merge_reports_marks_mixed_success_and_rate_limit_as_partial():
    plan = deterministic_plan({"query": "q", "source_preference": "tavily"})
    report = merge_reports(
        [
            ResearchReport(query="q", provider="tavily_mcp", status="completed"),
            ResearchReport(
                query="q",
                provider="tavily_mcp",
                status="failed",
                incomplete_reason="tavily_rate_limited",
            ),
        ],
        plan,
        raw_refs=[],
    )

    assert report.status == "partial"
    assert report.incomplete_reason == "tavily_rate_limited"
    assert "provider_failed" in report.gaps


def test_merge_reports_bounds_evidence_and_raw_refs():
    plan = deterministic_plan({"query": "q", "source_preference": "both"})
    first = ResearchReport(
        query="q",
        provider="tavily_mcp",
        evidence_refs=[f"chunk-{index}" for index in range(60)],
    )
    second = ResearchReport(
        query="q",
        provider="deepseek_native_web_search",
        evidence_refs=[f"chunk-{index}" for index in range(60, 105)],
    )

    report = merge_reports(
        [first, second],
        plan,
        raw_refs=[f"raw-{index}" for index in range(12)],
    )

    assert len(report.evidence_refs) == 100
    assert report.evidence_refs == [f"chunk-{index}" for index in range(100)]
    assert len(report.raw_refs) == 10
