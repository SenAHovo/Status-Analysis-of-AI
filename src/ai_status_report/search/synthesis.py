"""Source-grounded Search Agent report synthesis."""

from __future__ import annotations

import hashlib
import json

from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.router import ResponseCache
from ai_status_report.model.structured import StructuredOutputError, parse_structured, schema_hint
from ai_status_report.schemas.search import ResearchReport, ResearchSynthesis
from ai_status_report.search.report_validation import bound_gaps, validate_findings

MAX_SYNTHESIS_SOURCES = 20
MAX_SNIPPET_CHARS = 1200

SYNTHESIS_SYSTEM_PROMPT = """你是网络搜索 Agent 的研究汇报整理器。
你的任务是把已经检索到的来源整理成供主控 Agent 和文档 Agent 使用的 ResearchReport。
只使用用户消息中提供的来源和片段，不补写未出现的事实，不把模型常识当作搜索结论。
每条 key_finding 必须至少引用一个输入 source_id；没有来源支持的判断放入 gaps。
source_id 必须逐字使用输入列表中的值，不能创建、改写或猜测 source_id。
区分来源状态：retrieved 表示已有工具返回的正文片段，reported 表示 Provider 仅声明了来源。
summary 说明主题、主要趋势和证据边界；key_findings 只保留最重要的可追溯结论。
key_findings 中每项必须同时包含 finding_id、claim、explanation、source_ids、confidence；
source_ids 必须是非空数组，confidence 必须是 0 到 1 之间的小数。
没有足够证据时，key_findings 返回空数组，并把原因写入 gaps。
严格返回如下形状的 JSON：
{"summary":"简短汇报","key_findings":[{"finding_id":"finding-1","claim":"可核验结论","explanation":"依据来源的说明","source_ids":["输入中的source_id"],"confidence":0.8}],"gaps":[]}
输出 JSON，不输出 Markdown、解释文字或代码围栏。
"""


def _cache_key(model: str, messages: list[dict], options: dict) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "options": options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_context(report: ResearchReport) -> list[dict[str, str]]:
    return [
        {
            "source_id": source.source_id,
            "provider": source.provider,
            "title": source.title,
            "url": source.url,
            "snippet": source.snippet[:MAX_SNIPPET_CHARS],
            "verification_status": source.verification_status,
        }
        for source in report.sources[:MAX_SYNTHESIS_SOURCES]
    ]


def _fallback(report: ResearchReport, code: str) -> ResearchReport:
    summary = report.summary
    if not summary:
        summary = f"围绕“{report.query}”检索到 {len(report.sources)} 个来源；结构化汇报未完成，需依据 evidence_refs 继续核验。"
    return report.model_copy(
        update={
            "summary": summary[:12000],
            "gaps": bound_gaps([*report.gaps, f"model_synthesis_fallback:{code}"]),
        }
    )


def synthesize_report(
    client: DeepSeekClient | None,
    cache: ResponseCache | None,
    report: ResearchReport,
    *,
    max_output_tokens: int = 4096,
) -> ResearchReport:
    """Generate and validate a compact source-grounded research handoff."""

    if client is None or report.status == "failed" or not report.sources:
        return report

    input_payload = {
        "query": report.query,
        "provider": report.provider,
        "existing_summary": report.summary[:4000],
        "sources": _source_context(report),
    }
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT + schema_hint(ResearchSynthesis)},
        {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False)},
    ]
    options = {
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "max_tokens": max_output_tokens,
    }
    key = _cache_key(client.profile.model, messages, options)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            try:
                synthesis = ResearchSynthesis.model_validate(cached)
                return _apply_synthesis(report, synthesis)
            except (TypeError, ValueError):
                return _fallback(report, "cache_schema_mismatch")

    try:
        response = client.chat(messages, **options)
        content = response["choices"][0]["message"]["content"]
        synthesis = parse_structured(content, ResearchSynthesis)
    except StructuredOutputError as exc:
        return _fallback(report, str(exc) or "schema_mismatch")
    except ProviderError:
        return _fallback(report, "provider_error")
    except (KeyError, IndexError, TypeError, ValueError):
        return _fallback(report, "invalid_model_response")

    if cache is not None:
        cache.put(key, synthesis.model_dump(mode="json"))
    return _apply_synthesis(report, synthesis)


def _apply_synthesis(report: ResearchReport, synthesis: ResearchSynthesis) -> ResearchReport:
    findings, warnings = validate_findings(
        synthesis.key_findings,
        {source.source_id for source in report.sources},
    )
    return report.model_copy(
        update={
            "summary": synthesis.summary or report.summary,
            "key_findings": findings,
            "gaps": bound_gaps([*report.gaps, *synthesis.gaps, *warnings]),
        }
    )
