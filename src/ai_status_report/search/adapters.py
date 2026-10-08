"""Adapters from provider responses to the project's search contract."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from ai_status_report.schemas.search import ResearchReport, ResearchSource


def _is_http_url(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlsplit(value.strip())
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _source_id(provider: str, query: str, position: int) -> str:
    query_key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return f"{provider}-{query_key}-{position}"


def _append_source(
    sources: list[ResearchSource],
    source: dict[str, Any],
    raw_ref: str,
    query: str,
    *,
    verification_status: str = "reported",
) -> None:
    url = source.get("url") or source.get("href") or source.get("link")
    if not isinstance(url, str) or not url.strip():
        return
    if not _is_http_url(url):
        return
    if any(existing.url == url for existing in sources):
        return
    sources.append(
        ResearchSource(
            source_id=_source_id("deepseek", query, len(sources) + 1),
            provider="deepseek_native_web_search",
            title=str(source.get("title") or source.get("name") or url),
            url=url,
            snippet=str(source.get("snippet") or source.get("content") or ""),
            published_at=source.get("published_at") or source.get("published_date"),
            retrieved_at=datetime.now(UTC),
            raw_ref=raw_ref,
            verification_status=verification_status,
        )
    )


def tavily_report(query: str, response: dict[str, Any], raw_ref: str = "") -> ResearchReport:
    sources = [
        ResearchSource(
            source_id=_source_id("tavily", query, index),
            provider="tavily_mcp",
            title=str(item.get("title", "")),
            url=str(item.get("url", "")),
            snippet=str(item.get("content", "")),
            published_at=item.get("published_date"),
            retrieved_at=datetime.now(UTC),
            raw_ref=raw_ref,
            verification_status="retrieved" if str(item.get("content", "")).strip() else "reported",
        )
        for index, item in enumerate(response.get("results", []), start=1)
        if isinstance(item, dict) and _is_http_url(item.get("url"))
    ]
    failed = bool(response.get("error"))
    status = "failed" if failed else ("completed" if sources and str(response.get("answer") or "").strip() else "partial")
    incomplete_reason = str(response.get("error")) if failed else None
    if not failed and not sources:
        incomplete_reason = "no_citations"
    elif not failed and not str(response.get("answer") or "").strip():
        incomplete_reason = "empty_summary"
    return ResearchReport(
        query=query,
        provider="tavily_mcp",
        summary=str(response.get("answer") or ""),
        status=status,
        incomplete_reason=incomplete_reason,
        sources=sources,
        raw_ref=raw_ref,
    )


def deepseek_report(query: str, response: dict[str, Any], raw_ref: str = "") -> ResearchReport:
    sources: list[ResearchSource] = []
    final_text_parts: list[str] = []
    invalid_shape = False
    output = response.get("output", [])
    if not isinstance(output, list):
        invalid_shape = True
        output = []
    for item in output:
        if not isinstance(item, dict):
            invalid_shape = True
            continue
        if item.get("type") == "message":
            phase = item.get("phase")
            content_items = item.get("content", [])
            if not isinstance(content_items, list):
                invalid_shape = True
                content_items = []
            for content in content_items:
                if not isinstance(content, dict):
                    invalid_shape = True
                    continue
                if content.get("type") == "output_text" and phase in {
                    None,
                    "final",
                    "answer",
                    "final_answer",
                }:
                    text = str(content.get("text", ""))
                    try:
                        structured = json.loads(text)
                    except (TypeError, ValueError):
                        structured = None
                    if isinstance(structured, dict) and "summary" in structured:
                        final_text_parts.append(str(structured.get("summary") or ""))
                        structured_sources = structured.get("sources", [])
                        if isinstance(structured_sources, list):
                            for source in structured_sources:
                                if isinstance(source, dict):
                                    _append_source(sources, source, raw_ref, query)
                    else:
                        final_text_parts.append(text)
                    annotations = content.get("annotations", [])
                    if isinstance(annotations, list):
                        for annotation in annotations:
                            if isinstance(annotation, dict):
                                _append_source(sources, annotation, raw_ref, query)
        if item.get("type") == "web_search_call":
            item_sources = item.get("sources", [])
            if not isinstance(item_sources, list):
                invalid_shape = True
                item_sources = []
            for source in item_sources:
                if isinstance(source, dict):
                    _append_source(sources, source, raw_ref, query)
                else:
                    invalid_shape = True
            action = item.get("action") or {}
            if not isinstance(action, dict):
                invalid_shape = True
                action = {}
            url = action.get("url")
            clean_url = url.split("#ws_call_id=", 1)[0] if isinstance(url, str) else ""
            parsed_url = urlsplit(clean_url)
            if (
                clean_url
                and parsed_url.scheme in {"http", "https"}
                and parsed_url.netloc
                and not any(source.url == clean_url for source in sources)
            ):
                sources.append(
                    ResearchSource(
                        source_id=_source_id("deepseek", query, len(sources) + 1),
                        provider="deepseek_native_web_search",
                        title=clean_url.split("/")[-1] or clean_url,
                        url=clean_url,
                        snippet="",
                        retrieved_at=datetime.now(UTC),
                        raw_ref=raw_ref,
                        verification_status="reported",
                    )
                )
    response_status = response.get("status")
    status = "completed" if response_status == "completed" and final_text_parts else "partial"
    if response.get("error"):
        status = "failed"
    incomplete_details = response.get("incomplete_details")
    incomplete_reason = (
        incomplete_details.get("reason") if isinstance(incomplete_details, dict) else None
    )
    if incomplete_details is not None and not isinstance(incomplete_details, dict):
        invalid_shape = True
    if status == "completed" and not sources:
        status = "partial"
        if incomplete_reason is None:
            incomplete_reason = "no_citations"
    if invalid_shape and status == "completed":
        status = "partial"
    if invalid_shape and incomplete_reason is None:
        incomplete_reason = "invalid_provider_response"
    return ResearchReport(
        query=query,
        provider="deepseek_native_web_search",
        summary="\n".join(part for part in final_text_parts if part),
        status=status,
        incomplete_reason=incomplete_reason,
        sources=sources,
        raw_ref=raw_ref,
    )
