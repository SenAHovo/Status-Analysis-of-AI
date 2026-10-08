"""Merge provider reports into the compact A2A research handoff."""

from __future__ import annotations

import re

from ai_status_report.schemas.search import (
    ResearchFinding,
    ResearchReport,
    ResearchSource,
    SearchPlan,
)
from ai_status_report.search.report_validation import bound_gaps, validate_findings
from ai_status_report.storage.search_results import canonicalize_url

REPORT_SOURCE_SNIPPET_CHARS = 1200
_SAFE_REASON = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")


def _safe_incomplete_reason(reason: str | None) -> str | None:
    if not reason:
        return None
    return reason if _SAFE_REASON.fullmatch(reason) else "provider_error"


def merge_reports(
    reports: list[ResearchReport],
    plan: SearchPlan,
    *,
    raw_refs: list[str],
) -> ResearchReport:
    sources: list[ResearchSource] = []
    seen_urls: set[str] = set()
    summaries: list[str] = []
    gaps: list[str] = list(plan.validation_warnings)
    findings: list[ResearchFinding] = []
    incomplete_reasons: list[str] = []
    source_aliases: dict[str, str] = {}
    for report in reports:
        safe_reason = _safe_incomplete_reason(report.incomplete_reason)
        if safe_reason:
            incomplete_reasons.append(safe_reason)
        if report.summary:
            summaries.append(f"[{report.provider}] {report.summary}")
        for source in report.sources:
            canonical = canonicalize_url(source.url) if source.url else ""
            if canonical and canonical in seen_urls:
                existing = next(item for item in sources if canonicalize_url(item.url) == canonical)
                source_aliases[source.source_id] = existing.source_id
                continue
            if canonical:
                seen_urls.add(canonical)
            sources.append(source.model_copy(update={"snippet": source.snippet[:REPORT_SOURCE_SNIPPET_CHARS]}))
            source_aliases[source.source_id] = source.source_id
        findings.extend(report.key_findings)
    mapped_findings = [
        finding.model_copy(
            update={
                "source_ids": [source_aliases.get(source_id, source_id) for source_id in finding.source_ids]
            }
        )
        for finding in findings
    ]
    validated_findings, finding_warnings = validate_findings(
        mapped_findings,
        {source.source_id for source in sources},
    )
    gaps.extend(finding_warnings)
    if any(report.status == "failed" for report in reports):
        gaps.append("provider_failed")
    if any(report.status == "partial" for report in reports):
        gaps.append("provider_partial")
    if not reports or all(report.status == "failed" for report in reports):
        status = "failed"
    elif any(report.status in {"partial", "failed"} for report in reports):
        status = "partial"
    else:
        status = "completed"
    providers = list(dict.fromkeys(report.provider for report in reports))
    return ResearchReport(
        query=plan.original_query,
        provider=providers[0] if len(providers) == 1 else "multi_provider",
        providers=providers,
        plan_id=plan.plan_id,
        plan_source=plan.source,
        summary="\n\n".join(summaries)[:12000],
        key_findings=validated_findings,
        status=status,
        sources=sources[:50],
        evidence_refs=list(dict.fromkeys(ref for report in reports for ref in report.evidence_refs))[:100],
        raw_ref=raw_refs[0] if len(raw_refs) == 1 else "",
        raw_refs=raw_refs[:10],
        incomplete_reason=(incomplete_reasons[0] if incomplete_reasons else None),
        gaps=bound_gaps(gaps),
    )
