"""Validation helpers for source-grounded ResearchReport findings."""

from __future__ import annotations

from ai_status_report.schemas.search import ResearchFinding

MAX_FINDINGS = 12
MAX_GAPS = 20


def bound_gaps(gaps: list[str]) -> list[str]:
    """Deduplicate gaps and keep the report within its public contract."""

    unique = list(dict.fromkeys(gaps))
    priority = {
        "provider_failed",
        "provider_partial",
        "no_citations",
        "finding_without_sources",
    }
    ordered = [gap for gap in unique if gap in priority]
    ordered.extend(gap for gap in unique if gap not in priority)
    return ordered[:MAX_GAPS]


def validate_findings(
    findings: list[ResearchFinding],
    source_ids: set[str],
) -> tuple[list[ResearchFinding], list[str]]:
    """Keep only bounded findings with at least one report source."""

    valid: list[ResearchFinding] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for finding in findings:
        if finding.finding_id in seen_ids:
            warnings.append("duplicate_finding_id")
            continue
        seen_ids.add(finding.finding_id)
        if len(valid) >= MAX_FINDINGS:
            warnings.append("finding_limit_exceeded")
            continue
        linked_sources = list(dict.fromkeys(source_id for source_id in finding.source_ids if source_id in source_ids))
        if not linked_sources:
            warnings.append("finding_without_sources")
            continue
        if len(linked_sources) != len(finding.source_ids):
            warnings.append("finding_source_reduced")
        valid.append(finding.model_copy(update={"source_ids": linked_sources}))
    return valid, list(dict.fromkeys(warnings))
