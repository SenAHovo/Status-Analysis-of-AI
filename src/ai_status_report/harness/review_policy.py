"""Deterministic quality gates around the model's chapter-review decision."""

from __future__ import annotations

from ai_status_report.schemas.document import DocumentSectionResult
from ai_status_report.schemas.review import ReviewDecision, SupplementPlan, SupplementQuery


def _query_for_gap(*, section_title: str, description: str) -> str:
    """Create a bounded fallback query when a hard gap has no model query."""

    return f"{section_title} {description} 官方 数据 研究"[:1000]


def build_supplement_plan(
    *,
    decision: ReviewDecision,
    result: DocumentSectionResult,
    section_title: str,
    supplement_round: int,
) -> SupplementPlan | None:
    """Turn every unresolved hard EvidenceGap into a deduplicated search plan."""

    if supplement_round < 1 or supplement_round > 2:
        return None
    items: list[SupplementQuery] = []
    model_queries = list(decision.supplement_queries)
    for gap in result.evidence_gaps:
        if (
            gap.severity != "hard"
            or gap.resolution_status != "open"
            or gap.kind == "evidence_limitation"
        ):
            continue
        candidates = list(gap.suggested_queries)
        if not candidates and model_queries:
            candidates.append(model_queries.pop(0))
        if not candidates:
            candidates.append(_query_for_gap(section_title=section_title, description=gap.description))
        for query in candidates:
            if any(item.query == query for item in items):
                continue
            items.append(
                SupplementQuery(
                    gap_id=gap.gap_id,
                    query=query,
                    rationale=gap.description,
                    preferred_source_types=gap.preferred_source_types,
                )
            )
            if len(items) == 5:
                break
        if len(items) == 5:
            break
    if not items:
        return None
    source = "model" if all(item.query in decision.supplement_queries for item in items) else "mixed"
    if not any(item.query in decision.supplement_queries for item in items):
        source = "deterministic"
    return SupplementPlan(
        run_id=result.run_id,
        section_id=result.section_id,
        draft_version=result.draft_version,
        supplement_round=supplement_round,
        queries=items,
        source=source,
    )


def _plan_from_review_queries(
    *, decision: ReviewDecision, result: DocumentSectionResult, supplement_round: int
) -> SupplementPlan | None:
    """Keep a valid model ``need_evidence`` decision executable without a document gap."""

    if not decision.supplement_queries or supplement_round < 1 or supplement_round > 2:
        return None
    queries = []
    for index, query in enumerate(decision.supplement_queries[:5]):
        issue = decision.issues[min(index, len(decision.issues) - 1)] if decision.issues else None
        queries.append(
            SupplementQuery(
                gap_id=issue.issue_id if issue else f"review-query-{index + 1}",
                query=query,
                rationale=issue.requested_change if issue else decision.reason or "审核要求补充证据",
            )
        )
    return SupplementPlan(
        run_id=result.run_id,
        section_id=result.section_id,
        draft_version=result.draft_version,
        supplement_round=supplement_round,
        queries=queries,
        source="model" if decision.source in {"model", "model_cache"} else "deterministic",
    )


def enforce_hard_gap_policy(
    *,
    decision: ReviewDecision,
    result: DocumentSectionResult,
    section_title: str,
    supplement_round: int,
) -> tuple[ReviewDecision, SupplementPlan | None]:
    """Block only core evidence gaps; allow limited delivery for soft gaps."""

    blocking_gaps = [
        gap
        for gap in result.evidence_gaps
        if gap.severity == "hard"
        and gap.resolution_status == "open"
        and gap.kind != "evidence_limitation"
    ]
    if not blocking_gaps:
        open_gaps = [
            gap
            for gap in result.evidence_gaps
            if gap.resolution_status == "open"
        ]
        if decision.decision == "approve":
            if open_gaps:
                return (
                    decision.model_copy(
                        update={
                            "decision": "deliver_with_limits",
                            "reason": "仅存在非阻断性证据局限，允许带限制交付",
                        }
                    ),
                    None,
                )
            return decision, None
        if decision.decision == "deliver_with_limits":
            return decision, None
        return (
            decision,
            _plan_from_review_queries(
                decision=decision, result=result, supplement_round=supplement_round + 1
            )
            if decision.decision == "need_evidence"
            else None,
        )
    if supplement_round >= 2:
        return (
            decision.model_copy(
                update={
                    "decision": "need_evidence",
                    "reason": "未解决的硬证据缺口已达到两轮补证上限",
                    "supplement_queries": [],
                }
            ),
            None,
        )
    plan = build_supplement_plan(
        decision=decision,
        result=result,
        section_title=section_title,
        supplement_round=supplement_round + 1,
    )
    if plan is None:
        return decision, None
    existing = {issue.issue_id: issue for issue in decision.issues}
    fallback = ReviewDecision.from_document_result(
        run_id=result.run_id,
        section_id=result.section_id,
        draft_version=result.draft_version,
        evidence_gaps=[gap.model_dump(mode="json") for gap in blocking_gaps],
        review_round=supplement_round,
    )
    for issue in fallback.issues:
        existing.setdefault(issue.issue_id, issue)
    return (
        decision.model_copy(
            update={
                "decision": "need_evidence",
                "issues": list(existing.values()),
                "supplement_queries": [item.query for item in plan.queries],
                "reason": "存在未解决的硬证据缺口，主控已生成受限补证计划",
            }
        ),
        plan,
    )
