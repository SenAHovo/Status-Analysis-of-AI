"""Controller review node and bounded review decision handling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.graph_helpers import (
    approved_section_context,
    review_document_task,
)
from ai_status_report.harness.review_audit import persist_review_audit
from ai_status_report.harness.review_policy import enforce_hard_gap_policy
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.schemas.document import (
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.review import ReviewDecision


def review_section_node(
    state: WorkflowState,
    *,
    project_root: Path,
    reviewer: Callable[
        [DocumentWriteSectionTask | DocumentReviseSectionTask, DocumentSectionResult, int],
        ReviewDecision,
    ],
    whole_report: bool,
) -> WorkflowState:
    """Review one delivered section and select approval or the next bounded action."""

    trace = ControllerRunTrace.from_state(project_root, state)
    result = DocumentSectionResult.model_validate(state["document_result"])
    task = review_document_task(state)
    trace.emit(
        phase="review",
        event_type="controller.review.started",
        status="started",
        agent_id="controller",
        refs={"section_id": result.section_id, "draft_version": result.draft_version},
        summary="controller started the chapter evidence review",
    )
    supplement_round = int(state.get("supplement_round", 0))
    decision = reviewer(task, result, supplement_round)
    decision, supplement_plan = enforce_hard_gap_policy(
        decision=decision,
        result=result,
        section_title=task.section_title,
        supplement_round=supplement_round,
    )
    evidence_limit_reached = decision.decision == "need_evidence" and supplement_round >= 2
    revision_limit_reached = (
        not evidence_limit_reached
        and decision.decision == "need_revision"
        and int(state.get("revision_round", 0)) >= 2
    )
    if revision_limit_reached:
        decision = decision.model_copy(update={"reason": "修订轮次已达到上限，当前章节未获批准"})
    if (
        decision.run_id != result.run_id
        or decision.section_id != result.section_id
        or decision.draft_version != result.draft_version
    ):
        raise RuntimeError("review_decision_mismatch")
    audit_ref = persist_review_audit(
        project_root,
        result=result,
        decision=decision,
        plan=supplement_plan,
    )
    trace.emit(
        phase="review",
        event_type="controller.review.decided",
        status="completed",
        agent_id="controller",
        refs={
            "section_id": decision.section_id,
            "draft_version": decision.draft_version,
            "decision": decision.decision,
            "issue_count": str(len(decision.issues)),
            "review_round": str(decision.review_round),
            "source": decision.source,
            "fallback_reason": decision.fallback_reason,
            "review_audit_ref": str(audit_ref),
        },
        summary="controller completed the chapter review decision",
    )
    approval_update: dict[str, object] = {}
    if decision.decision in {"approve", "deliver_with_limits"} and whole_report:
        artifact, summary = approved_section_context(project_root, result=result)
        approved_artifacts = dict(state.get("approved_chapter_artifacts", {}))
        if result.section_id in approved_artifacts:
            raise RuntimeError("approved_chapter_duplicate")
        approved_artifacts[result.section_id] = artifact.model_dump(mode="json")
        approval_update = {
            "approved_chapter_artifacts": approved_artifacts,
            "approved_section_summaries": [
                *state.get("approved_section_summaries", []),
                summary,
            ],
            "report_section_index": int(state.get("report_section_index", 0)) + 1,
        }
    return {
        "phase": (
            "evidence_limit_reached"
            if evidence_limit_reached
            else "review_limit_reached"
            if revision_limit_reached
            else {
                "approve": "document_approved",
                "deliver_with_limits": "document_approved_with_limits",
                "need_evidence": "evidence_supplement_required",
                "need_revision": "document_revision_required",
            }[decision.decision]
        ),
        "review_decision": decision.model_dump(mode="json"),
        "review_issues": [issue.model_dump(mode="json") for issue in decision.issues],
        "review_audit_ref": str(audit_ref),
        **approval_update,
        "supplement_plan": supplement_plan.model_dump(mode="json") if supplement_plan else {},
        "events": [
            *state.get("events", []),
            "review.started",
            f"review.{decision.source}",
            (
                "review.evidence_limit_reached"
                if evidence_limit_reached
                else "review.limit_reached"
                if revision_limit_reached
                else f"review.{decision.decision}"
            ),
        ],
        **trace.state_update(),
    }
