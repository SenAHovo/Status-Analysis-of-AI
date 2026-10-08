"""Pure LangGraph routing decisions for the controller."""

from __future__ import annotations

from ai_status_report.harness.state import WorkflowState


def route_after_classification(state: WorkflowState) -> str:
    """Continue only for a report request."""

    return "continue" if state.get("intent") == "report_request" else "end"


def route_after_search(state: WorkflowState, *, document_agent_url: str | None) -> str:
    """Index a usable search result when the document Agent is configured."""

    report = state.get("research_report")
    if (
        document_agent_url
        and isinstance(report, dict)
        and report.get("status") in {"completed", "partial"}
    ):
        return "index_evidence"
    return "end"


def route_after_index(state: WorkflowState) -> str:
    return "write_section" if state.get("phase") == "rag_indexed" else "end"


def route_after_document(state: WorkflowState) -> str:
    """Only a usable draft can enter the controller's review stage."""

    return "review" if state.get("phase") == "document_drafted" else "end"


def route_after_revision(state: WorkflowState) -> str:
    return "review" if state.get("phase") == "document_revised" else "end"


def route_after_review(state: WorkflowState, *, whole_report: bool) -> str:
    decision = state.get("review_decision", {})
    if decision.get("decision") == "need_evidence" and state.get("phase") == "evidence_supplement_required":
        return "supplement"
    if decision.get("decision") == "need_revision" and state.get("phase") == "document_revision_required":
        return "revise"
    if decision.get("decision") in {"approve", "deliver_with_limits"}:
        sections = state.get("report_sections", [])
        if whole_report and int(state.get("report_section_index", 0)) < len(sections):
            return "next_section"
        if whole_report:
            return "generate_title"
    return "end"


def route_after_assembly(state: WorkflowState) -> str:
    return "export" if state.get("phase") == "report_assembled" else "end"
