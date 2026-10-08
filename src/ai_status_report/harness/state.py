"""Serializable state for the first LangGraph controller."""

from __future__ import annotations

from typing import TypedDict


class WorkflowState(TypedDict, total=False):
    run_id: str
    user_input: str
    research_brief: dict
    intent: str
    phase: str
    events: list[str]
    a2a_tasks: list[str]
    a2a_results: list[str]
    a2a_trace: list[dict[str, object]]
    trace_id: str
    trace_path: str
    a2a_trace_path: str
    execution_trace: list[dict[str, object]]
    research_report: dict
    research_sources: list[dict]
    document_task: dict
    document_result: dict
    review_decision: dict
    review_issues: list[dict]
    review_audit_ref: str
    supplement_plan: dict
    supplement_reports: list[dict]
    supplement_audit_ref: str
    evidence_delta: dict[str, list[str]]
    supplement_round: int
    revision_round: int
    rag_index: dict[str, object]
    mcp_calls: list[str]
    error: str
    report_spec: dict
    report_section_index: int
    report_title: str
    report_title_source: str
    report_sections: list[dict]
    approved_section_summaries: list[dict]
    approved_chapter_artifacts: dict[str, dict]
    report_markdown_artifact: dict
    report_pdf_artifact: dict
