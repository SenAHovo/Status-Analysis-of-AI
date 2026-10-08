"""Document Agent writing dispatch node."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.graph_helpers import (
    artifact_texts,
    document_write_task,
    validate_document_result,
    verified_chapter_artifact_hash,
)
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.schemas.document import DocumentSectionResult
from ai_status_report.schemas.report_spec import ReportSectionSpec, default_report_spec


async def write_section_node(
    state: WorkflowState,
    *,
    project_root: Path,
    document_agent_url: str | None,
    whole_report: bool,
    sender: Callable,
) -> WorkflowState:
    trace = ControllerRunTrace.from_state(project_root, state)
    report = state.get("research_report")
    if not isinstance(report, dict):
        raise TypeError("research_report_missing_for_document_dispatch")
    if whole_report:
        sections = state.get("report_sections") or [
            item.model_dump(mode="json") for item in default_report_spec().sections
        ]
        section = ReportSectionSpec.model_validate(
            sections[int(state.get("report_section_index", 0))]
        )
        approved = state.get("approved_chapter_artifacts", {})
        missing_dependencies = [
            dependency for dependency in section.depends_on if dependency not in approved
        ]
        if missing_dependencies:
            raise RuntimeError("report_section_dependencies_unapproved")
    else:
        section = ReportSectionSpec(
            section_id="current-status",
            title="当前发展现状",
            purpose="说明当前发展现状、已验证能力和限制。",
            core_questions=[state["user_input"]],
            evidence_boundary="关键事实和重要判断必须有可追溯证据。",
            output_requirements=["使用客观、可追溯的中文行业分析表达"],
        )
    task_model = document_write_task(
        root=project_root,
        run_id=state["run_id"],
        user_input=state["user_input"],
        report_payload=report,
        section_spec=section,
        recent_section_summaries=list(state.get("approved_section_summaries", [])),
        research_brief=state.get("research_brief"),
    )
    task = json.dumps(task_model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    trace.emit(
        phase="document_dispatch",
        event_type="controller.document.task_prepared",
        status="completed",
        agent_id="controller",
        refs={"section_id": task_model.section_id, "task_type": task_model.task_type},
        summary="controller prepared a document.write_section task",
    )
    trace.outbound(
        agent="document_agent",
        url=document_agent_url or "",
        task=task,
        task_type=task_model.task_type,
        section_id=task_model.section_id,
    )
    artifacts: list[str] = []
    async for response in sender(document_agent_url or "", task):
        trace.inbound(agent="document_agent", response=response)
        artifacts.extend(artifact_texts(response))
    result_text = next((text for text in reversed(artifacts) if text.strip()), "")
    if not result_text:
        raise RuntimeError("a2a_document_artifact_missing")
    try:
        result = DocumentSectionResult.model_validate_json(result_text)
    except ValueError as exc:
        raise RuntimeError("a2a_document_artifact_invalid") from exc
    validate_document_result(result, task_model)
    completed = result.status in {"drafted", "revised"}
    trace.emit(
        phase="document_dispatch",
        event_type="controller.document.artifact_received",
        status="completed",
        agent_id="controller",
        refs={"section_id": result.section_id, "result_status": result.status},
        summary="controller received the Document Agent Artifact",
    )
    if completed:
        if result.draft_artifact is None:
            raise RuntimeError("a2a_document_artifact_invalid")
        verified_chapter_artifact_hash(
            project_root, artifact=result.draft_artifact, section_id=result.section_id
        )
        trace.emit(
            phase="document_dispatch",
            event_type="controller.document.artifact_validated",
            status="completed",
            agent_id="controller",
            refs={"section_id": result.section_id, "draft_version": result.draft_version},
            summary="controller validated the Document Agent Artifact",
        )
    return {
        "phase": "document_drafted" if completed else "document_blocked",
        "a2a_tasks": [*state.get("a2a_tasks", []), task],
        "a2a_results": [*state.get("a2a_results", []), *artifacts],
        "document_task": task_model.model_dump(mode="json"),
        "document_result": result.model_dump(mode="json"),
        "events": [
            *state.get("events", []),
            "document.task.dispatched",
            "document.artifact.received",
            "document.drafted" if completed else "document.blocked",
        ],
        **trace.state_update(),
    }
