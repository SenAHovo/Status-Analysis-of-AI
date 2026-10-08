"""Document Agent revision dispatch node."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.graph_helpers import (
    artifact_texts,
    document_revise_task,
    validate_document_result,
    verified_chapter_artifact_hash,
)
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.schemas.document import DocumentSectionResult
from ai_status_report.schemas.review import ReviewDecision


async def revise_section_node(
    state: WorkflowState,
    *,
    project_root: Path,
    document_agent_url: str | None,
    sender: Callable,
) -> WorkflowState:
    """Send the bounded revision task to 8002 and retain the new Artifact."""

    trace = ControllerRunTrace.from_state(project_root, state)
    decision = ReviewDecision.model_validate(state["review_decision"])
    if decision.decision == "need_evidence":
        delta = state.get("evidence_delta")
        if not isinstance(delta, dict) or not delta.get("new_chunk_ids"):
            trace.emit(
                phase="document_revision_dispatch",
                event_type="controller.document.revision.skipped",
                status="failed",
                agent_id="controller",
                refs={"reason": "supplement_no_new_evidence"},
                summary="controller skipped a supplement revision without new EvidenceChunks",
            )
            return {
                "phase": "supplement_no_new_evidence",
                "events": [*state.get("events", []), "supplement.no_new_evidence"],
                **trace.state_update(),
            }
    task_model = document_revise_task(state)
    task = json.dumps(task_model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
    trace.emit(
        phase="document_revision_dispatch",
        event_type="controller.document.revision_prepared",
        status="completed",
        agent_id="controller",
        refs={
            "section_id": task_model.section_id,
            "base_version": task_model.base_draft.version,
            "target_version": task_model.target_draft_version,
        },
        summary="controller prepared a document.revise_section task",
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
        raise RuntimeError("a2a_document_revision_artifact_missing")
    try:
        result = DocumentSectionResult.model_validate_json(result_text)
    except ValueError as exc:
        raise RuntimeError("a2a_document_revision_artifact_invalid") from exc
    validate_document_result(result, task_model)
    revised = result.status == "revised"
    current_hash = ""
    base_hash = ""
    if revised and result.draft_artifact is not None:
        current_hash = verified_chapter_artifact_hash(
            project_root,
            artifact=result.draft_artifact,
            section_id=result.section_id,
        )
        base_hash = verified_chapter_artifact_hash(
            project_root,
            artifact=task_model.base_draft,
            section_id=task_model.section_id,
        )
    no_material_change = revised and current_hash == base_hash
    trace.emit(
        phase="document_revision_dispatch",
        event_type="controller.document.revision_artifact_received",
        status="failed" if no_material_change else "completed",
        agent_id="controller",
        refs={
            "section_id": result.section_id,
            "result_status": result.status,
            "draft_artifact": result.draft_artifact.access_ref if result.draft_artifact else "",
            "reason": "revision_no_material_change" if no_material_change else "",
        },
        summary="controller accepted the Document Agent revision Artifact",
    )
    return {
        "phase": (
            "document_revision_no_material_change"
            if no_material_change
            else "document_revised"
            if revised
            else "document_revision_blocked"
        ),
        "revision_round": int(state.get("revision_round", 0)) + 1,
        "a2a_tasks": [*state.get("a2a_tasks", []), task],
        "a2a_results": [*state.get("a2a_results", []), *artifacts],
        "document_task": task_model.model_dump(mode="json"),
        "document_result": result.model_dump(mode="json"),
        "events": [
            *state.get("events", []),
            "document.revision.dispatched",
            "document.revision.artifact_received",
            (
                "document.revision.no_material_change"
                if no_material_change
                else "document.revised"
                if revised
                else "document.revision.blocked"
            ),
        ],
        **trace.state_update(),
    }
