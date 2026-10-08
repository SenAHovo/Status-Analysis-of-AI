"""Report assembly and deterministic PDF delivery nodes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from ai_status_report.documents.pdf_export import export_report_pdf
from ai_status_report.documents.report_assembly import assemble_report
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.schemas.report import ArtifactRef
from ai_status_report.schemas.report_spec import default_report_spec


def assemble_report_node(state: WorkflowState, *, project_root: Path) -> WorkflowState:
    """Assemble exactly the approved four chapters into one report artifact."""

    trace = ControllerRunTrace.from_state(project_root, state)
    try:
        artifacts = {
            section_id: ArtifactRef.model_validate(payload)
            for section_id, payload in state.get("approved_chapter_artifacts", {}).items()
        }
        report_artifact = assemble_report(
            project_root,
            run_id=state["run_id"],
            chapter_artifacts=artifacts,
            spec=default_report_spec(),
            report_title=state.get("report_title"),
        )
    except (TypeError, ValueError):
        trace.emit(
            phase="report_assembly",
            event_type="controller.report.assembly_failed",
            status="failed",
            agent_id="controller",
            refs={"reason": "report_assembly_failed"},
            summary="controller could not assemble verified chapter artifacts",
        )
        return {
            "phase": "report_assembly_blocked",
            "error": "report_assembly_failed",
            "events": [*state.get("events", []), "report.assembly.failed"],
            **trace.state_update(),
        }
    trace.emit(
        phase="report_assembly",
        event_type="controller.report.assembled",
        status="completed",
        agent_id="controller",
        refs={"report_artifact": report_artifact.access_ref},
        summary="controller assembled the approved four-chapter Markdown report",
    )
    return {
        "phase": "report_assembled",
        "report_markdown_artifact": report_artifact.model_dump(mode="json"),
        "events": [*state.get("events", []), "report.assembled"],
        **trace.state_update(),
    }


async def export_report_node(
    state: WorkflowState,
    *,
    project_root: Path,
    exporter: Callable = export_report_pdf,
) -> WorkflowState:
    """Export the assembled report through deterministic local Chromium."""

    trace = ControllerRunTrace.from_state(project_root, state)
    try:
        report_artifact = ArtifactRef.model_validate(state["report_markdown_artifact"])
        pdf_artifact = await asyncio.to_thread(
            exporter,
            project_root,
            report_artifact=report_artifact,
        )
    except (KeyError, TypeError, ValueError) as exc:
        reason = (
            "report_markdown_invalid_encoding"
            if str(exc) == "report_markdown_invalid_encoding"
            else "report_pdf_export_failed"
        )
        trace.emit(
            phase="pdf_export",
            event_type="controller.report.pdf_export_failed",
            status="failed",
            agent_id="controller",
            refs={"reason": reason},
            summary="controller could not export the assembled report PDF",
        )
        return {
            "phase": "report_pdf_export_blocked",
            "error": reason,
            "events": [*state.get("events", []), "report.pdf_export.failed"],
            **trace.state_update(),
        }
    trace.emit(
        phase="pdf_export",
        event_type="controller.report.pdf_exported",
        status="completed",
        agent_id="controller",
        refs={"pdf_artifact": pdf_artifact.access_ref},
        summary="controller exported the deterministic report PDF",
    )
    return {
        "phase": "report_pdf_exported",
        "report_pdf_artifact": pdf_artifact.model_dump(mode="json"),
        "events": [*state.get("events", []), "report.pdf_exported"],
        **trace.state_update(),
    }
