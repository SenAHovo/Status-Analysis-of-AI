"""Classification and Search Agent dispatch nodes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.graph_helpers import artifact_texts
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.schemas.report import OutlineSection
from ai_status_report.schemas.report_spec import default_report_spec
from ai_status_report.storage.search_results import canonicalize_run_id


def classify_node(
    state: WorkflowState,
    *,
    project_root: Path,
    classify: Callable[[str], object],
    whole_report: bool,
) -> WorkflowState:
    trace = ControllerRunTrace.from_state(project_root, state)
    run_id = canonicalize_run_id(state["run_id"])
    intent = classify(state["user_input"])
    # Deterministic decisions expose ``kind`` while the model router exposes
    # ``intent``. Both must become the plain intent string held by WorkflowState.
    intent_value = getattr(intent, "intent", getattr(intent, "kind", intent))
    intent_kind = getattr(intent_value, "value", intent_value)
    intent_reason = getattr(intent, "reason", "external_classifier")
    trace.emit(
        phase="classification",
        event_type="controller.request.classified",
        status="completed",
        agent_id="controller",
        refs={"intent": str(intent_kind), "reason": str(intent_reason)},
        summary="controller classified the user request",
    )
    return {
        "run_id": run_id,
        "intent": str(intent_kind),
        "phase": "classified",
        "events": ["intent.classified"],
        **(
            {
                "report_spec": default_report_spec().model_dump(mode="json"),
                "report_sections": [item.model_dump(mode="json") for item in default_report_spec().sections],
                "report_section_index": 0,
                "approved_section_summaries": [],
                "approved_chapter_artifacts": {},
            }
            if whole_report
            else {}
        ),
        **trace.state_update(),
    }


async def prepare_report_node(
    state: WorkflowState,
    *,
    project_root: Path,
    search_agent_url: str | None,
    whole_report: bool,
    sender: Callable,
) -> WorkflowState:
    trace = ControllerRunTrace.from_state(project_root, state)
    spec = default_report_spec()
    if whole_report:
        section_index = int(state.get("report_section_index", 0))
        if section_index < 0 or section_index >= len(spec.sections):
            raise RuntimeError("report_section_index_invalid")
        section = spec.sections[section_index]
        section_questions = section.core_questions
    else:
        section = OutlineSection(
            section_id="current-status", title="当前发展现状", questions=[state["user_input"]]
        )
        section_questions = section.questions
    if not search_agent_url:
        trace.emit(
            phase="search_dispatch",
            event_type="controller.search.awaiting_dispatch",
            status="started",
            agent_id="controller",
            summary="search agent URL was not configured",
        )
        return {
            "phase": "awaiting_a2a_dispatch",
            "events": [*state.get("events", []), "report.ready_for_dispatch"],
            **trace.state_update(),
        }
    task = json.dumps(
        {
            "task_type": "research.search",
            "run_id": state["run_id"],
            "query": f"{state['user_input']} {section.title} {' '.join(section_questions)}",
            "max_results": 5,
            "source_preference": "auto",
            "evidence_requirement": "retrievable",
            "cache_policy": "default",
        },
        ensure_ascii=False,
    )
    trace.outbound(agent="search_agent", url=search_agent_url, task=task, task_type="research.search")
    artifacts: list[str] = []
    async for response in sender(search_agent_url, task):
        trace.inbound(agent="search_agent", response=response)
        artifacts.extend(artifact_texts(response))
    report_text = next((text for text in reversed(artifacts) if text.strip()), "")
    if not report_text:
        raise RuntimeError("a2a_report_artifact_missing")
    report = json.loads(report_text)
    report_status = str(report.get("status") or "")
    phase = {
        "completed": "search_completed",
        "partial": "search_partial",
        "failed": "search_failed",
    }.get(report_status, "search_failed")
    events = [*state.get("events", [])]
    if report.get("plan_source") in {"model", "model_cache", "deterministic"}:
        events.append("search.plan.generated")
    if report.get("plan_source") == "deterministic_fallback":
        events.append("search.plan.fallback")
    if report.get("providers"):
        events.append("search.route.selected")
    events.append(
        "search.results.merged"
        if report_status == "completed"
        else "search.results.partial"
        if report_status == "partial"
        else "search.results.failed"
    )
    events.append("a2a.search.completed")
    if report.get("raw_refs") or report.get("raw_ref"):
        events.append("search.persisted")
    trace.emit(
        phase="search_dispatch",
        event_type="controller.search.artifact_received",
        status="completed" if report_status in {"completed", "partial"} else "failed",
        agent_id="controller",
        refs={
            "report_status": report_status,
            "incomplete_reason": str(report.get("incomplete_reason") or ""),
        },
        summary="controller accepted the Search Agent Artifact",
    )
    return {
        "phase": phase,
        "a2a_tasks": [*state.get("a2a_tasks", []), task],
        "a2a_results": [*state.get("a2a_results", []), *artifacts],
        "research_report": report,
        "research_sources": report.get("sources", []),
        **(
            {
                "report_spec": spec.model_dump(mode="json"),
                "report_sections": [item.model_dump(mode="json") for item in spec.sections],
            }
            if whole_report
            else {}
        ),
        "events": events,
        **trace.state_update(),
    }
