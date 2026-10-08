"""Bounded evidence-supplement dispatch node for the controller graph."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.graph_helpers import artifact_texts, evidence_snapshot
from ai_status_report.harness.review_audit import persist_supplement_audit
from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.model.client import ProviderError
from ai_status_report.rag.chroma import RagIndexError
from ai_status_report.schemas.review import ReviewDecision, SupplementPlan
from ai_status_report.schemas.search import ResearchReport
from ai_status_report.settings import ConfigError
from ai_status_report.storage.search_results import persist_research_report


async def supplement_search_node(
    state: WorkflowState,
    *,
    project_root: Path,
    search_agent_url: str | None,
    evidence_indexer: Callable[[Path, str], dict[str, object]],
    sender: Callable,
) -> WorkflowState:
    """Dispatch at most five independent A2A supplement searches for a chapter."""

    trace = ControllerRunTrace.from_state(project_root, state)
    decision = ReviewDecision.model_validate(state["review_decision"])
    plan_payload = state.get("supplement_plan")
    if not isinstance(plan_payload, dict):
        raise TypeError("supplement_plan_missing")
    plan = SupplementPlan.model_validate(plan_payload)
    if not search_agent_url:
        return {
            "phase": "supplement_dispatch_blocked",
            "events": [*state.get("events", []), "supplement.search.blocked"],
            **trace.state_update(),
        }
    round_number = plan.supplement_round
    queries = [item.query for item in plan.queries]
    previous_sources, previous_chunks = evidence_snapshot(project_root, state["run_id"])
    reports: list[dict] = []
    artifacts: list[str] = []
    tasks: list[str] = []
    report_refs: list[str] = []
    for query in queries:
        task = json.dumps(
            {
                "task_type": "research.search",
                "run_id": state["run_id"],
                "query": query,
                "max_results": 5,
                "source_preference": "auto",
                "evidence_requirement": "retrievable",
                "cache_policy": "default",
                "supplement_round": round_number,
                "section_id": decision.section_id,
            },
            ensure_ascii=False,
        )
        tasks.append(task)
        trace.outbound(
            agent="search_agent",
            url=search_agent_url,
            task=task,
            task_type="research.search",
        )
        search_artifacts: list[str] = []
        async for response in sender(search_agent_url, task):
            trace.inbound(agent="search_agent", response=response)
            search_artifacts.extend(artifact_texts(response))
        report_text = next((text for text in reversed(search_artifacts) if text.strip()), "")
        if not report_text:
            continue
        try:
            report = ResearchReport.model_validate_json(report_text)
        except ValueError:
            continue
        reports.append(report.model_dump(mode="json"))
        artifacts.extend(search_artifacts)
        report_ref = persist_research_report(project_root, report, run_id=state["run_id"])
        report_refs.append(str(report_ref))
    index_result: dict[str, object] = {}
    if reports:
        try:
            index_result = evidence_indexer(project_root, state["run_id"])
            if not isinstance(index_result, dict):
                raise RagIndexError("invalid_rag_index_result")
        except (ConfigError, OSError, ProviderError, RagIndexError, RuntimeError, ValueError):
            current_sources, current_chunks = evidence_snapshot(project_root, state["run_id"])
            evidence_delta = {
                "new_source_ids": sorted(current_sources - previous_sources),
                "new_chunk_ids": sorted(current_chunks - previous_chunks),
            }
            supplement_audit_ref = persist_supplement_audit(
                project_root,
                plan=plan,
                report_refs=report_refs,
                new_source_ids=evidence_delta["new_source_ids"],
                new_chunk_ids=evidence_delta["new_chunk_ids"],
            )
            return {
                "phase": "supplement_index_blocked",
                "supplement_round": round_number,
                "supplement_reports": reports,
                "supplement_audit_ref": str(supplement_audit_ref),
                "evidence_delta": evidence_delta,
                "a2a_tasks": [*state.get("a2a_tasks", []), *tasks],
                "a2a_results": [*state.get("a2a_results", []), *artifacts],
                "events": [
                    *state.get("events", []),
                    "supplement.search.completed",
                    "supplement.index.failed",
                ],
                **trace.state_update(),
            }
    current_sources, current_chunks = evidence_snapshot(project_root, state["run_id"])
    evidence_delta = {
        "new_source_ids": sorted(current_sources - previous_sources),
        "new_chunk_ids": sorted(current_chunks - previous_chunks),
    }
    supplement_audit_ref = persist_supplement_audit(
        project_root,
        plan=plan,
        report_refs=report_refs,
        new_source_ids=evidence_delta["new_source_ids"],
        new_chunk_ids=evidence_delta["new_chunk_ids"],
    )
    trace.emit(
        phase="supplement_search",
        event_type="controller.supplement.completed",
        status="completed" if reports else "failed",
        agent_id="controller",
        refs={
            "round": str(round_number),
            "query_count": str(len(queries)),
            "report_count": str(len(reports)),
            "new_chunk_count": str(len(evidence_delta["new_chunk_ids"])),
            "supplement_audit_ref": str(supplement_audit_ref),
        },
        summary="controller completed the bounded evidence supplement dispatch",
    )
    events = [*state.get("events", [])]
    events.append("supplement.search.completed" if reports else "supplement.search.no_results")
    events.append("supplement.awaiting_revision")
    return {
        "phase": "awaiting_document_revision",
        "supplement_round": round_number,
        "supplement_reports": reports,
        "supplement_audit_ref": str(supplement_audit_ref),
        "evidence_delta": evidence_delta,
        "a2a_tasks": [*state.get("a2a_tasks", []), *tasks],
        "a2a_results": [*state.get("a2a_results", []), *artifacts],
        "rag_index": (
            {"status": "completed", **index_result}
            if index_result
            else state.get("rag_index", {})
        ),
        "events": events,
        **trace.state_update(),
    }
