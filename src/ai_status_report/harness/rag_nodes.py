"""RAG indexing node for the controller graph."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ai_status_report.harness.state import WorkflowState
from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.model.client import ProviderError
from ai_status_report.rag.chroma import RagIndexError
from ai_status_report.settings import ConfigError


def index_evidence_node(
    state: WorkflowState,
    *,
    project_root: Path,
    evidence_indexer: Callable[[Path, str], dict[str, object]],
) -> WorkflowState:
    """Index this run before 8002 creates its EvidenceBundle from Chroma."""

    trace = ControllerRunTrace.from_state(project_root, state)
    trace.emit(
        phase="rag_indexing",
        event_type="controller.rag.index.started",
        status="started",
        agent_id="controller",
        summary="controller started indexing Search Agent evidence for the document task",
    )
    try:
        index_result = evidence_indexer(project_root, state["run_id"])
        if not isinstance(index_result, dict):
            raise RagIndexError("invalid_rag_index_result")
    except (ConfigError, OSError, ProviderError, RagIndexError, RuntimeError, ValueError) as exc:
        reason = (
            "no_indexable_evidence"
            if isinstance(exc, RagIndexError) and str(exc) == "no_indexable_evidence"
            else "rag_indexing_failed"
        )
        trace.emit(
            phase="rag_indexing",
            event_type="controller.rag.index.failed",
            status="failed",
            agent_id="controller",
            refs={"reason": reason},
            summary="controller could not index evidence for the document task",
        )
        return {
            "phase": "rag_index_blocked",
            "rag_index": {"status": "failed", "reason": reason},
            "events": [*state.get("events", []), "rag.index.failed"],
            **trace.state_update(),
        }
    trace.emit(
        phase="rag_indexing",
        event_type="controller.rag.index.completed",
        status="completed",
        agent_id="controller",
        refs={"index_version": str(index_result.get("index_version") or "")},
        summary="controller indexed Search Agent evidence for the document task",
    )
    return {
        "phase": "rag_indexed",
        "rag_index": {"status": "completed", **index_result},
        "events": [*state.get("events", []), "rag.index.completed"],
        **trace.state_update(),
    }
