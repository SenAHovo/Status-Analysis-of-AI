"""Run-scoped controller trace for human-readable A2A teaching demonstrations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_status_report.a2a.client import describe_response
from ai_status_report.observability import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_STARTED,
    JsonlSink,
    Recorder,
    is_sensitive_key,
    redact_text,
)
from ai_status_report.schemas.common import new_id
from ai_status_report.storage.search_results import canonicalize_run_id, run_directory


def _a2a_event_status(item: dict[str, object]) -> str:
    """Map an A2A envelope to the project event status vocabulary."""

    state = str(item.get("state") or "")
    if state in {"TASK_STATE_FAILED", "TASK_STATE_CANCELED", "TASK_STATE_REJECTED"}:
        return STATUS_FAILED
    if state == "TASK_STATE_COMPLETED" or item.get("kind") == "artifact_update":
        return STATUS_COMPLETED
    return STATUS_STARTED


@dataclass
class ControllerRunTrace:
    """Record controller facts and received A2A envelopes for one graph run.

    The compact ``a2a_trace`` remains available in LangGraph state for a UI or
    CLI. The sanitized event ledger is persisted under the run directory for
    replay. It describes actual events only; controller-local actions and A2A
    protocol envelopes deliberately use different event types.
    """

    root: Path
    run_id: str
    trace_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    a2a_trace: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_state(cls, root: Path, state: dict[str, Any]) -> ControllerRunTrace:
        run_id = canonicalize_run_id(str(state["run_id"]))
        return cls(
            root=root,
            run_id=run_id,
            trace_id=str(state.get("trace_id") or new_id("trace")),
            events=list(state.get("execution_trace", [])),
            a2a_trace=list(state.get("a2a_trace", [])),
        )

    @property
    def path(self) -> Path:
        return run_directory(self.root, self.run_id) / "traces" / f"{self.trace_id}.jsonl"

    @property
    def a2a_path(self) -> Path:
        """Return the compact A2A envelope ledger for this controller run."""

        return run_directory(self.root, self.run_id) / "traces" / f"{self.trace_id}.a2a.jsonl"

    @staticmethod
    def _sanitize_a2a_value(value: object) -> object:
        """Keep the demo trace readable without retaining credentials or huge payloads."""

        if isinstance(value, dict):
            return {
                str(key): (
                    "[REDACTED]"
                    if is_sensitive_key(str(key))
                    else ControllerRunTrace._sanitize_a2a_value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ControllerRunTrace._sanitize_a2a_value(item) for item in value]
        if isinstance(value, str):
            sanitized = redact_text(value)
            # A2A business payloads are text parts. When one contains JSON,
            # redact field names too (for example, a mistakenly supplied API
            # key) before the protocol trace is persisted.
            try:
                parsed = json.loads(sanitized)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (dict, list)):
                sanitized = json.dumps(
                    ControllerRunTrace._sanitize_a2a_value(parsed),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return sanitized if len(sanitized) <= 4000 else sanitized[:4000] + "…[TRUNCATED]"
        return value

    def _append_a2a(self, item: dict[str, object]) -> dict[str, Any]:
        """Append one sanitized protocol envelope to state and its run ledger."""

        safe_item = self._sanitize_a2a_value(item)
        if not isinstance(safe_item, dict):  # Defensive: input always starts as a dict.
            raise TypeError("invalid_a2a_trace_item")
        self.a2a_trace.append(safe_item)
        self.a2a_path.parent.mkdir(parents=True, exist_ok=True)
        with self.a2a_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_item, ensure_ascii=False, separators=(",", ":")) + "\n")
        return safe_item

    def emit(
        self,
        *,
        phase: str,
        event_type: str,
        status: str,
        agent_id: str | None = None,
        refs: dict[str, str] | None = None,
        summary: str | None = None,
    ) -> None:
        event = Recorder(
            self.run_id,
            "controller",
            initial_sequence=len(self.events),
            sink=JsonlSink(self.path),
        ).emit(
            phase=phase,
            event_type=event_type,
            status=status,
            agent_id=agent_id,
            refs=refs,
            summary=summary,
            trace_id=self.trace_id,
        )
        self.events.append(event.model_dump(mode="json"))

    def outbound(
        self,
        *,
        agent: str,
        url: str,
        task: str,
        task_type: str,
        section_id: str | None = None,
    ) -> None:
        self._append_a2a(
            {
                "kind": "outbound_task",
                "agent": agent,
                "url": url,
                "task_type": task_type,
                "text": task,
            }
        )
        refs = {"task_type": task_type}
        if section_id:
            refs["section_id"] = section_id
        self.emit(
            phase="a2a_dispatch",
            event_type="controller.a2a.dispatch",
            status=STATUS_STARTED,
            agent_id=agent,
            refs=refs,
            summary=f"controller dispatched {task_type}",
        )

    def inbound(self, *, agent: str, response: object) -> dict[str, object]:
        item = describe_response(response)
        item["agent"] = agent
        item = self._append_a2a(item)
        refs = {
            key: str(item[key])
            for key in ("task_id", "context_id", "artifact_id", "name", "progress_code")
            if item.get(key)
        }
        self.emit(
            phase="a2a_stream",
            event_type=f"a2a.{item['kind']}",
            status=_a2a_event_status(item),
            agent_id=agent,
            refs=refs,
            summary=str(item.get("text") or "")[:2000] or None,
        )
        return item

    def state_update(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "trace_path": str(self.path),
            "a2a_trace_path": str(self.a2a_path),
            "execution_trace": self.events,
            "a2a_trace": self.a2a_trace,
        }
