"""Observability events: fact recording only, never routing decisions.

An event describes what happened (phase, status, refs, metrics). Logs and
events never decide business branches and never fabricate model reasoning.
A ``skill.loaded`` event alone is not proof that guidance reached the model;
the context assembly record is the evidence for that.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import Field, field_validator

from ai_status_report.schemas.common import ProjectModel, new_id

STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
_REDACTED = "[REDACTED]"
_SECRET_TEXT = re.compile(r"(?i)(bearer\s+|sk-[a-z0-9_-]{8,}|api[_-]?key\s*[:=]\s*)[^\s,;]+")
_SENSITIVE_KEY = re.compile(r"(?i)(secret|token|password|api[_-]?key)")


def redact_text(value: str) -> str:
    """Remove credential-shaped fragments before a value reaches a trace."""

    return _SECRET_TEXT.sub(r"\1" + _REDACTED, value)


def is_sensitive_key(key: str) -> bool:
    """Return whether a field name must never be recorded verbatim."""

    return bool(_SENSITIVE_KEY.search(key))


def redact_refs(value: dict[str, str]) -> dict[str, str]:
    """Redact secret-bearing reference fields while retaining safe identifiers."""

    return {
        key: _REDACTED if is_sensitive_key(key) else redact_text(item)
        for key, item in value.items()
    }


class Event(ProjectModel):
    """One sanitized execution event."""

    event_id: str = Field(max_length=128)
    run_id: str = Field(max_length=128)
    producer_id: str = Field(max_length=64)
    sequence: int = Field(ge=0)
    timestamp: datetime
    trace_id: str | None = Field(default=None, max_length=128)
    span_id: str | None = Field(default=None, max_length=128)
    parent_span_id: str | None = Field(default=None, max_length=128)
    agent_id: str | None = Field(default=None, max_length=64)
    phase: str = Field(max_length=64)
    event_type: str = Field(max_length=128)
    status: str = Field(max_length=16)
    refs: dict[str, str] = Field(default_factory=dict)
    summary: str | None = Field(default=None, max_length=2000)
    metrics: dict[str, int | float] = Field(default_factory=dict)
    payload_ref: str | None = Field(default=None, max_length=300)

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event timestamps must carry a timezone")
        return value

    @field_validator("refs", mode="before")
    @classmethod
    def _redact_refs(cls, value: dict[str, str]) -> dict[str, str]:
        if not isinstance(value, dict):
            return value
        return redact_refs(value)

    @field_validator("summary", mode="before")
    @classmethod
    def _redact_summary(cls, value: str | None) -> str | None:
        return None if value is None else redact_text(value)

    def as_line(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))


class EventSink(Protocol):
    def write(self, event: Event) -> None: ...


class InMemorySink:
    def __init__(self):
        self.events: list[Event] = []

    def write(self, event: Event) -> None:
        self.events.append(event)


class JsonlSink:
    """Append events to one JSONL file owned by a single producer."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, event: Event) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.as_line() + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class Recorder:
    """Creates monotonic events for one producer and run."""

    def __init__(
        self,
        run_id: str,
        producer_id: str,
        *,
        initial_sequence: int = 0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        sink: EventSink | None = None,
    ):
        if initial_sequence < 0:
            raise ValueError("initial_sequence must be non-negative")
        self.run_id = run_id
        self.producer_id = producer_id
        self._clock = clock
        self._sequence = initial_sequence
        self._sequence_lock = threading.Lock()
        self.sink = sink

    def emit(
        self,
        *,
        phase: str,
        event_type: str,
        status: str,
        agent_id: str | None = None,
        refs: dict[str, str] | None = None,
        summary: str | None = None,
        metrics: dict[str, int | float] | None = None,
        trace_id: str | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> Event:
        with self._sequence_lock:
            self._sequence += 1
            sequence = self._sequence
        event = Event(
            event_id=new_id("evt", now=self._clock()),
            run_id=self.run_id,
            producer_id=self.producer_id,
            sequence=sequence,
            timestamp=self._clock(),
            agent_id=agent_id,
            phase=phase,
            event_type=event_type,
            status=status,
            refs=refs or {},
            summary=summary,
            metrics=metrics or {},
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        if self.sink is not None:
            self.sink.write(event)
        return event
