"""Observability events: timezone enforcement, sinks and recorder sequence."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_status_report.observability import (
    STATUS_COMPLETED,
    Event,
    InMemorySink,
    JsonlSink,
    Recorder,
)

NOW = datetime(2026, 9, 8, 3, 0, tzinfo=UTC)


def test_event_requires_aware_timestamp():
    naive = datetime.fromisoformat("2026-09-08T03:00:00")
    payload = {
        "event_id": "evt-1",
        "run_id": "run-1",
        "producer_id": "unit",
        "sequence": 1,
        "timestamp": naive,
        "phase": "write",
        "event_type": "model.call.started",
        "status": "started",
    }
    with pytest.raises(ValidationError):
        Event(**payload)


def test_recorder_sequences_and_sinks(tmp_path):
    sink = InMemorySink()
    recorder = Recorder("run-1", "controller", clock=lambda: NOW, sink=sink)
    first = recorder.emit(phase="plan", event_type="graph.node", status="started")
    second = recorder.emit(
        phase="plan",
        event_type="graph.node",
        status=STATUS_COMPLETED,
        agent_id="controller",
        refs={"section_id": "s1"},
        metrics={"elapsed_ms": 12},
    )
    assert first.sequence == 1
    assert second.sequence == 2
    assert len(sink.events) == 2
    assert sink.events[1].refs["section_id"] == "s1"


def test_recorder_without_sink_still_builds_events():
    recorder = Recorder("run-1", "controller", clock=lambda: NOW)
    event = recorder.emit(phase="model", event_type="model.call.started", status="started")
    assert event.timestamp.tzinfo is not None


def test_jsonl_sink_appends_parseable_lines(tmp_path):
    path = tmp_path / "trace" / "events.jsonl"
    sink = JsonlSink(path)
    recorder = Recorder("run-1", "controller", clock=lambda: NOW, sink=sink)
    recorder.emit(phase="plan", event_type="graph.node", status="started")

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"run_id":"run-1"' in lines[0]
    assert "run-1" in str(Event.model_validate_json(lines[0]))
