"""Controller trace persistence and trace-only redaction."""

import json

from ai_status_report.harness.trace import ControllerRunTrace
from ai_status_report.presentation.terminal_trace import TerminalTraceFollower


def test_controller_trace_persists_a2a_envelopes_without_json_credentials(tmp_path):
    trace = ControllerRunTrace.from_state(
        tmp_path,
        {"run_id": "run-1", "trace_id": "trace-1"},
    )

    trace.outbound(
        agent="search_agent",
        url="http://127.0.0.1:8001/",
        task=json.dumps({"task_type": "research.search", "api_key": "secret-value"}),
        task_type="research.search",
    )

    recorded = json.loads(trace.a2a_path.read_text(encoding="utf-8"))
    assert recorded["text"] == '{"task_type":"research.search","api_key":"[REDACTED]"}'
    assert "secret-value" not in trace.a2a_path.read_text(encoding="utf-8")


def test_document_dispatch_trace_carries_section_for_terminal_display(tmp_path):
    trace = ControllerRunTrace.from_state(
        tmp_path,
        {"run_id": "run-1", "trace_id": "trace-section"},
    )
    trace.outbound(
        agent="document_agent",
        url="http://127.0.0.1:8002/",
        task='{"task_type":"document.write_section"}',
        task_type="document.write_section",
        section_id="trends",
    )

    output: list[str] = []
    rendered = TerminalTraceFollower(trace.path.parent, output=output.append).poll_once()

    event = json.loads(trace.path.read_text(encoding="utf-8"))
    assert event["refs"] == {"task_type": "document.write_section", "section_id": "trends"}
    assert rendered == 1
    assert output == ["主控已通过 A2A 向文档 Agent 派发趋势章节写作任务。"]
