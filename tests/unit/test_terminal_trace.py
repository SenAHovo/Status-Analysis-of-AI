import json

from ai_status_report.presentation.terminal_trace import ChineseEventMapper, TerminalTraceFollower


def event(event_id, event_type, *, refs=None, agent_id=None, summary=""):
    return {
        "event_id": event_id,
        "event_type": event_type,
        "refs": refs or {},
        "agent_id": agent_id,
        "summary": summary,
    }


def test_mapper_uses_controlled_fields_and_ignores_untrusted_summary():
    message = ChineseEventMapper().render(
        event(
            "evt-1",
            "controller.document.task_prepared",
            refs={"section_id": "current-status"},
            summary="secret payload must not be printed",
        )
    )

    assert message == "主控已准备现状章节的写作任务。"
    assert "secret" not in message


def test_mapper_renders_review_and_a2a_delivery_messages():
    mapper = ChineseEventMapper()

    assert mapper.render(
        event("evt-1", "controller.review.decided", refs={"section_id": "trends", "decision": "need_evidence"})
    ) == "趋势章节需要定向补证。"
    assert mapper.render(
        event("evt-2", "a2a.artifact_update", agent_id="document_agent")
    ) == "文档 Agent 已通过 A2A 交付章节成果。"


def test_mapper_renders_controlled_tavily_rate_limit_messages():
    mapper = ChineseEventMapper()

    assert mapper.render(
        event(
            "evt-1",
            "a2a.status_update",
            refs={"progress_code": "tavily_rate_limited_wait:42:2"},
            agent_id="search_agent",
        )
    ) == "网络搜索 Agent 检测到 Tavily 限流，等待 42 秒后重试（第 2/2 次）。"
    assert mapper.render(
        event(
            "evt-2",
            "controller.search.artifact_received",
            refs={"report_status": "failed", "incomplete_reason": "tavily_rate_limited"},
        )
    ) == "网络搜索 Agent 被 Tavily 限流，当前章节未进入证据索引。"


def test_follower_reads_only_controller_trace_lines_once_and_handles_partial_lines(tmp_path):
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir()
    trace_path = trace_directory / "trace-1.jsonl"
    a2a_path = trace_directory / "trace-1.a2a.jsonl"
    output = []
    follower = TerminalTraceFollower(trace_directory, output=output.append)

    trace_path.write_text(
        json.dumps(event("evt-1", "controller.request.classified"), ensure_ascii=False) + "\n"
        + json.dumps(event("evt-2", "controller.document.task_prepared", refs={"section_id": "background"}), ensure_ascii=False)[:20],
        encoding="utf-8",
    )
    a2a_path.write_text(json.dumps(event("evt-a2a", "controller.report.pdf_exported")) + "\n", encoding="utf-8")

    assert follower.poll_once() == 1
    assert output == ["主控已识别报告请求，开始准备固定四章任务。"]

    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event("evt-2", "controller.document.task_prepared", refs={"section_id": "background"}), ensure_ascii=False)[20:] + "\n")

    assert follower.poll_once() == 1
    assert output[-1] == "主控已准备背景章节的写作任务。"
    assert follower.poll_once() == 0
