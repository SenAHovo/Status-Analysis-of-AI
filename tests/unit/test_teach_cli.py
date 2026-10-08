import sys
from types import SimpleNamespace
from typing import ClassVar

from ai_status_report import cli
from ai_status_report.cli import run_teaching_session
from ai_status_report.rag.chroma import RagIndexError


class FakeManager:
    def __init__(self, root):
        self.root = root
        self.log_directory = root / "data" / "teaching_sessions" / "session-1" / "services"
        self.stopped = False

    def ensure_services(self):
        return [
            SimpleNamespace(spec=SimpleNamespace(name="chroma"), reused=False),
            SimpleNamespace(spec=SimpleNamespace(name="search-agent"), reused=True),
        ]

    def stop(self):
        self.stopped = True


class FakeFollower:
    instances: ClassVar[list["FakeFollower"]] = []

    def __init__(self, trace_directory, *, output):
        self.trace_directory = trace_directory
        self.output = output
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def poll_once(self):
        return 0


def test_teaching_session_connects_services_input_trace_and_existing_workflow(tmp_path):
    output = []
    managers = []
    calls = []

    def manager_factory(root):
        manager = FakeManager(root)
        managers.append(manager)
        return manager

    def workflow(root, **kwargs):
        calls.append((root, kwargs))
        return {
            "run_id": kwargs["run_id"],
            "phase": "report_pdf_exported",
            "report_markdown_artifact": {"access_ref": "report.md"},
            "report_pdf_artifact": {"access_ref": "report.pdf"},
            "events": ["done"],
            "a2a_tasks": ["search", "document"],
        }

    answers = iter(["智能体交易安全", "默认"])
    result = run_teaching_session(
        tmp_path,
        run_id="teach-test",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=False,
        input_reader=lambda _prompt: next(answers),
        output=output.append,
        service_manager_cls=manager_factory,
        trace_follower_cls=FakeFollower,
        workflow_runner=workflow,
        rag_preflight=lambda _root: 0,
    )

    assert result["phase"] == "report_pdf_exported"
    assert calls[0][1]["topic"] == "智能体交易安全"
    assert calls[0][1]["run_id"] == "teach-test"
    assert calls[0][1]["research_brief"]["run_id"] == "teach-test"
    assert FakeFollower.instances[-1].trace_directory.parent.name == "teach-test__智能体交易安全"
    assert FakeFollower.instances[-1].started is True
    assert FakeFollower.instances[-1].stopped is True
    assert managers[0].stopped is True
    assert any("服务日志" in line for line in output)
    assert any("直接回复“默认”将使用：写作风格：公正客观" in line for line in output)
    assert not any("（公正客观" in line for line in output)
    assert any("[偏好] 写作风格：公正客观（默认）" in line for line in output)
    assert any("[偏好] 使用场景：教学演示（默认）" in line for line in output)
    assert not any("[偏好] 资料偏好" in line for line in output)
    assert not any("[偏好] 篇幅" in line for line in output)
    assert any("索引预检通过（当前索引块：0）" in line for line in output)


def test_teaching_session_recovers_owned_broken_rag_store_before_workflow(tmp_path):
    output = []
    calls = []
    resets = []
    answers = iter(["人工智能现状", "默认"])

    def workflow(root, **kwargs):
        calls.append(kwargs)
        return {
            "run_id": kwargs["run_id"],
            "phase": "report_pdf_exported",
            "report_markdown_artifact": {},
            "report_pdf_artifact": {},
            "events": [],
            "a2a_tasks": [],
        }

    outcomes = iter([RagIndexError("chroma_index_unavailable"), 0])

    def preflight(_root):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = run_teaching_session(
        tmp_path,
        run_id="teach-recovered",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=False,
        input_reader=lambda _prompt: next(answers),
        output=output.append,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
        workflow_runner=workflow,
        rag_preflight=preflight,
        rag_store_reset=lambda root: resets.append(root) or {"status": "cleared"},
    )

    assert result["phase"] == "report_pdf_exported"
    assert resets == [tmp_path]
    assert len(calls) == 1
    assert any("索引预检失败，正在重建" in line for line in output)


def test_teaching_session_marks_non_pdf_terminal_phase_as_incomplete(tmp_path):
    output = []
    answers = iter(["人工智能现状", "默认"])

    result = run_teaching_session(
        tmp_path,
        run_id="teach-search-failed",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=False,
        input_reader=lambda _prompt: next(answers),
        output=output.append,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
        workflow_runner=lambda _root, **kwargs: {
            "run_id": kwargs["run_id"],
            "phase": "search_failed",
            "report_markdown_artifact": {},
            "report_pdf_artifact": {},
            "events": [],
            "a2a_tasks": [],
        },
        rag_preflight=lambda _root: 0,
    )

    assert result["status"] == "incomplete"
    assert any("[未完成] 阶段：search_failed" in line for line in output)


def test_teach_command_runs_without_allow_paid_flag(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'teaching-test'\n", encoding="utf-8")
    calls = []

    def fake_session(root, **kwargs):
        calls.append((root, kwargs))
        return {"status": "cancelled"}

    monkeypatch.setattr(cli, "run_teaching_session", fake_session)
    monkeypatch.setattr(sys, "argv", ["ai-status", "--root", str(tmp_path), "teach"])

    assert cli.main() == 0
    assert calls[0][0] == tmp_path.resolve()


def test_run_report_command_runs_without_allow_paid_flag(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'teaching-test'\n", encoding="utf-8")
    calls = []

    def fake_workflow(root, **kwargs):
        calls.append((root, kwargs))
        return {
            "run_id": kwargs["run_id"],
            "phase": "report_pdf_exported",
            "report_pdf_artifact": {"access_ref": "report.pdf"},
            "events": [],
            "a2a_tasks": [],
        }

    monkeypatch.setattr(cli, "_run_report_workflow", fake_workflow)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-status",
            "--root",
            str(tmp_path),
            "run-report",
            "--topic",
            "topic",
            "--run-id",
            "run-1",
        ],
    )

    assert cli.main() == 0
    assert calls[0][1]["run_id"] == "run-1"
    assert (
        tmp_path
        / "data"
        / "runs"
        / ".run_labels"
        / "run-1.json"
    ).is_file()
    assert cli.run_directory(tmp_path, "run-1").name == "run-1__topic"
