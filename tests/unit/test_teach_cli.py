import json
import sys
from types import SimpleNamespace
from typing import ClassVar

from ai_status_report import cli
from ai_status_report.cli import run_teaching_session
from ai_status_report.rag.chroma import RagIndexError
from ai_status_report.storage.search_results import register_run_directory, run_directory
from ai_status_report.token_budget import RUN, BudgetLimits
from ai_status_report.token_budget.runtime import PersistentTokenLedger


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


def test_teaching_dialogue_ledger_promotes_into_the_labeled_report_run(tmp_path):
    session_root = tmp_path / "data" / "teaching_sessions" / "session-1"
    ledger = PersistentTokenLedger(
        "teach-ledger",
        session_root / "dialogue_token_usage.json",
        limits=BudgetLimits(),
    )
    reservation = ledger.reserve(RUN, 20)
    ledger.settle(reservation, 12)
    run_root = register_run_directory(tmp_path, "teach-ledger", "人工智能现状")

    destination = cli._promote_teaching_dialogue_ledger(ledger, run_directory_path=run_root)

    assert destination == run_root / "token_usage.json"
    assert json.loads(destination.read_text(encoding="utf-8"))["actual"] == {RUN: 12}
    assert run_directory(tmp_path, "teach-ledger") == run_root
    assert not ledger.path.exists()
    assert not ledger.lock_path.exists()


def test_teaching_session_accounts_model_dialogue_in_the_report_ledger(monkeypatch, tmp_path):
    output = []
    ledgers = []

    class FakeDeepSeekClient:
        def __init__(self, generation, timeout, *, ledger):
            assert generation == "generation-config"
            assert timeout == 20
            self.ledger = ledger
            ledgers.append(ledger)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def chat(self, _messages, **_options):
            reservation = self.ledger.reserve(RUN, 20)
            self.ledger.settle(reservation, 12)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "report_request",
                                    "reply": "",
                                    "report_subject": "人工智能现状",
                                    "needs_clarification": False,
                                    "questions": [],
                                    "preferences": [],
                                    "confidence": 1.0,
                                    "reason_code": "explicit_defaults",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda _root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(cli, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(
        cli,
        "_new_teaching_dialogue_ledger",
        lambda _root, *, run_id, session_directory: PersistentTokenLedger(
            run_id,
            session_directory / "dialogue_token_usage.json",
            limits=BudgetLimits(),
        ),
    )

    result = run_teaching_session(
        tmp_path,
        run_id="teach-dialogue-ledger",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=True,
        input_reader=lambda _prompt: "输出一份人工智能现状分析报告，偏好默认即可",
        output=output.append,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
        workflow_runner=lambda _root, **kwargs: {
            "run_id": kwargs["run_id"],
            "phase": "report_pdf_exported",
            "report_markdown_artifact": {},
            "report_pdf_artifact": {},
            "events": [],
            "a2a_tasks": [],
        },
        rag_preflight=lambda _root: 0,
    )

    usage_path = run_directory(tmp_path, "teach-dialogue-ledger") / "token_usage.json"
    assert result["status"] == "completed"
    assert ledgers[0].run_id == "teach-dialogue-ledger"
    assert json.loads(usage_path.read_text(encoding="utf-8"))["actual"] == {RUN: 12}
    assert not list((tmp_path / "data" / "teaching_sessions").rglob("dialogue_token_usage.json"))


def test_teaching_session_discards_dialogue_ledger_without_a_report(monkeypatch, tmp_path):
    class FakeDeepSeekClient:
        def __init__(self, _generation, _timeout, *, ledger):
            self.ledger = ledger

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def chat(self, _messages, **_options):
            reservation = self.ledger.reserve(RUN, 20)
            self.ledger.settle(reservation, 12)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "conversation",
                                    "reply": "你好。",
                                    "report_subject": "",
                                    "needs_clarification": False,
                                    "questions": [],
                                    "preferences": [],
                                    "confidence": 1.0,
                                    "reason_code": "conversation",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda _root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(cli, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(
        cli,
        "_new_teaching_dialogue_ledger",
        lambda _root, *, run_id, session_directory: PersistentTokenLedger(
            run_id,
            session_directory / "dialogue_token_usage.json",
            limits=BudgetLimits(),
        ),
    )
    answers = iter(["你好", ""])

    result = run_teaching_session(
        tmp_path,
        run_id="teach-chat-only",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=True,
        input_reader=lambda _prompt: next(answers),
        output=lambda _line: None,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
    )

    assert result["status"] == "cancelled"
    assert not list((tmp_path / "data" / "teaching_sessions").rglob("dialogue_token_usage.json"))
    assert not (tmp_path / "data" / "runs").exists()


def test_teaching_session_keeps_confirmed_report_ledger_when_service_start_fails(monkeypatch, tmp_path):
    class FailingManager(FakeManager):
        def ensure_services(self):
            raise RuntimeError("service_start_failed")

    class FakeDeepSeekClient:
        def __init__(self, _generation, _timeout, *, ledger):
            self.ledger = ledger

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def chat(self, _messages, **_options):
            reservation = self.ledger.reserve(RUN, 20)
            self.ledger.settle(reservation, 12)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "report_request",
                                    "reply": "",
                                    "report_subject": "人工智能现状",
                                    "needs_clarification": False,
                                    "questions": [],
                                    "preferences": [],
                                    "confidence": 1.0,
                                    "reason_code": "explicit_defaults",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda _root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(cli, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(
        cli,
        "_new_teaching_dialogue_ledger",
        lambda _root, *, run_id, session_directory: PersistentTokenLedger(
            run_id,
            session_directory / "dialogue_token_usage.json",
            limits=BudgetLimits(),
        ),
    )

    try:
        run_teaching_session(
            tmp_path,
            run_id="teach-service-failure",
            search_agent_url="http://127.0.0.1:8001/",
            document_agent_url="http://127.0.0.1:8002/",
            model_route=True,
            input_reader=lambda _prompt: "输出一份人工智能现状分析报告，偏好默认即可",
            output=lambda _line: None,
            service_manager_cls=FailingManager,
            trace_follower_cls=FakeFollower,
        )
    except RuntimeError as exc:
        assert str(exc) == "service_start_failed"
    else:
        raise AssertionError("expected_service_start_failure")

    usage_path = run_directory(tmp_path, "teach-service-failure") / "token_usage.json"
    assert json.loads(usage_path.read_text(encoding="utf-8"))["actual"] == {RUN: 12}
    assert not list((tmp_path / "data" / "teaching_sessions").rglob("dialogue_token_usage.json"))


def test_teaching_session_preserves_dialogue_usage_after_model_failure(monkeypatch, tmp_path):
    class FakeDeepSeekClient:
        def __init__(self, _generation, _timeout, *, ledger):
            self.ledger = ledger

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def chat(self, _messages, **_options):
            reservation = self.ledger.reserve(RUN, 20)
            self.ledger.settle(reservation, 12)
            return {"choices": [{"message": {"content": "not-json"}}]}

    monkeypatch.setattr(
        cli,
        "load_settings",
        lambda _root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(cli, "DeepSeekClient", FakeDeepSeekClient)
    monkeypatch.setattr(
        cli,
        "_new_teaching_dialogue_ledger",
        lambda _root, *, run_id, session_directory: PersistentTokenLedger(
            run_id,
            session_directory / "dialogue_token_usage.json",
            limits=BudgetLimits(),
        ),
    )
    output = []

    result = run_teaching_session(
        tmp_path,
        run_id="teach-dialogue-failure",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=True,
        input_reader=lambda _prompt: "你好",
        output=output.append,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
    )

    usage_path = run_directory(tmp_path, "teach-dialogue-failure") / "token_usage.json"
    assert result == {"status": "failed", "error": "invalid_json"}
    assert run_directory(tmp_path, "teach-dialogue-failure").name.endswith("__对话失败")
    assert json.loads(usage_path.read_text(encoding="utf-8"))["actual"] == {RUN: 24}
    assert any(f"Token 账本：{usage_path}" == line for line in output)
    assert not list((tmp_path / "data" / "teaching_sessions").rglob("dialogue_token_usage.json"))


def test_teaching_session_rejects_an_occupied_explicit_run_id_before_model_call(monkeypatch, tmp_path):
    register_run_directory(tmp_path, "teach-occupied", "既有报告")
    output = []

    class UnexpectedDeepSeekClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("model_should_not_be_called")

    monkeypatch.setattr(cli, "DeepSeekClient", UnexpectedDeepSeekClient)

    result = run_teaching_session(
        tmp_path,
        run_id="teach-occupied",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        model_route=True,
        input_reader=lambda _prompt: (_ for _ in ()).throw(AssertionError("input_should_not_be_read")),
        output=output.append,
        service_manager_cls=FakeManager,
        trace_follower_cls=FakeFollower,
    )

    assert result == {"status": "failed", "error": "teaching_run_id_already_exists"}
    assert output == ["[对话失败] 错误码：teaching_run_id_already_exists，请使用新的 --run-id。"]
    assert not list((tmp_path / "data" / "teaching_sessions").rglob("dialogue_token_usage.json"))


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
