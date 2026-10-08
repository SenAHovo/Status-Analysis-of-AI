import json
from pathlib import Path

import pytest

from ai_status_report.harness.dialogue import (
    DialogueModelError,
    DialogueSession,
    StructuredDialogueModel,
)
from ai_status_report.model.client import ProviderError
from ai_status_report.schemas.dialogue import DialogueModelOutput, PreferenceField
from ai_status_report.schemas.intent import Intent


def test_deterministic_session_supports_report_preferences_and_handoff(tmp_path: Path):
    session = DialogueSession(tmp_path)

    first = session.handle("智能体交易安全")
    second = session.handle("默认")

    assert first.status == "needs_preferences"
    assert len(session.state["pending_questions"]) == 2
    assert second.status == "ready_for_report"
    assert second.brief is not None
    assert second.brief.topic == "智能体交易安全"
    assert second.brief.run_id
    assert session.state["session_status"] == "ready_for_report"


def test_report_clarification_uses_program_owned_questions(tmp_path: Path):
    def fake_model(_messages):
        return DialogueModelOutput(
            intent=Intent.REPORT_REQUEST,
            report_subject="人工智能现状",
            needs_clarification=True,
            confidence=0.9,
            reason_code="missing_preferences",
        )

    session = DialogueSession(tmp_path, model=fake_model)
    turn = session.handle("输出一份人工智能现状分析报告")

    assert turn.status == "needs_preferences"
    assert [item["field"] for item in session.state["pending_questions"]] == [
        field.value for field in PreferenceField
    ]


def test_default_preferences_bypass_the_model_and_start_the_report(tmp_path: Path):
    calls = 0

    def fake_model(_messages):
        nonlocal calls
        calls += 1
        return DialogueModelOutput(
            intent=Intent.REPORT_REQUEST,
            report_subject="人工智能现状",
            needs_clarification=True,
            confidence=0.9,
            reason_code="missing_preferences",
        )

    session = DialogueSession(tmp_path, model=fake_model)
    session.handle("输出一份人工智能现状分析报告")
    turn = session.handle("默认")

    assert calls == 1
    assert turn.status == "ready_for_report"
    assert turn.brief is not None
    assert turn.brief.topic


def test_first_report_turn_with_explicit_default_preferences_skips_clarification(tmp_path: Path):
    calls = 0

    def fake_model(_messages):
        nonlocal calls
        calls += 1
        return DialogueModelOutput(
            intent=Intent.REPORT_REQUEST,
            report_subject="人工智能现状",
            needs_clarification=True,
            confidence=0.9,
            reason_code="missing_preferences",
        )

    turn = DialogueSession(tmp_path, model=fake_model).handle(
        "输出一份人工智能现状分析报告，偏好默认即可"
    )

    assert calls == 1
    assert turn.status == "ready_for_report"
    assert turn.brief is not None
    assert {item.field: item.origin.value for item in turn.brief.defaults_applied}["writing_style"] == "default"


def test_waiting_preferences_accepts_default_synonym_without_a_second_model_call(tmp_path: Path):
    calls = 0

    def fake_model(_messages):
        nonlocal calls
        calls += 1
        return DialogueModelOutput(
            intent=Intent.REPORT_REQUEST,
            report_subject="人工智能现状",
            needs_clarification=True,
            confidence=0.9,
            reason_code="missing_preferences",
        )

    session = DialogueSession(tmp_path, model=fake_model)
    session.handle("输出一份人工智能现状分析报告")
    turn = session.handle("偏好默认即可")

    assert calls == 1
    assert turn.status == "ready_for_report"


def test_deterministic_session_keeps_small_talk_in_process(tmp_path: Path):
    session = DialogueSession(tmp_path)

    turn = session.handle("你有什么能力")

    assert turn.status == "conversation"
    assert turn.output.intent is Intent.CONVERSATION
    assert session.state["session_status"] == "awaiting_input"


def test_model_failure_is_safe_and_does_not_create_brief(tmp_path: Path):
    def broken(_messages):
        raise DialogueModelError("dialogue_model_failed")

    session = DialogueSession(tmp_path, model=broken)
    with pytest.raises(DialogueModelError, match="^dialogue_model_failed$"):
        session.handle("分析人工智能现状")
    assert "brief" not in session.state


def test_model_adapter_can_return_valid_structured_output(tmp_path: Path):
    def fake_model(_messages):
        return DialogueModelOutput(
            intent=Intent.CONVERSATION,
            reply="可以继续交流。",
            confidence=0.9,
            reason_code="conversation",
        )

    turn = DialogueSession(tmp_path, model=fake_model).handle("你好")

    assert turn.status == "conversation"
    assert turn.message == "可以继续交流。"


def test_empty_model_reply_gets_safe_conversation_fallback(tmp_path: Path):
    def fake_model(_messages):
        return DialogueModelOutput(
            intent=Intent.CONVERSATION,
            confidence=0.9,
            reason_code="conversation",
        )

    turn = DialogueSession(tmp_path, model=fake_model).handle("你好")

    assert turn.message


def test_dialogue_history_keeps_only_bounded_window(tmp_path: Path):
    def fake_model(_messages):
        return DialogueModelOutput(
            intent=Intent.CONVERSATION,
            reply="继续。",
            confidence=0.9,
            reason_code="conversation",
        )

    session = DialogueSession(tmp_path, model=fake_model)
    for index in range(20):
        session.handle(f"你好 {index}")

    assert len(session.state["messages"]) <= 10
    assert all(len(item["content"]) <= 4000 for item in session.state["messages"])


def test_model_result_must_be_validated_dialogue_output(tmp_path: Path):
    session = DialogueSession(tmp_path, model=lambda _messages: {"intent": "conversation"})

    with pytest.raises(DialogueModelError, match="^dialogue_model_failed$"):
        session.handle("你好")


def test_structured_model_adapter_validates_provider_json():
    class FakeClient:
        def chat(self, messages, **options):
            assert messages[0]["role"] == "system"
            assert options["response_format"] == {"type": "json_object"}
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "conversation",
                "reply": "可以继续交流。",
                "confidence": 0.9,
                "reason_code": "conversation",
            }, ensure_ascii=False)}}]}

    output = StructuredDialogueModel(FakeClient())([
        {"role": "user", "content": "你好"},
    ])

    assert output.intent is Intent.CONVERSATION


def test_structured_model_adapter_preserves_safe_provider_error_code():
    class FakeClient:
        def chat(self, messages, **options):
            raise ProviderError("http_401")

    with pytest.raises(DialogueModelError, match="^http_401$"):
        StructuredDialogueModel(FakeClient())([])


def test_structured_model_adapter_preserves_safe_parse_error_code():
    class FakeClient:
        def chat(self, messages, **options):
            return {"choices": [{"message": {"content": "not-json"}}]}

    with pytest.raises(DialogueModelError, match="^invalid_json$"):
        StructuredDialogueModel(FakeClient())([])


def test_structured_model_adapter_accepts_nulls_for_inapplicable_fields():
    class FakeClient:
        def chat(self, messages, **options):
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "conversation",
                "reply": "你好，我可以继续和你交流。",
                "report_subject": None,
                "preferences": None,
                "confidence": 0.9,
                "reason_code": "conversation",
            }, ensure_ascii=False)}}]}

    output = StructuredDialogueModel(FakeClient())([])

    assert output.report_subject == ""
    assert output.preferences == []


def test_structured_model_adapter_accepts_empty_object_for_empty_preferences():
    class FakeClient:
        def chat(self, messages, **options):
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "conversation",
                "reply": "你好。",
                "report_subject": None,
                "needs_clarification": False,
                "questions": [],
                "preferences": {},
                "confidence": 0.9,
                "reason_code": "greeting",
            }, ensure_ascii=False)}}]}

    output = StructuredDialogueModel(FakeClient())([])

    assert output.preferences == []


def test_structured_model_adapter_uses_program_questions_for_prose_question_list():
    class FakeClient:
        def chat(self, messages, **options):
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "report_request",
                "reply": "请补充偏好。",
                "report_subject": "人工智能现状",
                "needs_clarification": True,
                "questions": ["写作风格？", "使用场景？"],
                "preferences": [],
                "confidence": 0.9,
                "reason_code": "missing_preferences",
            }, ensure_ascii=False)}}]}

    output = StructuredDialogueModel(FakeClient())([])

    assert output.needs_clarification is True
    assert output.questions == []


def test_structured_model_adapter_retries_once_after_invalid_json():
    class FakeClient:
        def __init__(self):
            self.calls = []

        def chat(self, messages, **options):
            self.calls.append((messages, options))
            if len(self.calls) == 1:
                return {"choices": [{"message": {"content": "not-json"}}]}
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "conversation",
                "reply": "我可以生成报告。",
                "report_subject": "",
                "needs_clarification": False,
                "questions": [],
                "preferences": [],
                "confidence": 0.9,
                "reason_code": "capability_intro",
            }, ensure_ascii=False)}}]}

    client = FakeClient()
    output = StructuredDialogueModel(client)([{"role": "user", "content": "你能做什么"}])

    assert output.intent is Intent.CONVERSATION
    assert client.calls[0][1]["max_tokens"] == 2048
    assert len(client.calls) == 2
    assert client.calls[1][1]["temperature"] == 0
    assert client.calls[1][1]["max_tokens"] == 512


def test_structured_model_prompt_defines_explicit_default_preference_contract():
    class FakeClient:
        def chat(self, messages, **_options):
            system_prompt = messages[0]["content"]
            assert "默认即可" in system_prompt
            assert "needs_clarification=false" in system_prompt
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "report_request",
                "reply": "",
                "report_subject": "人工智能现状",
                "needs_clarification": False,
                "questions": [],
                "preferences": [],
                "confidence": 0.9,
                "reason_code": "explicit_defaults",
            }, ensure_ascii=False)}}]}

    output = StructuredDialogueModel(FakeClient())([
        {"role": "user", "content": "输出人工智能现状报告，偏好默认即可"},
    ])

    assert output.intent is Intent.REPORT_REQUEST


def test_structured_model_adapter_stops_after_second_invalid_json():
    class FakeClient:
        def chat(self, messages, **options):
            return {"choices": [{"message": {"content": "not-json"}}]}

    with pytest.raises(DialogueModelError, match="^invalid_json$"):
        StructuredDialogueModel(FakeClient())([])
