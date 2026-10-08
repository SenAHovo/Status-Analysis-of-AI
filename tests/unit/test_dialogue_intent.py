"""ADR-0005 P0: entry intent decisions must never misroute small talk.

The reported failure was a capability question starting a full report run.
These regressions pin the deterministic layer, the upgraded decision contract
and the safe fallback when the model output cannot be validated.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_status_report.briefing.intents import classify
from ai_status_report.model.structured import StructuredOutputError, parse_structured
from ai_status_report.schemas import DialogueModelOutput
from ai_status_report.schemas.intent import DecisionSource, Intent, IntentDecision

CAPABILITY_PHRASES = (
    "你有什么能力",
    "你能做什么",
    "介绍下你自己",
    "有什么功能",
    "你会什么",
    "你是谁",
)


def test_capability_questions_never_route_to_a_report() -> None:
    for text in CAPABILITY_PHRASES:
        decision = classify(text)
        assert decision.kind is Intent.CONVERSATION, text


def test_rule_decisions_are_certain_and_deterministic() -> None:
    decision = classify("你有什么能力")

    assert decision.confidence == 1.0
    assert decision.source is DecisionSource.DETERMINISTIC
    assert decision.reason


def test_report_and_out_of_scope_boundaries_are_unchanged() -> None:
    assert classify("帮我分析人工智能现状").kind is Intent.REPORT_REQUEST
    assert classify("请分析人工智能在天气预报中的应用现状").kind is Intent.REPORT_REQUEST
    assert classify("你好，请分析一下人工智能发展现状").kind is Intent.REPORT_REQUEST
    assert classify("请帮我订一张去北京的机票").kind is Intent.UNSUPPORTED
    assert classify("今天天气怎么样").kind is Intent.UNSUPPORTED


def test_intent_decision_rejects_bad_confidence_source_and_reason() -> None:
    with pytest.raises(ValidationError):
        IntentDecision(kind=Intent.CONVERSATION, reason="ok", confidence=-0.1)
    with pytest.raises(ValidationError):
        IntentDecision(kind=Intent.CONVERSATION, reason="ok", confidence=1.1)
    with pytest.raises(ValidationError):
        IntentDecision(kind=Intent.CONVERSATION, reason="ok", source="hearsay")
    with pytest.raises(ValidationError):
        IntentDecision(kind=Intent.CONVERSATION, reason="")


def test_intent_decision_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        IntentDecision(kind="chat", reason="ok")


def test_invalid_model_output_never_yields_a_decision() -> None:
    for raw in ("not json at all", json.dumps({"intent": "chat", "confidence": 0.5})):
        with pytest.raises(StructuredOutputError):
            parse_structured(raw, DialogueModelOutput)


def test_valid_model_output_yields_a_model_sourced_decision() -> None:
    raw = json.dumps(
        {
            "intent": "conversation",
            "reply": "我负责分析人工智能现状并生成报告。",
            "confidence": 0.81,
            "reason_code": "capability_question",
        },
        ensure_ascii=False,
    )

    output = parse_structured(raw, DialogueModelOutput)
    decision = output.intent_decision(source=DecisionSource.MODEL)

    assert decision.kind is Intent.CONVERSATION
    assert decision.source is DecisionSource.MODEL
    assert decision.confidence == 0.81
