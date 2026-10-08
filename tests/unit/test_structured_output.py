"""Structured-output validation and safe error handling."""

import pytest
from pydantic import BaseModel, Field

from ai_status_report.model.structured import (
    StructuredOutputError,
    coerce_json,
    parse_structured,
    schema_hint,
)


class Ready(BaseModel):
    ready: bool


class Plan(BaseModel):
    topic: str = Field(min_length=1)
    sections: list[str] = Field(min_length=1)


def test_coerce_json_plain_and_fenced():
    assert coerce_json('{"ready": true}') == {"ready": True}
    assert coerce_json('```json\n{"ready": true}\n```') == {"ready": True}
    assert coerce_json('```{"ready": true}```') == {"ready": True}


def test_coerce_json_tolerates_surrounding_prose():
    assert coerce_json('以下是结果：\n{"ready": true}\n以上。') == {"ready": True}


def test_coerce_json_rejects_invalid():
    with pytest.raises(StructuredOutputError):
        coerce_json("not json at all")


def test_parse_structured_validates_contract():
    parsed = parse_structured('{"topic": "AI", "sections": ["背景"]}', Plan)
    assert parsed.topic == "AI"
    assert parsed.sections == ["背景"]


def test_parse_structured_fenced_content():
    parsed = parse_structured('```json\n{"ready": false}\n```', Ready)
    assert parsed.ready is False


def test_parse_structured_schema_mismatch_is_safe():
    # The error must not echo the offending model output or payload.
    with pytest.raises(StructuredOutputError, match="schema_mismatch"):
        parse_structured('{"topic": "", "sections": []}', Plan)


def test_schema_hint_lists_required_fields():
    hint = schema_hint(Ready)
    assert "ready" in hint
    assert "必填" in hint
