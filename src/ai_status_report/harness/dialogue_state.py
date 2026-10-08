"""Serializable state for the lightweight dialogue graph (ADR-0005 P1).

A TypedDict keeps the shape consistent with :mod:`ai_status_report.harness.state`
while the payloads carried inside (intent decision, questions, preferences,
brief) are validated Pydantic contracts. The message history stays in-process
and is trimmed by the caller; no cross-process checkpointer is used.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class DialogueMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class DialogueState(TypedDict, total=False):
    session_id: str
    messages: list[DialogueMessage]
    intent: str
    intent_reason: str
    intent_source: str
    confidence: float
    reply: str
    report_subject: str
    pending_questions: list[dict]
    selections: list[dict]
    brief: dict
    session_status: str
    exit_reason: str
    error: str
