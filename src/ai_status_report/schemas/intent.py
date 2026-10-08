"""Entry intent vocabulary shared by the rule layer and the dialogue layer.

These names live in the schema layer so typed payloads can reference the intent
vocabulary without importing the briefing logic that classifies raw text.
``briefing.intents`` imports them and adds the deterministic classifier, so the
public import path used by existing callers stays ``briefing.intents``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from ai_status_report.schemas.common import ProjectModel


class Intent(str, Enum):
    """The three fixed entry kinds (ADR-0005)."""

    REPORT_REQUEST = "report_request"
    CONVERSATION = "conversation"
    UNSUPPORTED = "unsupported"


class DecisionSource(str, Enum):
    """Where an entry intent decision came from."""

    DETERMINISTIC = "deterministic"
    MODEL = "model"
    MODEL_CACHE = "model_cache"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"


class IntentDecision(ProjectModel):
    """Frozen three-way entry decision shared by the rule and dialogue layers.

    ``confidence`` and ``source`` record how the decision was produced so a
    model failure can never masquerade as a certain classification.
    """

    kind: Intent
    reason: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: DecisionSource = DecisionSource.DETERMINISTIC
