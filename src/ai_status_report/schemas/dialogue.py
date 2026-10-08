"""Dialogue-layer contracts for the interactive command-line entry (ADR-0005).

These payloads are frozen in P0 before the dialogue graph exists. They carry
only what the dialogue layer may decide: the entry intent, the clarification
questions, the two preference answers and the session summary. The dialogue
layer never starts services, changes budgets or bypasses report-graph nodes;
model output is validated here and bound into a decision by the program.
"""

from __future__ import annotations

from enum import Enum

from pydantic import Field, model_validator

from ai_status_report.schemas.common import ProjectModel
from ai_status_report.schemas.intent import DecisionSource, Intent, IntentDecision
from ai_status_report.schemas.research import (
    DEFAULT_READER_STYLE,
    DEFAULT_WRITING_STYLE,
    ReaderStyle,
    ResearchBrief,
    WritingStyle,
)


class PreferenceField(str, Enum):
    """The two teaching preferences collected once before a report run."""

    WRITING_STYLE = "writing_style"
    USAGE_SCENARIO = "usage_scenario"

OTHER_OPTION = "other"

_ALLOWED_VALUES: dict[PreferenceField, frozenset[str]] = {
    PreferenceField.WRITING_STYLE: frozenset(item.value for item in WritingStyle),
    PreferenceField.USAGE_SCENARIO: frozenset(item.value for item in ReaderStyle),
}

DEFAULT_PREFERENCES: dict[PreferenceField, str] = {
    PreferenceField.WRITING_STYLE: DEFAULT_WRITING_STYLE.value,
    PreferenceField.USAGE_SCENARIO: DEFAULT_READER_STYLE.value,
}

# Reader-facing labels for every trusted option. Options keep their machine
# value so validation stays strict, while the terminal renders Chinese text.
# The ADR "教学演示" default for the usage scenario is the label of
# ``ReaderStyle.GENERAL`` below, so that mapping exists in code.
PREFERENCE_OPTION_LABELS: dict[PreferenceField, dict[str, str]] = {
    PreferenceField.WRITING_STYLE: {
        WritingStyle.OBJECTIVE.value: "公正客观",
        WritingStyle.ANALYTICAL.value: "深度分析",
        WritingStyle.CONCISE.value: "简明扼要",
        WritingStyle.PRACTICAL.value: "务实应用",
    },
    PreferenceField.USAGE_SCENARIO: {
        ReaderStyle.GENERAL.value: "教学演示",
        ReaderStyle.STUDENT.value: "学生学习",
        ReaderStyle.ACADEMIC.value: "学术研究",
        ReaderStyle.BUSINESS.value: "企业决策",
    },
}

for _field in PreferenceField:
    PREFERENCE_OPTION_LABELS[_field][OTHER_OPTION] = "其他"


def option_label(field: PreferenceField, value: str) -> str:
    """Return the reader-facing label for one trusted option value."""

    return PREFERENCE_OPTION_LABELS[field].get(value, value)


class PreferenceQuestion(ProjectModel):
    """One clarification question with its bounded options and default."""

    field: PreferenceField
    question: str = Field(min_length=1, max_length=200)
    options: list[str] = Field(min_length=1, max_length=8)
    default_value: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _options_and_default_are_trusted(self) -> PreferenceQuestion:
        allowed = _ALLOWED_VALUES[self.field]
        if any(option not in allowed and option != OTHER_OPTION for option in self.options):
            raise ValueError(f"unsupported option for {self.field.value}")
        if len(set(self.options)) != len(self.options):
            raise ValueError("question options must be unique")
        if len(self.options) > 1 and self.options[-1] != OTHER_OPTION:
            raise ValueError("other option must be last")
        if self.default_value not in self.options:
            raise ValueError(f"unsupported default for {self.field.value}")
        return self


class PreferenceSelection(ProjectModel):
    """One accepted preference answer, including bounded free text."""

    field: PreferenceField
    value: str = Field(min_length=1, max_length=200)
    custom_text: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def _value_is_allowed(self) -> PreferenceSelection:
        if self.value == OTHER_OPTION:
            if self.custom_text is None or not self.custom_text.strip():
                raise ValueError("other option requires custom_text")
            return self
        if self.value not in _ALLOWED_VALUES[self.field]:
            raise ValueError(f"unsupported option for {self.field.value}")
        if self.custom_text is not None:
            raise ValueError("custom_text is only valid for other option")
        return self

    @property
    def resolved_text(self) -> str:
        """Return user text for ``other`` or the trusted machine value."""

        return self.custom_text.strip() if self.value == OTHER_OPTION else self.value


class DialogueModelOutput(ProjectModel):
    """Untrusted structured output of one dialogue model call.

    The model proposes an intent, prose and clarification questions only. It
    carries no session identity, so the program binds the decision before any
    routing happens.
    """

    intent: Intent
    reply: str = Field(default="", max_length=2000)
    report_subject: str = Field(default="", max_length=200)
    needs_clarification: bool = False
    questions: list[PreferenceQuestion] = Field(default_factory=list, max_length=2)
    preferences: list[PreferenceSelection] = Field(default_factory=list, max_length=2)
    confidence: float = Field(ge=0, le=1)
    reason_code: str = Field(min_length=1, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _normalize_model_nulls(cls, data: object) -> object:
        """Treat common JSON nulls as omitted optional response fields.

        Providers frequently emit ``null`` for fields that do not apply to an
        intent. The program-owned contract still stores canonical empty values
        so downstream routing never has to handle a second representation.
        """

        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field, empty in (
            ("reply", ""),
            ("report_subject", ""),
            ("questions", []),
            ("preferences", []),
        ):
            if normalized.get(field) is None:
                normalized[field] = empty
        # Some JSON-mode models express an empty collection as ``{}``.  It is
        # equivalent to no question or selection only when empty; non-empty
        # objects still fail the list-based contract below.
        for field in ("questions", "preferences"):
            if normalized.get(field) == {}:
                normalized[field] = []
        # The program owns the fixed clarification wording and options. Some
        # models nevertheless return prose questions as a list of strings;
        # retain only the model's clarification decision in that case.
        if (
            normalized.get("intent") == Intent.REPORT_REQUEST.value
            and normalized.get("needs_clarification") is True
            and isinstance(normalized.get("questions"), list)
            and all(isinstance(item, str) for item in normalized["questions"])
        ):
            normalized["questions"] = []
        return normalized

    @model_validator(mode="after")
    def _validate_intent_payload(self) -> DialogueModelOutput:
        question_fields = [question.field for question in self.questions]
        selection_fields = [selection.field for selection in self.preferences]
        if len(set(question_fields)) != len(question_fields):
            raise ValueError("question fields must be unique")
        if len(set(selection_fields)) != len(selection_fields):
            raise ValueError("preference fields must be unique")
        if self.intent is Intent.REPORT_REQUEST:
            if not self.needs_clarification and self.questions:
                raise ValueError("questions require clarification")
        elif self.questions or self.preferences:
            raise ValueError("non-report intent cannot carry preferences")
        return self

    def intent_decision(self, *, source: DecisionSource) -> IntentDecision:
        """Bind the validated output into the program-owned decision contract."""

        return IntentDecision(
            kind=self.intent,
            reason=self.reason_code,
            confidence=self.confidence,
            source=source,
        )


class ReportStartRequest(ProjectModel):
    """Handoff from the dialogue layer into the existing four-chapter graph."""

    run_id: str = Field(min_length=1, max_length=128)
    brief: ResearchBrief

    @model_validator(mode="after")
    def _run_id_matches_brief(self) -> ReportStartRequest:
        if self.brief.run_id != self.run_id:
            raise ValueError("report start run_id must match the brief run_id")
        return self


class SessionStatus(str, Enum):
    """Terminal state of one interactive teaching session."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SectionBrief(ProjectModel):
    """Short reader-facing summary of one delivered chapter."""

    section_id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=80)
    summary: str = Field(default="", max_length=600)


class SessionSummary(ProjectModel):
    """Structured output printed when a session ends; holds no credentials."""

    run_id: str = Field(min_length=1, max_length=128)
    status: SessionStatus
    phase: str = Field(default="", max_length=64)
    topic: str = Field(default="", max_length=200)
    report_markdown_ref: str = Field(default="", max_length=500)
    report_pdf_ref: str = Field(default="", max_length=500)
    trace_directory: str = Field(default="", max_length=500)
    token_usage_ref: str = Field(default="", max_length=500)
    service_log_directory: str = Field(default="", max_length=500)
    sections: list[SectionBrief] = Field(default_factory=list, max_length=8)
    error: str = Field(default="", max_length=64)
