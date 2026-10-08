"""Research task contracts: brief, job and period bounds.

The normalized :class:`ResearchBrief` is the shared constraint for every later
stage. Fields keep an explicit origin so the user can always see what came
from their request, what was filled with project defaults, and what changed.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import Field, field_validator

from ai_status_report.schemas.common import (
    ProjectModel,
    new_id,
    validate_version,
)


class OriginKind(str, Enum):
    """Where a brief field value came from."""

    USER = "user"
    DEFAULT = "default"
    UPDATED = "updated"


class ReaderStyle(str, Enum):
    """Target reader; affects emphasis and depth, never evidence standards."""

    GENERAL = "general"
    STUDENT = "student"
    ACADEMIC = "academic"
    BUSINESS = "business"


class Geoscope(str, Enum):
    """Geography covered by the report."""

    GLOBAL_WITH_CHINA = "global_with_china"  # 全球并兼顾中国
    GLOBAL = "global"
    CHINA = "china"


class WritingStyle(str, Enum):
    """Requested writing style for the report body (ADR-0005 preference)."""

    OBJECTIVE = "objective"  # 公正客观
    ANALYTICAL = "analytical"  # 深度分析
    CONCISE = "concise"  # 简明扼要
    PRACTICAL = "practical"  # 务实应用


# Single source of truth for the brief-level preference defaults: the dialogue
# layer and the brief normalizer both read these instead of repeating values.
DEFAULT_WRITING_STYLE = WritingStyle.OBJECTIVE
DEFAULT_READER_STYLE = ReaderStyle.GENERAL
DEFAULT_LENGTH_RANGE = (4000, 6000)


class PeriodSpec(ProjectModel):
    """Fixed research window written into the brief at run start."""

    start_date: date
    end_date: date
    relative_expression: str = Field(default="", max_length=80)
    zone: str = Field(default="Asia/Shanghai", max_length=64)

    @field_validator("end_date")
    @classmethod
    def _end_not_before_start(cls, value: date, info) -> date:
        start = info.data.get("start_date")
        if start is not None and value < start:
            raise ValueError("period end precedes start")
        return value


class DefaultedField(ProjectModel):
    """One resolved field and its origin for display and tracing."""

    field: str = Field(max_length=64)
    value: str = Field(max_length=500)
    origin: OriginKind


class LengthTarget(ProjectModel):
    """Words/characters target; the suggestion is configurable, not a limit."""

    min_words: int = Field(ge=200)
    max_words: int = Field(ge=200)
    label: str = Field(default="", max_length=64)

    @field_validator("max_words")
    @classmethod
    def _max_not_below_min(cls, value: int, info) -> int:
        minimum = info.data.get("min_words")
        if minimum is not None and value < minimum:
            raise ValueError("max_words below min_words")
        return value


class ResearchBrief(ProjectModel):
    """Normalized brief shared by controller, research and document stages.

    ``defaults_applied`` records every resolved field that did not come from
    the user, keeping the choice traceable without turning the request into a
    required form.
    """

    run_id: str = Field(min_length=1, max_length=128)
    brief_version: str = "1"
    topic: str = Field(min_length=1, max_length=200)
    audience: ReaderStyle = DEFAULT_READER_STYLE
    custom_reader: str | None = Field(default=None, max_length=300)
    writing_style: WritingStyle = DEFAULT_WRITING_STYLE
    custom_writing_style: str | None = Field(default=None, max_length=300)
    custom_usage_scenario: str | None = Field(default=None, max_length=300)
    geography: Geoscope = Geoscope.GLOBAL_WITH_CHINA
    region: str = Field(default="", max_length=120)
    period: PeriodSpec
    language: str = Field(default="zh", pattern=r"^[a-z]{2,3}$")
    length_target: LengthTarget
    report_structure: list[str] = Field(default_factory=list, max_length=20)
    output_format: str = Field(default="pdf", max_length=16)
    defaults_applied: list[DefaultedField] = Field(default_factory=list)
    created_at: datetime

    @field_validator("brief_version")
    @classmethod
    def _brief_version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("created_at")
    @classmethod
    def _created_at_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("created_at must carry a timezone")
        return value

    @field_validator("topic", "region")
    @classmethod
    def _no_control_chars(cls, value: str) -> str:
        if any(ord(ch) < 32 for ch in value):
            raise ValueError("control characters are not allowed")
        return value

    @classmethod
    def new_run_id(cls, now: datetime | None = None) -> str:
        return new_id("run", now=now)


class ResearchJob(ProjectModel):
    """One parallel research unit; topic is the division, source is coverage."""

    job_id: str = Field(max_length=128)
    run_id: str = Field(max_length=128)
    brief_version: str
    topic_id: str = Field(max_length=64)
    section_ids: list[str] = Field(default_factory=list)
    question: str = Field(min_length=1, max_length=1000)
    source_types: list[str] = Field(default_factory=list)
    period: PeriodSpec
    limits: dict[str, int] = Field(default_factory=dict)

    @field_validator("brief_version")
    @classmethod
    def _brief_version(cls, value: str) -> str:
        return validate_version(value)
