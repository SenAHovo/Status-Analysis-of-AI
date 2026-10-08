"""Business contracts for document-writing A2A tasks and artifacts.

These models are the JSON payload carried inside an A2A text message.  The
A2A SDK owns protocol task identifiers, Task state transitions and Artifact
events; this module owns only the project's writing inputs and outcomes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from ai_status_report.schemas.common import ProjectModel, validate_version
from ai_status_report.schemas.report import ArtifactRef, ReviewIssue

_MAX_TASK_TEXT_ITEM_CHARS = 1000
_MAX_GAP_CHARS = 1000
_MAX_QUALITY_FLAG_CHARS = 256
_MAX_SOURCE_ID_CHARS = 160
_MAX_CHUNK_ID_CHARS = 160
_MAX_GAP_SOURCE_TYPES = 5
_MAX_GAP_QUERIES = 5


def _validate_text_items(
    values: list[str], *, label: str, max_item_chars: int
) -> list[str]:
    """Reject blank or oversized list items before they enter an A2A payload."""

    if any(not value.strip() or len(value) > max_item_chars for value in values):
        raise ValueError(f"invalid {label}")
    return values


class DocumentBudget(ProjectModel):
    """Explicit limits supplied by the controller for one chapter operation."""

    context_window_tokens: int = Field(ge=1024, le=1_000_000)
    evidence_tokens: int = Field(ge=100, le=1_000_000)
    reserved_input_tokens: int = Field(ge=0, le=1_000_000)
    max_output_tokens: int = Field(ge=256, le=1_000_000)

    @model_validator(mode="after")
    def _total_fits_window(self) -> DocumentBudget:
        total = self.evidence_tokens + self.reserved_input_tokens + self.max_output_tokens
        if total > self.context_window_tokens:
            raise ValueError("chapter budget exceeds the context window")
        return self


class RecentSectionSummary(ProjectModel):
    """A compact, approved predecessor summary admitted to a chapter context."""

    section_id: str = Field(min_length=1, max_length=64)
    approved_version: str
    summary: str = Field(min_length=1, max_length=2000)
    artifact_ref: str = Field(min_length=1, max_length=500)

    @field_validator("approved_version")
    @classmethod
    def _approved_version(cls, value: str) -> str:
        return validate_version(value)


class EvidenceGap(ProjectModel):
    """A content-evidence deficiency for the controller's future review loop."""

    gap_id: str = Field(min_length=1, max_length=80)
    kind: Literal[
        "evidence_limitation",
        "content_issue",
        "style_issue",
        "structure_issue",
        "citation_issue",
    ] = "evidence_limitation"
    severity: Literal["hard", "soft"]
    description: str = Field(min_length=1, max_length=_MAX_GAP_CHARS)
    preferred_source_types: list[str] = Field(default_factory=list, max_length=_MAX_GAP_SOURCE_TYPES)
    suggested_queries: list[str] = Field(default_factory=list, max_length=_MAX_GAP_QUERIES)
    resolution_status: Literal["open", "resolved", "accepted_limit"] = "open"
    attempt_count: int = Field(default=0, ge=0, le=2)

    @field_validator("preferred_source_types")
    @classmethod
    def _source_types(cls, values: list[str]) -> list[str]:
        return _validate_text_items(values, label="gap source type", max_item_chars=80)

    @field_validator("suggested_queries")
    @classmethod
    def _suggested_queries(cls, values: list[str]) -> list[str]:
        return _validate_text_items(
            values, label="gap suggested query", max_item_chars=_MAX_TASK_TEXT_ITEM_CHARS
        )


class DocumentWriteSectionTask(ProjectModel):
    """Controller-to-document-agent request to draft one report section."""

    task_type: Literal["document.write_section"] = "document.write_section"
    run_id: str = Field(min_length=1, max_length=128)
    research_report_ref: str = Field(min_length=1, max_length=500)
    research_summary: str = Field(min_length=1, max_length=12000)
    outline_ref: str = Field(min_length=1, max_length=500)
    outline_version: str
    section_id: str = Field(min_length=1, max_length=64)
    section_title: str = Field(min_length=1, max_length=200)
    section_goal: str = Field(min_length=1, max_length=2000)
    retrieval_queries: list[str] = Field(min_length=1, max_length=8)
    evidence_requirements: list[str] = Field(default_factory=list, max_length=12)
    writing_constraints: list[str] = Field(default_factory=list, max_length=12)
    writing_template_version: str = "1"
    target_draft_version: str = "1"
    recent_section_summaries: list[RecentSectionSummary] = Field(default_factory=list, max_length=5)
    citation_style: Literal["reader_numeric"] = "reader_numeric"
    budget: DocumentBudget

    @field_validator("outline_version", "writing_template_version", "target_draft_version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("retrieval_queries")
    @classmethod
    def _queries(cls, values: list[str]) -> list[str]:
        return _validate_text_items(
            values, label="retrieval query", max_item_chars=_MAX_TASK_TEXT_ITEM_CHARS
        )

    @field_validator("evidence_requirements", "writing_constraints")
    @classmethod
    def _task_text_items(cls, values: list[str]) -> list[str]:
        return _validate_text_items(
            values, label="task text item", max_item_chars=_MAX_TASK_TEXT_ITEM_CHARS
        )

    @model_validator(mode="after")
    def _first_draft_is_v1(self) -> DocumentWriteSectionTask:
        if self.target_draft_version != "1":
            raise ValueError("initial section draft version must be 1")
        if any(summary.section_id == self.section_id for summary in self.recent_section_summaries):
            raise ValueError("recent summaries cannot include the current section")
        return self


class DocumentReviseSectionTask(ProjectModel):
    """Controller-to-document-agent request to produce a new chapter version."""

    task_type: Literal["document.revise_section"] = "document.revise_section"
    run_id: str = Field(min_length=1, max_length=128)
    research_report_ref: str = Field(min_length=1, max_length=500)
    outline_ref: str = Field(min_length=1, max_length=500)
    outline_version: str
    section_id: str = Field(min_length=1, max_length=64)
    section_title: str = Field(min_length=1, max_length=200)
    base_draft: ArtifactRef
    target_draft_version: str
    review_issues: list[ReviewIssue] = Field(min_length=1, max_length=20)
    retrieval_queries: list[str] = Field(default_factory=list, max_length=8)
    writing_constraints: list[str] = Field(default_factory=list, max_length=12)
    writing_template_version: str = "1"
    budget: DocumentBudget

    @field_validator("outline_version", "target_draft_version", "writing_template_version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("retrieval_queries", "writing_constraints")
    @classmethod
    def _text_items(cls, values: list[str], info) -> list[str]:
        return _validate_text_items(
            values,
            label="retrieval query" if info.field_name == "retrieval_queries" else "writing constraint",
            max_item_chars=_MAX_TASK_TEXT_ITEM_CHARS,
        )

    @model_validator(mode="after")
    def _revision_matches_its_base(self) -> DocumentReviseSectionTask:
        if self.base_draft.run_id != self.run_id:
            raise ValueError("base draft run does not match revision run")
        if (
            self.base_draft.kind != "chapter_markdown"
            or self.base_draft.mime_type != "text/markdown"
            or not self.base_draft.access_ref
        ):
            raise ValueError("base draft must be an accessible chapter_markdown artifact")
        if int(self.target_draft_version) <= int(self.base_draft.version):
            raise ValueError("revision version must exceed the base draft version")
        if any(
            issue.section_id != self.section_id or issue.draft_version != self.base_draft.version
            for issue in self.review_issues
        ):
            raise ValueError("review issue does not match the base draft")
        return self


class DocumentSectionResult(ProjectModel):
    """Document-agent business outcome returned in one A2A Artifact."""

    task_type: Literal["document.write_section", "document.revise_section"]
    status: Literal["drafted", "revised", "blocked", "failed"]
    run_id: str = Field(min_length=1, max_length=128)
    section_id: str = Field(min_length=1, max_length=64)
    draft_version: str
    draft_artifact: ArtifactRef | None = None
    section_summary: str = Field(default="", max_length=2000)
    evidence_bundle_ref: str = Field(default="", max_length=500)
    citation_map_ref: str = Field(default="", max_length=500)
    used_source_ids: list[str] = Field(default_factory=list, max_length=50)
    used_chunk_ids: list[str] = Field(default_factory=list, max_length=100)
    gaps: list[str] = Field(default_factory=list, max_length=20)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list, max_length=20)
    quality_flags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("draft_version")
    @classmethod
    def _draft_version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("used_source_ids")
    @classmethod
    def _source_ids(cls, values: list[str]) -> list[str]:
        return _validate_text_items(values, label="source id", max_item_chars=_MAX_SOURCE_ID_CHARS)

    @field_validator("used_chunk_ids")
    @classmethod
    def _chunk_ids(cls, values: list[str]) -> list[str]:
        return _validate_text_items(values, label="chunk id", max_item_chars=_MAX_CHUNK_ID_CHARS)

    @field_validator("gaps")
    @classmethod
    def _gaps(cls, values: list[str]) -> list[str]:
        return _validate_text_items(values, label="gap", max_item_chars=_MAX_GAP_CHARS)

    @field_validator("quality_flags")
    @classmethod
    def _quality_flags(cls, values: list[str]) -> list[str]:
        return _validate_text_items(
            values, label="quality flag", max_item_chars=_MAX_QUALITY_FLAG_CHARS
        )

    @model_validator(mode="after")
    def _artifact_matches_successful_result(self) -> DocumentSectionResult:
        success = self.status in {"drafted", "revised"}
        if success and (
            self.draft_artifact is None
            or not self.evidence_bundle_ref
            or not self.citation_map_ref
        ):
            raise ValueError(
                "successful section result requires draft artifact, evidence bundle and citation map"
            )
        if self.draft_artifact is not None:
            if self.draft_artifact.run_id != self.run_id:
                raise ValueError("draft artifact run does not match result run")
            if self.draft_artifact.version != self.draft_version:
                raise ValueError("draft artifact version does not match result version")
            if self.draft_artifact.kind != "chapter_markdown" or not self.draft_artifact.access_ref:
                raise ValueError("draft artifact must be an accessible chapter_markdown artifact")
        if self.status == "drafted" and self.task_type != "document.write_section":
            raise ValueError("drafted status requires a write task")
        if self.status == "revised" and self.task_type != "document.revise_section":
            raise ValueError("revised status requires a revise task")
        if self.status == "drafted" and self.draft_version != "1":
            raise ValueError("drafted status requires initial draft version 1")
        return self
