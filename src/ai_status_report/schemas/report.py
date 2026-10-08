"""Outline, section results, review issues and artifact references."""

from __future__ import annotations

from pydantic import Field, field_validator

from ai_status_report.schemas.common import (
    ProjectModel,
    validate_version,
)


class OutlineSection(ProjectModel):
    """One stable section of the report outline."""

    section_id: str = Field(max_length=64)
    title: str = Field(min_length=1, max_length=200)
    questions: list[str] = Field(default_factory=list)
    evidence_requirements: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    length_target: int | None = Field(default=None, ge=100)


class Outline(ProjectModel):
    """Versioned outline; section ids stay stable while titles may change."""

    outline_version: str = "1"
    sections: list[OutlineSection] = Field(min_length=1)

    @field_validator("outline_version")
    @classmethod
    def _outline_version(cls, value: str) -> str:
        return validate_version(value)

    def section(self, section_id: str) -> OutlineSection:
        for section in self.sections:
            if section.section_id == section_id:
                return section
        raise KeyError(section_id)


class SectionResult(ProjectModel):
    """Result of writing one section; body is persisted, summary is returned."""

    section_id: str = Field(max_length=64)
    draft_version: str
    artifact_ref: str | None = Field(default=None, max_length=128)
    summary: str = Field(default="", max_length=2000)
    claim_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    input_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("draft_version")
    @classmethod
    def _draft_version(cls, value: str) -> str:
        return validate_version(value)


class ReviewIssue(ProjectModel):
    """A review finding bound to one section draft version."""

    issue_id: str = Field(max_length=128)
    section_id: str = Field(max_length=64)
    draft_version: str
    location: str = Field(default="", max_length=300)
    type: str = Field(max_length=64)  # e.g. missing_evidence / structure / citation
    severity: str = Field(default="medium", pattern=r"^(low|medium|high)$")
    evidence_refs: list[str] = Field(default_factory=list)
    requested_change: str = Field(default="", max_length=2000)
    resolution: str | None = Field(default=None, max_length=64)

    @field_validator("draft_version")
    @classmethod
    def _draft_version(cls, value: str) -> str:
        return validate_version(value)


class ArtifactRef(ProjectModel):
    """Reference to a controlled artifact; access is resolved by a service."""

    artifact_id: str = Field(max_length=128)
    run_id: str = Field(max_length=128)
    version: str
    kind: str = Field(max_length=32)
    mime_type: str = Field(default="", max_length=64)
    size: int = Field(default=0, ge=0)
    hash: str = Field(default="", max_length=128)
    access_ref: str = Field(default="", max_length=300)
    input_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_version(value)

    @classmethod
    def new_artifact_id(cls, run_id: str, kind: str, version: str) -> str:
        return f"artifact-{run_id}-{kind}-{version}"
