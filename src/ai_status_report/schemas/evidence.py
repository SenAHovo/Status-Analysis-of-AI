"""Evidence retrieval bundles consumed by writing stages.

The bundle records what actually entered the model for one section, so a
replay can show the exact fragments, their provenance and any fragments that
were filtered out together with the reason.
"""

from __future__ import annotations

import hashlib
import re

from pydantic import Field

from ai_status_report.rag.index_versions import DEFAULT_INDEX_VERSION
from ai_status_report.schemas.common import ProjectModel, new_id


class Excerpt(ProjectModel):
    """One candidate fragment returned by retrieval."""

    chunk_id: str = Field(max_length=200)
    source_id: str = Field(default="", max_length=160)
    text: str = Field(min_length=1, max_length=8000)
    heading_path: str = Field(default="", max_length=500)
    page_or_offset: str = Field(default="", max_length=64)
    token_estimate: int = Field(default=0, ge=0)
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    provider: str = Field(default="", max_length=64)
    raw_ref: str = Field(default="", max_length=500)
    content_ref: str = Field(default="", max_length=500)
    verification_status: str = Field(default="", max_length=32)
    distance: float = Field(default=0.0, ge=0.0)
    retrieval_chunk_id: str = Field(default="", max_length=200)
    parent_chunk_id: str = Field(default="", max_length=160)


class EvidenceBundle(ProjectModel):
    """Actual fragments passed to the writer plus filtering context."""

    retrieval_id: str = Field(max_length=128)
    run_id: str = Field(max_length=128)
    section_id: str = Field(max_length=64)
    index_version: str = Field(default=DEFAULT_INDEX_VERSION, min_length=1, max_length=64)
    queries: list[str] = Field(default_factory=list)
    chunk_refs: list[str] = Field(default_factory=list)
    excerpts: list[Excerpt] = Field(default_factory=list)
    source_metadata: dict[str, str] = Field(default_factory=dict)
    quality_flags: list[str] = Field(default_factory=list)
    budget_used: int = Field(default=0, ge=0)

    @classmethod
    def new_retrieval_id(cls, run_id: str, section_id: str) -> str:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"retrieval-{run_id}-{section_id}").strip("-")
        if len(prefix) > 90:
            digest = hashlib.sha256(f"{run_id}\n{section_id}".encode()).hexdigest()[:24]
            prefix = f"retrieval-{digest}"
        return new_id(prefix)
