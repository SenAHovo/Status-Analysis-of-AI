"""Contracts for the retrieval layer between EvidenceChunk and Chroma."""

from __future__ import annotations

from pydantic import Field

from ai_status_report.schemas.common import ProjectModel


class RetrievalChunk(ProjectModel):
    """A bounded embedding unit that retains its parent evidence reference."""

    retrieval_chunk_id: str = Field(min_length=1, max_length=200)
    evidence_chunk_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=128)
    source_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=1000)
    position: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    provider: str = Field(min_length=1, max_length=64)
    raw_ref: str = Field(default="", max_length=500)
    content_ref: str = Field(default="", max_length=500)
    verification_status: str = Field(min_length=1, max_length=32)
    index_version: str = Field(min_length=1, max_length=64)


class RetrievalMatch(ProjectModel):
    """A Chroma result returned to the future EvidenceBundle builder."""

    retrieval_chunk_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=1000)
    distance: float = Field(ge=0.0)
    evidence_chunk_id: str = Field(min_length=1, max_length=160)
    source_id: str = Field(min_length=1, max_length=160)
    run_id: str = Field(min_length=1, max_length=128)
    title: str = Field(default="", max_length=500)
    provider: str = Field(default="", max_length=64)
    url: str = Field(default="", max_length=2000)
    raw_ref: str = Field(default="", max_length=500)
    content_ref: str = Field(default="", max_length=500)
    verification_status: str = Field(min_length=1, max_length=32)
    position: int = Field(default=0, ge=0)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    rerank_score: float | None = Field(default=None, ge=0.0)
