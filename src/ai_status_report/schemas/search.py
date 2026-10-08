"""Provider-neutral search result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from ai_status_report.schemas.common import ProjectModel, new_id


class ResearchSource(ProjectModel):
    source_id: str = Field(min_length=1, max_length=160)
    provider: str = Field(min_length=1, max_length=64)
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    snippet: str = Field(default="", max_length=10000)
    published_at: str | None = Field(default=None, max_length=128)
    retrieved_at: datetime
    raw_ref: str = Field(default="", max_length=500)
    content_ref: str = Field(default="", max_length=500)
    verification_status: Literal["reported", "retrieved", "verified"] = "reported"


class SearchQuery(ProjectModel):
    """One bounded query variant produced by the Search Agent planner."""

    query: str = Field(min_length=1, max_length=1000)
    language: str = Field(default="", max_length=32)
    region: str = Field(default="", max_length=64)
    purpose: str = Field(default="", max_length=500)


class SearchSourceRequirements(ProjectModel):
    preferred_types: list[str] = Field(default_factory=list, max_length=8)
    min_sources: int = Field(default=1, ge=1, le=50)


class SearchLimits(ProjectModel):
    max_queries: int = Field(default=4, ge=1, le=4)
    max_results_per_query: int = Field(default=5, ge=1, le=10)
    max_output_tokens: int = Field(default=65536, ge=1024, le=384000)
    max_parallel_providers: int = Field(default=2, ge=1, le=2)


class SearchPlan(ProjectModel):
    """Provider-neutral execution plan for a structured research search."""

    plan_id: str = Field(default_factory=lambda: new_id("plan"), min_length=1, max_length=160)
    original_query: str = Field(default="", max_length=1000)
    queries: list[SearchQuery] = Field(min_length=1, max_length=4)
    providers: list[Literal["tavily", "deepseek"]] = Field(min_length=1, max_length=2)
    # A downstream requirement. ``retrievable`` requires captured page text
    # that can become EvidenceChunk records and enter the RAG index.
    evidence_requirement: Literal["none", "retrievable"] = "none"
    source_requirements: SearchSourceRequirements = Field(default_factory=SearchSourceRequirements)
    limits: SearchLimits = Field(default_factory=SearchLimits)
    need_cross_validation: bool = False
    reason: str = Field(default="", max_length=1000)
    source: Literal["model", "model_cache", "deterministic", "deterministic_fallback"] = (
        "deterministic"
    )
    validation_warnings: list[str] = Field(default_factory=list, max_length=16)


class SearchRouteDecision(ProjectModel):
    """Minimal model output used to select one search provider."""

    provider: Literal["tavily", "deepseek"]
    reason: str = Field(default="", max_length=1000)


class ResearchFinding(ProjectModel):
    """A bounded conclusion that can be traced to report sources."""

    finding_id: str = Field(min_length=1, max_length=160)
    claim: str = Field(min_length=1, max_length=1000)
    explanation: str = Field(default="", max_length=3000)
    source_ids: list[str] = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0.0, le=1.0)


class ResearchSynthesis(ProjectModel):
    """Model output used to turn provider results into a compact handoff."""

    summary: str = Field(default="", max_length=12000)
    key_findings: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    gaps: list[str] = Field(default_factory=list, max_length=20)


class ResearchReport(ProjectModel):
    schema_version: str = "1"
    query: str = Field(min_length=1, max_length=1000)
    provider: str = Field(min_length=1, max_length=64)
    providers: list[str] = Field(default_factory=list, max_length=2)
    plan_id: str = Field(default="", max_length=160)
    plan_source: str = Field(default="", max_length=32)
    summary: str = Field(default="", max_length=12000)
    key_findings: list[ResearchFinding] = Field(default_factory=list, max_length=12)
    status: Literal["completed", "partial", "failed"] = "partial"
    incomplete_reason: str | None = Field(default=None, max_length=256)
    sources: list[ResearchSource] = Field(default_factory=list, max_length=50)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    raw_ref: str = Field(default="", max_length=500)
    raw_refs: list[str] = Field(default_factory=list, max_length=10)
    gaps: list[str] = Field(default_factory=list, max_length=20)


class EvidenceChunk(ProjectModel):
    """A durable, provider-neutral excerpt ready for later RAG indexing."""

    chunk_id: str = Field(min_length=1, max_length=160)
    source_id: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=8000)
    title: str = Field(default="", max_length=500)
    url: str = Field(default="", max_length=2000)
    provider: str = Field(min_length=1, max_length=64)
    position: int = Field(default=0, ge=0)
    content_hash: str = Field(min_length=1, max_length=128)
    raw_ref: str = Field(default="", max_length=500)
    content_ref: str = Field(default="", max_length=500)
    created_at: datetime
    verification_status: Literal["retrieved", "verified"]
