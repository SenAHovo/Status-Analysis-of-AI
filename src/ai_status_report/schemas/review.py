"""Controller-side review decisions for the chapter evidence loop."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from ai_status_report.schemas.common import ProjectModel, validate_version
from ai_status_report.schemas.report import ReviewIssue


class SupplementQuery(ProjectModel):
    """One bounded, gap-specific query the controller can send to Search Agent."""

    gap_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=1000)
    preferred_source_types: list[str] = Field(default_factory=list, max_length=5)


class SupplementPlan(ProjectModel):
    """Controller-internal plan for a single bounded evidence supplement round."""

    run_id: str = Field(min_length=1, max_length=128)
    section_id: str = Field(min_length=1, max_length=64)
    draft_version: str
    supplement_round: int = Field(ge=1, le=2)
    queries: list[SupplementQuery] = Field(min_length=1, max_length=5)
    source: Literal["model", "deterministic", "mixed"]

    @field_validator("draft_version")
    @classmethod
    def _draft_version(cls, value: str) -> str:
        return validate_version(value)


class ReviewDecision(ProjectModel):
    """A bounded controller decision after one document-agent delivery."""

    decision: Literal["approve", "deliver_with_limits", "need_evidence", "need_revision"]
    run_id: str = Field(min_length=1, max_length=128)
    section_id: str = Field(min_length=1, max_length=64)
    draft_version: str
    review_round: int = Field(default=0, ge=0, le=2)
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=20)
    supplement_queries: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=2000)
    source: Literal["deterministic", "model", "model_cache", "deterministic_fallback"] = (
        "deterministic"
    )
    fallback_reason: str = Field(default="", max_length=128)

    @field_validator("draft_version")
    @classmethod
    def _draft_version(cls, value: str) -> str:
        return validate_version(value)

    @field_validator("supplement_queries")
    @classmethod
    def _queries(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("invalid supplement query")
        return list(dict.fromkeys(values))

    @classmethod
    def from_document_result(
        cls,
        *,
        run_id: str,
        section_id: str,
        draft_version: str,
        evidence_gaps: list[dict],
        review_round: int = 0,
    ) -> ReviewDecision:
        """Convert structured document gaps into a deterministic first decision."""

        issues: list[ReviewIssue] = []
        queries: list[str] = []
        for index, gap in enumerate(evidence_gaps, start=1):
            gap_id = str(gap.get("gap_id") or f"gap-{index}")
            issue_type = "missing_evidence"
            severity = "high" if gap.get("severity") == "hard" else "medium"
            issues.append(
                ReviewIssue(
                    issue_id=f"review-{gap_id}",
                    section_id=section_id,
                    draft_version=draft_version,
                    type=issue_type,
                    severity=severity,
                    requested_change=str(gap.get("description") or "补充可追溯证据"),
                )
            )
            for query in gap.get("suggested_queries") or []:
                if isinstance(query, str) and query.strip() and query not in queries:
                    queries.append(query)
        if not issues:
            return cls(
                decision="approve",
                run_id=run_id,
                section_id=section_id,
                draft_version=draft_version,
                review_round=review_round,
                reason="章节成果没有交付结构化证据缺口",
            )
        has_blocking_gap = any(
            gap.get("severity") == "hard"
            and gap.get("kind", "evidence_limitation") != "evidence_limitation"
            for gap in evidence_gaps
        )
        has_revision_gap = any(
            gap.get("kind") in {"content_issue", "style_issue", "structure_issue", "citation_issue"}
            for gap in evidence_gaps
        )
        if has_blocking_gap and queries:
            decision = "need_evidence"
        elif has_revision_gap:
            decision = "need_revision"
        else:
            decision = "deliver_with_limits"
        return cls(
            decision=decision,
            run_id=run_id,
            section_id=section_id,
            draft_version=draft_version,
            review_round=review_round,
            issues=issues,
            supplement_queries=queries[:5],
            reason=(
                "存在可执行的核心证据缺口查询"
                if decision == "need_evidence"
                else "存在缺口，但没有可执行的补证查询"
            ),
        )


class EvidenceReviewOutput(ProjectModel):
    """Model-only review fields before the controller binds task identity."""

    decision: Literal["approve", "deliver_with_limits", "need_evidence", "need_revision"]
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=20)
    supplement_queries: list[str] = Field(default_factory=list, max_length=5)
    reason: str = Field(default="", max_length=2000)

    @field_validator("supplement_queries")
    @classmethod
    def _queries(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 1000 for value in values):
            raise ValueError("invalid supplement query")
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def _decision_matches_findings(self) -> EvidenceReviewOutput:
        if self.decision == "approve" and (self.issues or self.supplement_queries):
            raise ValueError("approved review cannot contain findings")
        if self.decision == "deliver_with_limits" and self.supplement_queries:
            raise ValueError("limited delivery cannot require supplement queries")
        if self.decision == "need_evidence" and (not self.issues or not self.supplement_queries):
            raise ValueError("evidence review requires findings and queries")
        if self.decision == "need_revision" and not self.issues:
            raise ValueError("revision review requires findings")
        return self

    def bind(
        self,
        *,
        run_id: str,
        section_id: str,
        draft_version: str,
        review_round: int,
        source: Literal["model", "model_cache"],
    ) -> ReviewDecision:
        """Bind untrusted model findings to the controller's dispatched chapter."""

        if any(
            issue.section_id != section_id or issue.draft_version != draft_version
            for issue in self.issues
        ):
            raise ValueError("review_issue_identity_mismatch")
        return ReviewDecision(
            decision=self.decision,
            run_id=run_id,
            section_id=section_id,
            draft_version=draft_version,
            review_round=review_round,
            issues=self.issues,
            supplement_queries=self.supplement_queries,
            reason=self.reason,
            source=source,
        )
