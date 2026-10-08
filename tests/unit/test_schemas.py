"""Schemas: contracts, versions, identifiers and validation gates."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ai_status_report.schemas import (
    ArtifactRef,
    EvidenceBundle,
    Excerpt,
    Geoscope,
    LengthTarget,
    Outline,
    OutlineSection,
    PeriodSpec,
    ResearchBrief,
    ResearchJob,
    ReviewDecision,
    ReviewIssue,
    SectionResult,
)
from ai_status_report.schemas.common import (
    SCHEMA_VERSION,
    ProjectModel,
    new_id,
    next_version,
    validate_version,
)

NOW = datetime(2026, 9, 8, 2, 0, tzinfo=UTC)


def period(start="2025-09-08", end="2026-09-08") -> PeriodSpec:
    return PeriodSpec(
        start_date=datetime.fromisoformat(start).date(),
        end_date=datetime.fromisoformat(end).date(),
    )


def brief(**overrides) -> ResearchBrief:
    base = {
        "run_id": "run-1",
        "topic": "人工智能现状",
        "period": period(),
        "length_target": LengthTarget(min_words=4000, max_words=6000, label="4000-6000字"),
        "created_at": NOW,
    }
    base.update(overrides)
    return ResearchBrief(**base)


class ExtraModel(ProjectModel):
    pass


# --- common ---

def test_schema_version_present_on_every_contract():
    assert SCHEMA_VERSION == "1"
    assert brief().schema_version == SCHEMA_VERSION


def test_project_model_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ExtraModel(unknown="surprise")


def test_research_brief_requires_non_empty_run_id():
    with pytest.raises(ValidationError):
        brief(run_id="")


def test_id_and_version_helpers():
    first = new_id("run", now=NOW)
    second = new_id("run", now=NOW)
    assert first != second
    assert first.startswith("run-20260908T020000-")
    assert validate_version("12") == "12"
    with pytest.raises(ValueError):
        validate_version("v1")
    assert next_version("2") == "3"


def test_new_id_without_clock_is_utc_prefixed():
    value = new_id("evt")
    assert value.startswith("evt-")


# --- research ---

def test_period_spec_rejects_end_before_start():
    with pytest.raises(ValidationError):
        PeriodSpec(
            start_date=datetime.fromisoformat("2026-09-08").date(),
            end_date=datetime.fromisoformat("2025-01-01").date(),
        )


def test_length_target_rejects_inverted_range():
    with pytest.raises(ValidationError):
        LengthTarget(min_words=6000, max_words=4000, label="bad")


def test_research_brief_validators():
    with pytest.raises(ValidationError):
        brief(brief_version="1.0")
    with pytest.raises(ValidationError):
        brief(topic="a\u0001b")
    # created_at must be timezone aware; naive stamps are rejected.
    naive = datetime.fromisoformat("2026-09-08T02:00:00")
    with pytest.raises(ValidationError):
        brief(created_at=naive)
    assert brief().audience.value == "general"
    assert brief(geography=Geoscope.CHINA).region == ""


def test_research_job_requires_brief_version():
    job = ResearchJob(
        job_id="job-1",
        run_id="run-1",
        brief_version="1",
        topic_id="t1",
        section_ids=["s1"],
        question="待验证主张",
        source_types=["paper"],
        period=period(),
        limits={"max_results": 5},
    )
    assert job.topic_id == "t1"
    with pytest.raises(ValidationError):
        ResearchJob(
            job_id="job-2",
            run_id="run-1",
            brief_version="bad",
            topic_id="t1",
            question="x",
            period=period(),
        )


# --- outline / sections ---

def test_outline_lookup_and_version():
    outline = Outline(
        sections=[OutlineSection(section_id="s1", title="背景", questions=["q1"])],
        outline_version="1",
    )
    assert outline.section("s1").title == "背景"
    with pytest.raises(KeyError):
        outline.section("missing")
    with pytest.raises(ValidationError):
        Outline(sections=[], outline_version="1")


def test_section_result_and_review_issue_versions():
    result = SectionResult(section_id="s1", draft_version="3", summary="ok")
    assert result.draft_version == "3"
    with pytest.raises(ValidationError):
        SectionResult(section_id="s1", draft_version="x", summary="ok")

    issue = ReviewIssue(
        issue_id="i1", section_id="s1", draft_version="3", type="missing_evidence"
    )
    assert issue.severity == "medium"
    with pytest.raises(ValidationError):
        ReviewIssue(issue_id="i2", section_id="s1", draft_version="3", severity="fatal", type="x")


def test_review_decision_maps_structured_gaps_to_bounded_supplement_queries():
    decision = ReviewDecision.from_document_result(
        run_id="run-1",
        section_id="s1",
        draft_version="1",
        evidence_gaps=[
                {
                    "gap_id": "adoption",
                    "kind": "content_issue",
                    "severity": "hard",
                "description": "缺少采用率数据",
                "suggested_queries": ["智能体采用率", "智能体采用率"],
            }
        ],
    )

    assert decision.decision == "need_evidence"
    assert decision.issues[0].severity == "high"
    assert decision.supplement_queries == ["智能体采用率"]

    approved = ReviewDecision.from_document_result(
        run_id="run-1", section_id="s1", draft_version="1", evidence_gaps=[]
    )
    assert approved.decision == "approve"


def test_artifact_ref_validation():
    ref = ArtifactRef(
        artifact_id=ArtifactRef.new_artifact_id("run-1", "chapter", "1"),
        run_id="run-1",
        version="1",
        kind="chapter",
        access_ref="outputs/run-1/chapters/s1/v001.md",
    )
    assert ref.artifact_id.startswith("artifact-")
    with pytest.raises(ValidationError):
        ArtifactRef(artifact_id="a", run_id="r", version="two", kind="chapter")


# --- evidence ---

def test_evidence_bundle_and_excerpt():
    bundle = EvidenceBundle(
        retrieval_id=EvidenceBundle.new_retrieval_id("run-1", "s1"),
        run_id="run-1",
        section_id="s1",
        excerpts=[Excerpt(chunk_id="c1", source_id="src1", text="官方报告原文")],
        budget_used=120,
    )
    assert bundle.retrieval_id.startswith("retrieval-")
    assert len(bundle.excerpts) == 1
    with pytest.raises(ValidationError):
        Excerpt(chunk_id="c1", source_id="src1", text="")
