"""Chapter-writing task and result contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ai_status_report.harness.graph_helpers import document_revise_task
from ai_status_report.schemas import (
    ArtifactRef,
    DocumentBudget,
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
    RecentSectionSummary,
    ReviewIssue,
)
from ai_status_report.schemas.review import ReviewDecision


def budget() -> DocumentBudget:
    return DocumentBudget(
        context_window_tokens=16000,
        evidence_tokens=4000,
        reserved_input_tokens=3000,
        max_output_tokens=3000,
    )


def chapter_artifact(version: str = "1", run_id: str = "run-doc-001") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactRef.new_artifact_id(run_id, "chapter_markdown", version),
        run_id=run_id,
        version=version,
        kind="chapter_markdown",
        mime_type="text/markdown",
        access_ref=f"data/runs/{run_id}/chapters/s1/v{version.zfill(3)}.md",
    )


def write_task(**overrides: object) -> DocumentWriteSectionTask:
    payload: dict[str, object] = {
        "run_id": "run-doc-001",
        "research_report_ref": "data/runs/run-doc-001/reports/research.json",
        "research_summary": "研究资料表明智能体交互涵盖聊天、GUI 和人类监督。",
        "outline_ref": "data/runs/run-doc-001/outlines/v001.json",
        "outline_version": "1",
        "section_id": "s1",
        "section_title": "智能体交互方式",
        "section_goal": "说明主要交互方式及其发展方向。",
        "retrieval_queries": ["智能体交互方式"],
        "budget": budget(),
    }
    payload.update(overrides)
    return DocumentWriteSectionTask(**payload)


def test_write_section_task_carries_only_compact_handoffs() -> None:
    task = write_task(
        recent_section_summaries=[
            RecentSectionSummary(
                section_id="s0",
                approved_version="2",
                summary="前章说明了智能体定义。",
                artifact_ref="data/runs/run-doc-001/chapters/s0/v002.md",
            )
        ]
    )

    assert task.task_type == "document.write_section"
    assert task.target_draft_version == "1"
    assert task.recent_section_summaries[0].approved_version == "2"


def test_write_section_task_rejects_non_initial_version_and_current_summary() -> None:
    with pytest.raises(ValidationError, match="initial section draft version"):
        write_task(target_draft_version="2")
    with pytest.raises(ValidationError, match="current section"):
        write_task(
            recent_section_summaries=[
                RecentSectionSummary(
                    section_id="s1",
                    approved_version="1",
                    summary="不应包含当前章节。",
                    artifact_ref="data/runs/run-doc-001/chapters/s1/v001.md",
                )
            ]
        )


def test_document_budget_and_task_text_items_stay_within_bounded_payloads() -> None:
    with pytest.raises(ValidationError, match="exceeds the context window"):
        DocumentBudget(
            context_window_tokens=16000,
            evidence_tokens=12000,
            reserved_input_tokens=1000,
            max_output_tokens=4000,
        )
    with pytest.raises(ValidationError, match="invalid task text item"):
        write_task(writing_constraints=["x" * 1001])
    with pytest.raises(ValidationError, match="invalid task text item"):
        write_task(evidence_requirements=["   "])


def test_document_budget_accepts_the_configured_one_million_token_window() -> None:
    budget = DocumentBudget(
        context_window_tokens=1_000_000,
        evidence_tokens=800_000,
        reserved_input_tokens=32_000,
        max_output_tokens=64_000,
    )

    assert budget.context_window_tokens == 1_000_000


def test_revision_task_requires_matching_issue_and_newer_version() -> None:
    issue = ReviewIssue(
        issue_id="issue-1",
        section_id="s1",
        draft_version="1",
        type="citation",
        requested_change="为关键结论补充证据引用。",
    )
    task = DocumentReviseSectionTask(
        run_id="run-doc-001",
        research_report_ref="data/runs/run-doc-001/reports/research.json",
        outline_ref="data/runs/run-doc-001/outlines/v001.json",
        outline_version="1",
        section_id="s1",
        section_title="智能体交互方式",
        base_draft=chapter_artifact(),
        target_draft_version="2",
        review_issues=[issue],
        budget=budget(),
    )
    assert task.task_type == "document.revise_section"
    assert task.target_draft_version == "2"

    invalid_version = task.model_dump()
    invalid_version["target_draft_version"] = "1"
    with pytest.raises(ValidationError, match="revision version"):
        DocumentReviseSectionTask(**invalid_version)
    invalid_issue = task.model_dump()
    invalid_issue["review_issues"] = [issue.model_copy(update={"section_id": "wrong-section"})]
    with pytest.raises(ValidationError, match="review issue"):
        DocumentReviseSectionTask(**invalid_issue)


def test_revision_task_inherits_the_initial_writing_preferences() -> None:
    issue = ReviewIssue(
        issue_id="issue-style",
        section_id="s1",
        draft_version="1",
        type="style",
        requested_change="保持学生教学场景并增强分析性。",
    )
    initial_task = write_task(
        writing_constraints=["写作风格：analytical", "使用场景：student"]
    )
    result = DocumentSectionResult(
        task_type="document.write_section",
        status="drafted",
        run_id="run-doc-001",
        section_id="s1",
        draft_version="1",
        draft_artifact=chapter_artifact(),
        section_summary="章节摘要。",
        evidence_bundle_ref="data/runs/run-doc-001/evidence_bundles/s1.json",
        citation_map_ref="data/runs/run-doc-001/chapters/s1/v001.citations.json",
    )
    decision = ReviewDecision(
        decision="need_revision",
        run_id="run-doc-001",
        section_id="s1",
        draft_version="1",
        issues=[issue],
        reason="需要修订表达。",
    )

    revision = document_revise_task(
        {
            "document_task": initial_task.model_dump(mode="json"),
            "document_result": result.model_dump(mode="json"),
            "review_decision": decision.model_dump(mode="json"),
        }
    )

    assert revision.writing_constraints == [
        "写作风格：analytical",
        "使用场景：student",
    ]


def test_document_section_result_binds_artifact_version_and_operation() -> None:
    result = DocumentSectionResult(
        task_type="document.write_section",
        status="drafted",
        run_id="run-doc-001",
        section_id="s1",
        draft_version="1",
        draft_artifact=chapter_artifact(),
        section_summary="章节摘要。",
        evidence_bundle_ref="data/runs/run-doc-001/evidence_bundles/s1.json",
        citation_map_ref="data/runs/run-doc-001/chapters/s1/v001.citations.json",
        used_source_ids=["source-1"],
        used_chunk_ids=["chunk-1"],
    )
    assert result.draft_artifact is not None

    with pytest.raises(ValidationError, match="evidence bundle"):
        DocumentSectionResult(
            task_type="document.write_section",
            status="drafted",
            run_id="run-doc-001",
            section_id="s1",
            draft_version="1",
            draft_artifact=chapter_artifact(),
            citation_map_ref="data/runs/run-doc-001/chapters/s1/v001.citations.json",
        )
    with pytest.raises(ValidationError, match="revised status"):
        DocumentSectionResult(
            task_type="document.write_section",
            status="revised",
            run_id="run-doc-001",
            section_id="s1",
            draft_version="1",
            draft_artifact=chapter_artifact(),
            evidence_bundle_ref="data/runs/run-doc-001/evidence_bundles/s1.json",
            citation_map_ref="data/runs/run-doc-001/chapters/s1/v001.citations.json",
        )
    with pytest.raises(ValidationError, match="initial draft version"):
        DocumentSectionResult(
            task_type="document.write_section",
            status="drafted",
            run_id="run-doc-001",
            section_id="s1",
            draft_version="2",
            draft_artifact=chapter_artifact(version="2"),
            evidence_bundle_ref="data/runs/run-doc-001/evidence_bundles/s1.json",
            citation_map_ref="data/runs/run-doc-001/chapters/s1/v002.citations.json",
        )
    with pytest.raises(ValidationError, match="invalid source id"):
        DocumentSectionResult(
            task_type="document.write_section",
            status="blocked",
            run_id="run-doc-001",
            section_id="s1",
            draft_version="1",
            used_source_ids=["x" * 161],
        )
