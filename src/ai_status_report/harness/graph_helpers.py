"""Shared pure helpers for controller nodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ai_status_report.a2a.client import response_text
from ai_status_report.documents.markdown import (
    MarkdownProtocolError,
    extract_reader_section_summary,
    validate_persisted_chapter_artifact,
)
from ai_status_report.harness.state import WorkflowState
from ai_status_report.rag.indexing import index_run_evidence
from ai_status_report.schemas.common import next_version
from ai_status_report.schemas.dialogue import PREFERENCE_OPTION_LABELS, PreferenceField
from ai_status_report.schemas.document import (
    DocumentBudget,
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.report import ArtifactRef, Outline, OutlineSection
from ai_status_report.schemas.report_spec import ReportSectionSpec
from ai_status_report.schemas.review import ReviewDecision
from ai_status_report.schemas.search import ResearchReport
from ai_status_report.settings import ConfigError, Settings, load_settings
from ai_status_report.storage.search_results import persist_research_report, run_directory
from ai_status_report.token_budget.config import model_budget, task_budget


def artifact_texts(response: object) -> list[str]:
    """Keep only A2A Artifact content as business results, never status text."""

    if not isinstance(response, str) and response.WhichOneof("payload") != "artifact_update":
        return []
    text = response_text(response)
    return [text] if text.strip() else []


def persist_outline(root: Path, run_id: str, section: OutlineSection) -> Path:
    """Persist the controller's current single-section pilot outline by content hash."""

    outline = Outline(outline_version="1", sections=[section])
    payload = json.dumps(outline.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    path = run_directory(root, run_id) / "outlines" / f"controller-{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(payload, encoding="utf-8")
    return path


def document_write_task(
    *, root: Path, run_id: str, user_input: str, report_payload: dict,
    section_spec: ReportSectionSpec, recent_section_summaries: list[dict],
    research_brief: dict | None = None,
) -> DocumentWriteSectionTask:
    """Build the controller-managed section task from a Search Artifact."""

    report = ResearchReport.model_validate(report_payload)
    try:
        generation_model = load_settings(root).generation.model
    except ConfigError:
        # Offline graph tests inject A2A results without model credentials.
        generation_model = "deepseek-v4-flash"
    model_profile = model_budget(root, generation_model)
    profile = task_budget(root, generation_model, "document_write")
    report_ref = persist_research_report(root, report, run_id=run_id)
    section = OutlineSection(
        section_id=section_spec.section_id,
        title=section_spec.title,
        questions=[*section_spec.core_questions, user_input],
        evidence_requirements=[section_spec.evidence_boundary],
        dependencies=section_spec.depends_on,
    )
    outline_ref = persist_outline(root, run_id, section)
    queries = list(dict.fromkeys(item for item in (report.query, user_input) if item.strip()))[:8]
    brief = research_brief or {}
    preference_lines: list[str] = []
    for label, custom_value, value, field in (
        (
            "写作风格",
            brief.get("custom_writing_style"),
            brief.get("writing_style"),
            PreferenceField.WRITING_STYLE,
        ),
        (
            "使用场景",
            brief.get("custom_usage_scenario"),
            brief.get("audience"),
            PreferenceField.USAGE_SCENARIO,
        ),
    ):
        resolved = custom_value or PREFERENCE_OPTION_LABELS[field].get(str(value), str(value))
        if resolved not in (None, "", "None"):
            preference_lines.append(f"{label}：{resolved}")
    return DocumentWriteSectionTask(
        run_id=run_id,
        research_report_ref=str(report_ref),
        research_summary=report.summary or f"围绕“{report.query}”完成{section.title}分析。",
        outline_ref=str(outline_ref),
        outline_version="1",
        section_id=section.section_id,
        section_title=section.title,
        section_goal=section_spec.purpose,
        retrieval_queries=queries,
        evidence_requirements=[*section.evidence_requirements, *section_spec.output_requirements],
        writing_constraints=[
            "使用客观、可追溯的中文行业分析表达。",
            "可用证据来自多个来源时，优先综合相关来源；不得为凑数量添加无关引用。",
            "不得编造数据、来源或引用。",
            *preference_lines,
        ],
        recent_section_summaries=recent_section_summaries,
        budget=DocumentBudget(
            context_window_tokens=model_profile.context_window_tokens,
            evidence_tokens=profile.evidence_tokens or 0,
            reserved_input_tokens=profile.reserved_input_tokens or 0,
            max_output_tokens=profile.max_output_tokens,
        ),
    )


def index_evidence_for_document(root: Path, run_id: str, settings: Settings) -> dict[str, object]:
    """Create the RAG bridge between durable Search Agent evidence and 8002."""

    return index_run_evidence(root, run_id, settings).as_dict()


def evidence_snapshot(root: Path, run_id: str) -> tuple[set[str], set[str]]:
    """Return durable source and EvidenceChunk identities already present in a run."""

    directory = run_directory(root, run_id) / "evidence"
    source_ids: set[str] = set()
    chunk_ids: set[str] = set()
    if not directory.is_dir():
        return source_ids, chunk_ids
    for path in directory.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            if isinstance(item.get("source_id"), str):
                source_ids.add(item["source_id"])
            if isinstance(item.get("chunk_id"), str):
                chunk_ids.add(item["chunk_id"])
    return source_ids, chunk_ids


def review_document_task(state: WorkflowState) -> DocumentWriteSectionTask | DocumentReviseSectionTask:
    """Recreate the controller's current chapter task for the review boundary."""

    payload = state.get("document_task")
    if not isinstance(payload, dict):
        raise TypeError("document_task_missing_for_review")
    model = {
        "document.write_section": DocumentWriteSectionTask,
        "document.revise_section": DocumentReviseSectionTask,
    }.get(payload.get("task_type"))
    if model is None:
        raise TypeError("document_task_invalid_for_review")
    try:
        return model.model_validate(payload)
    except ValueError as exc:
        raise TypeError("document_task_invalid_for_review") from exc


def document_revise_task(state: WorkflowState) -> DocumentReviseSectionTask:
    """Build a new immutable revision task from the reviewed chapter Artifact."""

    result = DocumentSectionResult.model_validate(state["document_result"])
    decision = ReviewDecision.model_validate(state["review_decision"])
    previous_task = state.get("document_task")
    if not isinstance(previous_task, dict) or result.draft_artifact is None:
        raise TypeError("document_revision_inputs_missing")
    try:
        retrieval_queries = list(dict.fromkeys([
            *decision.supplement_queries,
            *[str(query) for query in previous_task.get("retrieval_queries", [])],
        ]))[:8]
        return DocumentReviseSectionTask(
            run_id=result.run_id,
            research_report_ref=str(previous_task["research_report_ref"]),
            outline_ref=str(previous_task["outline_ref"]),
            outline_version=str(previous_task["outline_version"]),
            section_id=result.section_id,
            section_title=str(previous_task["section_title"]),
            base_draft=result.draft_artifact,
            target_draft_version=next_version(result.draft_version),
            review_issues=decision.issues,
            retrieval_queries=retrieval_queries,
            writing_constraints=[
                str(item) for item in previous_task.get("writing_constraints", [])
            ],
            writing_template_version=str(previous_task.get("writing_template_version", "1")),
            budget=DocumentBudget.model_validate(previous_task["budget"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("document_revision_inputs_missing") from exc


def validate_document_result(
    result: DocumentSectionResult,
    task: DocumentWriteSectionTask | DocumentReviseSectionTask,
) -> None:
    """Bind a returned specialist Artifact to the controller's dispatched task."""

    if (
        result.task_type != task.task_type
        or result.run_id != task.run_id
        or result.section_id != task.section_id
        or result.draft_version != task.target_draft_version
    ):
        raise RuntimeError("a2a_document_result_mismatch")


def verified_chapter_artifact_hash(root: Path, *, artifact: ArtifactRef, section_id: str) -> str:
    """Verify an A2A chapter Artifact against local immutable content."""

    try:
        _, content_hash = validate_persisted_chapter_artifact(
            root, artifact=artifact, section_id=section_id
        )
    except MarkdownProtocolError as exc:
        raise RuntimeError("a2a_document_artifact_integrity_failed") from exc
    return content_hash


def approved_section_context(
    root: Path, *, result: DocumentSectionResult
) -> tuple[ArtifactRef, dict[str, str]]:
    """Read predecessor context from an approved, integrity-checked chapter only."""

    if result.draft_artifact is None:
        raise RuntimeError("approved_chapter_artifact_missing")
    try:
        path, _ = validate_persisted_chapter_artifact(
            root, artifact=result.draft_artifact, section_id=result.section_id
        )
        summary = extract_reader_section_summary(path.read_text(encoding="utf-8"))
    except (MarkdownProtocolError, OSError, UnicodeError) as exc:
        raise RuntimeError("approved_chapter_artifact_integrity_failed") from exc
    return result.draft_artifact, {
        "section_id": result.section_id,
        "approved_version": result.draft_version,
        "summary": summary,
        "artifact_ref": result.draft_artifact.access_ref,
    }
