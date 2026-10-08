"""Budgeted context assembly for document draft and revision operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai_status_report.documents.markdown import (
    CitationPlan,
    MarkdownProtocolError,
    build_citation_plan,
    validate_persisted_chapter_artifact,
)
from ai_status_report.schemas.document import DocumentReviseSectionTask, DocumentWriteSectionTask
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt
from ai_status_report.skill_runtime import ActiveSkills, SkillLoader
from ai_status_report.storage.search_results import run_directory
from ai_status_report.token_budget.estimator import estimate_messages


class DocumentContextError(ValueError):
    """Safe context-assembly reason code."""


@dataclass(frozen=True)
class SectionContextAssembly:
    messages: list[dict]
    skill_hash: str
    input_tokens: int
    evidence_tokens: int
    included_chunk_ids: tuple[str, ...]
    dropped: tuple[str, ...]
    citation_plan: CitationPlan


def _excerpt_text(excerpt: Excerpt, citation_plan: CitationPlan) -> str:
    citation = citation_plan.for_excerpt(excerpt)
    return (
        f"[reader_citation=[{citation.number}]; source_id={excerpt.source_id}; chunk_id={excerpt.chunk_id}; "
        f"retrieval_chunk_id={excerpt.retrieval_chunk_id}; url={excerpt.url}; "
        f"verification_status={excerpt.verification_status}]\n{excerpt.text}"
    )


def _run_path(root: Path, run_id: str, reference: str, *, error: str) -> Path:
    """Resolve a local artifact reference only inside its declared run directory."""

    run_root = run_directory(root, run_id).resolve()
    candidate = Path(reference)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not target.is_relative_to(run_root) or not target.is_file():
        raise DocumentContextError(error)
    return target


def _base_draft_path(root: Path, task: DocumentReviseSectionTask) -> Path:
    """Resolve and verify the exact immutable chapter Artifact being revised."""

    try:
        path, _ = validate_persisted_chapter_artifact(
            root,
            artifact=task.base_draft,
            section_id=task.section_id,
        )
    except MarkdownProtocolError as exc:
        raise DocumentContextError("revision_base_draft_unavailable") from exc
    return path


def _outline_section(root: Path, task: DocumentReviseSectionTask) -> dict:
    """Load the current section definition from the controller's persisted outline."""

    path = _run_path(root, task.run_id, task.outline_ref, error="revision_outline_unavailable")
    try:
        outline = json.loads(path.read_text(encoding="utf-8"))
        sections = outline["sections"]
        section = next(
            item for item in sections if isinstance(item, dict) and item.get("section_id") == task.section_id
        )
    except (json.JSONDecodeError, KeyError, OSError, StopIteration, TypeError) as exc:
        raise DocumentContextError("revision_outline_unavailable") from exc
    return section


def _assemble(
    *,
    skill_name: str,
    phase: str,
    system_role: str,
    task_payload: dict,
    bundle: EvidenceBundle,
    budget,
    root: Path,
) -> SectionContextAssembly:
    """Load one Skill and admit complete evidence excerpts within the task budget."""

    skill = SkillLoader(root / "skills").load(skill_name)
    active = ActiveSkills()
    active.register(skill, phase=phase)
    active.ensure_required([(skill_name, phase)])
    guidance = active.guidance([skill_name])[0]
    citation_plan = build_citation_plan(bundle)
    system = {
        "role": "system",
        "content": f"{system_role}\n\n适用业务指导：\n{guidance.body}",
    }
    user_prefix = "章节任务：\n" + json.dumps(task_payload, ensure_ascii=False)
    fixed_messages = [system, {"role": "user", "content": user_prefix}]
    fixed_tokens = estimate_messages(fixed_messages)
    input_limit = budget.context_window_tokens - budget.max_output_tokens
    if fixed_tokens > budget.reserved_input_tokens or fixed_tokens > input_limit:
        raise DocumentContextError("reserved_input_budget_exceeded")

    evidence_parts: list[str] = []
    included: list[str] = []
    dropped: list[str] = []
    for excerpt in bundle.excerpts:
        candidate = evidence_parts + [_excerpt_text(excerpt, citation_plan)]
        messages = [
            system,
            {"role": "user", "content": user_prefix + "\n\nEvidenceBundle：\n" + "\n\n".join(candidate)},
        ]
        evidence_used = estimate_messages(messages) - fixed_tokens
        if evidence_used <= budget.evidence_tokens and estimate_messages(messages) <= input_limit:
            evidence_parts = candidate
            included.append(excerpt.chunk_id)
        else:
            dropped.append(f"evidence_budget:{excerpt.chunk_id}")
    content = user_prefix
    if evidence_parts:
        content += "\n\nEvidenceBundle：\n" + "\n\n".join(evidence_parts)
    messages = [system, {"role": "user", "content": content}]
    return SectionContextAssembly(
        messages=messages,
        skill_hash=guidance.version_hash,
        input_tokens=estimate_messages(messages),
        evidence_tokens=estimate_messages(messages) - fixed_tokens,
        included_chunk_ids=tuple(included),
        dropped=tuple(dropped),
        citation_plan=citation_plan,
    )


class SectionContextAssembler:
    """Load ``report-writing`` and assemble a first-draft model context."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def assemble(
        self, task: DocumentWriteSectionTask, bundle: EvidenceBundle
    ) -> SectionContextAssembly:
        return _assemble(
            skill_name="report-writing",
            phase="write",
            system_role="你是文档生成 Agent，负责撰写可追溯的中文人工智能现状分析章节。",
            task_payload={
                "task_type": task.task_type,
                "section_id": task.section_id,
                "section_title": task.section_title,
                "section_goal": task.section_goal,
                "research_summary": task.research_summary,
                "outline_ref": task.outline_ref,
                "outline_version": task.outline_version,
                "evidence_requirements": task.evidence_requirements,
                "writing_constraints": task.writing_constraints,
                "citation_style": task.citation_style,
                "recent_section_summaries": [item.model_dump() for item in task.recent_section_summaries],
            },
            bundle=bundle,
            budget=task.budget,
            root=self.root,
        )


class RevisionContextAssembler:
    """Load ``report-revision`` with its old chapter and current evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def assemble(
        self, task: DocumentReviseSectionTask, bundle: EvidenceBundle
    ) -> SectionContextAssembly:
        base_path = _base_draft_path(self.root, task)
        try:
            base_markdown = base_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise DocumentContextError("revision_base_draft_unavailable") from exc
        return _assemble(
            skill_name="report-revision",
            phase="revise",
            system_role="你是文档生成 Agent，负责修订可追溯的中文人工智能现状分析章节。",
            task_payload={
                "task_type": task.task_type,
                "section_id": task.section_id,
                "section_title": task.section_title,
                "outline_version": task.outline_version,
                "outline_section": _outline_section(self.root, task),
                "base_draft_version": task.base_draft.version,
                "target_draft_version": task.target_draft_version,
                "review_issues": [issue.model_dump() for issue in task.review_issues],
                "retrieval_queries": task.retrieval_queries,
                "writing_constraints": task.writing_constraints,
                "base_markdown": base_markdown,
            },
            bundle=bundle,
            budget=task.budget,
            root=self.root,
        )
