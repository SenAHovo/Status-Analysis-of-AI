"""Model-grounded chapter review for the controller's revision loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ai_status_report.context.builder import ContextBuilder, ContextError
from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.router import ResponseCache
from ai_status_report.model.structured import StructuredOutputError, parse_structured, schema_hint
from ai_status_report.schemas.document import (
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.evidence import EvidenceBundle
from ai_status_report.schemas.review import EvidenceReviewOutput, ReviewDecision
from ai_status_report.settings import ConfigError, load_settings
from ai_status_report.skill_runtime import ActiveSkills, MissingSkillError, SkillError, SkillLoader
from ai_status_report.storage.search_results import run_directory
from ai_status_report.token_budget.config import task_budget
from ai_status_report.token_budget.runtime import new_run_ledger

type DocumentTask = DocumentWriteSectionTask | DocumentReviseSectionTask

MAX_REVIEW_EXCERPTS = 50
MAX_REVIEW_EXCERPT_CHARS = 6000


@dataclass(frozen=True)
class ReviewContextAssembly:
    """Model-ready review messages and the loaded Skill version."""

    messages: list[dict]
    skill_hash: str
    dropped: tuple[str, ...]


class ReviewContextError(ValueError):
    """Safe reason code for unavailable or inconsistent review inputs."""


def _review_task_budget(root: Path, model: str):
    """Resolve the configured review budget; test doubles may use synthetic models."""

    try:
        return task_budget(root, model, "evidence_review")
    except ValueError:
        return task_budget(root, "deepseek-v4-flash", "evidence_review")


def _resolve_run_file(root: Path, run_id: str, reference: str, *, error: str) -> Path:
    run_root = run_directory(root, run_id).resolve()
    candidate = Path(reference)
    target = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if not target.is_relative_to(run_root) or not target.is_file():
        raise ReviewContextError(error)
    return target


def _chapter_markdown(root: Path, result: DocumentSectionResult) -> Path:
    if result.draft_artifact is None:
        raise ReviewContextError("review_chapter_artifact_unavailable")
    path = _resolve_run_file(
        root,
        result.run_id,
        result.draft_artifact.access_ref,
        error="review_chapter_artifact_unavailable",
    )
    expected = (
        run_directory(root, result.run_id) / "chapters" / result.section_id / f"v{result.draft_version}.md"
    ).resolve()
    if path != expected:
        raise ReviewContextError("review_chapter_artifact_unavailable")
    if result.draft_artifact.hash:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != result.draft_artifact.hash:
            raise ReviewContextError("review_chapter_artifact_unavailable")
    return path


def _load_json(root: Path, run_id: str, reference: str, *, error: str) -> dict:
    path = _resolve_run_file(root, run_id, reference, error=error)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ReviewContextError(error) from exc
    if not isinstance(payload, dict):
        raise ReviewContextError(error)
    return payload


def _review_evidence(bundle: EvidenceBundle) -> list[dict[str, object]]:
    """Expose bounded, source-linked excerpts rather than the whole source store."""

    return [
        {
            "source_id": item.source_id,
            "chunk_id": item.chunk_id,
            "title": item.title,
            "url": item.url,
            "verification_status": item.verification_status,
            "text": item.text[:MAX_REVIEW_EXCERPT_CHARS],
        }
        for item in bundle.excerpts[:MAX_REVIEW_EXCERPTS]
    ]


class ReviewContextAssembler:
    """Load ``evidence-review`` and assemble a bounded controller review request."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def assemble(
        self,
        *,
        task: DocumentTask,
        result: DocumentSectionResult,
        review_round: int,
    ) -> ReviewContextAssembly:
        chapter_path = _chapter_markdown(self.root, result)
        citation_map = _load_json(
            self.root,
            result.run_id,
            result.citation_map_ref,
            error="review_citation_map_unavailable",
        )
        bundle_payload = _load_json(
            self.root,
            result.run_id,
            result.evidence_bundle_ref,
            error="review_evidence_bundle_unavailable",
        )
        try:
            bundle = EvidenceBundle.model_validate(bundle_payload)
        except ValueError as exc:
            raise ReviewContextError("review_evidence_bundle_unavailable") from exc
        if bundle.run_id != result.run_id or bundle.section_id != result.section_id:
            raise ReviewContextError("review_evidence_bundle_unavailable")

        skill = SkillLoader(self.root / "skills").load("evidence-review")
        active = ActiveSkills()
        active.register(skill, phase="review")
        settings = load_settings(self.root)
        review_budget = _review_task_budget(self.root, settings.generation.model)
        builder = ContextBuilder(
            "你是主控 Agent 的章节审核器。只依据当前任务、章节、citation map 和证据包输出审核 JSON。"
            "不调用搜索工具、不改写章节、不把过程信息当作读者内容。",
            active,
            window_tokens=max(1024, task.budget.context_window_tokens - review_budget.max_output_tokens),
        )
        review_input = {
            "section": {
                "section_id": result.section_id,
                "title": task.section_title,
                "draft_version": result.draft_version,
                "review_round": review_round,
            },
            "task_constraints": {
                "outline_version": task.outline_version,
                "evidence_requirements": getattr(task, "evidence_requirements", []),
                "writing_constraints": getattr(task, "writing_constraints", []),
            },
            "chapter_markdown": chapter_path.read_text(encoding="utf-8"),
            "citation_map": citation_map,
            "document_evidence_gaps": [gap.model_dump(mode="json") for gap in result.evidence_gaps],
            "evidence": {
                "queries": bundle.queries,
                "quality_flags": bundle.quality_flags,
                "source_metadata": bundle.source_metadata,
                "excerpts": _review_evidence(bundle),
            },
        }
        assembly = builder.build(
            task=json.dumps(review_input, ensure_ascii=False),
            phase="review",
            required_skills=["evidence-review"],
        )
        return ReviewContextAssembly(
            messages=assembly.messages,
            skill_hash=skill.content_hash,
            dropped=assembly.dropped,
        )


def _cache_key(model: str, messages: list[dict], options: dict) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "options": options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ModelEvidenceReviewer:
    """Use the configured generation model with a deterministic safe fallback."""

    def __init__(self, root: Path, cache: ResponseCache | None = None) -> None:
        self.root = root
        self.cache = cache or ResponseCache(root / "data" / "cache" / "evidence_review.sqlite3")

    def review(
        self,
        *,
        task: DocumentTask,
        result: DocumentSectionResult,
        review_round: int,
    ) -> ReviewDecision:
        fallback = ReviewDecision.from_document_result(
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            evidence_gaps=[item.model_dump(mode="json") for item in result.evidence_gaps],
            review_round=review_round,
        )
        try:
            assembly = ReviewContextAssembler(self.root).assemble(
                task=task,
                result=result,
                review_round=review_round,
            )
            settings = load_settings(self.root)
            review_budget = _review_task_budget(self.root, settings.generation.model)
            options = {
                "response_format": {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "temperature": 0.1,
                "max_tokens": review_budget.max_output_tokens,
            }
            messages = [
                {
                    **assembly.messages[0],
                    "content": assembly.messages[0]["content"]
                    + "\n\n"
                    + schema_hint(EvidenceReviewOutput),
                },
                *assembly.messages[1:],
            ]
            key = _cache_key(settings.generation.model, messages, options)
            cached = self.cache.get(key)
            if cached is not None:
                output = EvidenceReviewOutput.model_validate(cached)
                return output.bind(
                    run_id=result.run_id,
                    section_id=result.section_id,
                    draft_version=result.draft_version,
                    review_round=review_round,
                    source="model_cache",
                )
            try:
                client_context = DeepSeekClient(
                    settings.generation,
                    settings.timeout,
                    ledger=new_run_ledger(result.run_id, root=self.root),
                )
            except TypeError as exc:
                if "ledger" not in str(exc):
                    raise
                client_context = DeepSeekClient(settings.generation, settings.timeout)
            with client_context as client:
                response = client.chat(messages, **options)
            content = response["choices"][0]["message"]["content"]
            output = parse_structured(content, EvidenceReviewOutput)
            decision = output.bind(
                run_id=result.run_id,
                section_id=result.section_id,
                draft_version=result.draft_version,
                review_round=review_round,
                source="model",
            )
            self.cache.put(key, output.model_dump(mode="json"))
            return decision
        except (
            ConfigError,
            AttributeError,
            ContextError,
            KeyError,
            MissingSkillError,
            OSError,
            ProviderError,
            ReviewContextError,
            SkillError,
            StructuredOutputError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            code = _safe_fallback_code(exc)
            return fallback.model_copy(
                update={
                    "source": "deterministic_fallback",
                    "fallback_reason": code,
                }
            )


def _safe_fallback_code(exc: Exception) -> str:
    """Expose a stable diagnostic category without persisting provider internals."""

    if isinstance(exc, ReviewContextError | ContextError | MissingSkillError | SkillError):
        return "review_context_unavailable"
    if isinstance(exc, ProviderError | ConfigError):
        return "review_provider_failed"
    if isinstance(exc, StructuredOutputError | ValueError):
        if str(exc) == "review_issue_identity_mismatch":
            return "review_issue_identity_mismatch"
        return "review_structured_output_invalid"
    return "review_execution_failed"
