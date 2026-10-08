"""Document Agent execution for bounded writing and revision A2A tasks.

Each valid chapter task retrieves and persists an ``EvidenceBundle``,
loads its task-specific Skill, calls the configured model for Markdown, and
delivers an immutable chapter Artifact through the A2A task stream.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from pydantic import ValidationError

from ai_status_report.context.document import (
    DocumentContextError,
    RevisionContextAssembler,
    SectionContextAssembler,
    SectionContextAssembly,
)
from ai_status_report.documents.markdown import (
    MarkdownProtocolError,
    ParsedSectionMarkdown,
    normalize_reader_citations,
    parse_section_markdown,
    persist_citation_map,
    persist_section_markdown,
    render_reader_markdown,
    validate_markdown_evidence,
)
from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.rag.bundle import build_evidence_bundle, persist_evidence_bundle
from ai_status_report.rag.chroma import ChromaEvidenceIndex, RagIndexError
from ai_status_report.rag.glm import GLMClient
from ai_status_report.schemas.document import (
    DocumentReviseSectionTask,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.evidence import EvidenceBundle
from ai_status_report.settings import ConfigError, load_settings
from ai_status_report.skill_runtime import MissingSkillError, SkillError
from ai_status_report.token_budget.config import model_budget, task_budget
from ai_status_report.token_budget.runtime import new_run_ledger

type DocumentTask = DocumentWriteSectionTask | DocumentReviseSectionTask
_SAFE_TASK_ERROR_CODES = frozenset(
    {
        "invalid_document_task",
        "unsupported_document_task",
        "input_encoding_corrupted",
        "document_task_budget_invalid",
    }
)
DOCUMENT_RETRIEVAL_CANDIDATES = 32


class DocumentTaskError(ValueError):
    """A safe task-validation failure that can be sent through A2A."""


def _has_likely_encoding_corruption(value: str) -> bool:
    """Detect a query whose original characters were lossily replaced by ``?``.

    Windows terminal misconfiguration can replace each non-ASCII input
    character before Python receives stdin. The original text cannot be
    recovered here, so reject an all-question-mark query before it can spend
    an embedding request or pollute a persisted evidence bundle. A single
    ordinary question-mark query remains valid.
    """

    compact = "".join(value.split())
    return len(compact) >= 2 and set(compact) == {"?"}


def _validate_task_encoding(task: DocumentTask) -> None:
    """Reject lossy retrieval queries before the model or index boundary."""

    if any(_has_likely_encoding_corruption(query) for query in task.retrieval_queries):
        raise DocumentTaskError("input_encoding_corrupted")


@dataclass(frozen=True)
class RetrievedEvidenceBundle:
    """A persisted evidence bundle ready for a writing stage."""

    bundle: EvidenceBundle
    path: Path


class DocumentEvidenceRetriever:
    """Retrieve and persist evidence for one validated chapter task."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self, task: DocumentTask) -> RetrievedEvidenceBundle:
        settings = load_settings(self.root)
        with GLMClient(
            settings.embedding,
            settings.timeout,
            ledger=new_run_ledger(task.run_id, root=self.root),
        ) as client:
            index_kwargs = {"dimensions": settings.dimensions}
            if getattr(settings, "chroma_client_mode", "persistent") == "http":
                index_kwargs.update(
                    client_mode="http",
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
            index = ChromaEvidenceIndex(self.root, **index_kwargs)
            queries = task.retrieval_queries or [task.section_title]
            matches = [
                match
                for query in queries
                for match in index.query(
                    query,
                    client,
                    n_results=DOCUMENT_RETRIEVAL_CANDIDATES,
                    run_id=task.run_id,
                )
            ]
        bundle = build_evidence_bundle(
            run_id=task.run_id,
            section_id=task.section_id,
            queries=queries,
            matches=matches,
            budget_tokens=task.budget.evidence_tokens,
            index_version=index.index_version,
            root=self.root,
        )
        return RetrievedEvidenceBundle(
            bundle=bundle, path=persist_evidence_bundle(self.root, bundle)
        )


@dataclass(frozen=True)
class GeneratedMarkdown:
    markdown: str
    parsed: ParsedSectionMarkdown
    skill_hash: str
    dropped_context: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSectionWrite:
    """A fully assembled, model-ready section-writing request."""

    assembly: SectionContextAssembly


@dataclass(frozen=True)
class PreparedSectionRevision:
    """A fully assembled, model-ready section-revision request."""

    assembly: SectionContextAssembly


def _generate_markdown(
    root: Path,
    *,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    run_id: str,
) -> str:
    """Call the document model and return its Markdown-only content."""

    settings = load_settings(root)
    options = {
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "temperature": temperature,
    }
    try:
        client_context = DeepSeekClient(
            settings.generation,
            settings.timeout,
            ledger=new_run_ledger(run_id, root=root),
        )
    except TypeError as exc:
        # Keep legacy injected test clients compatible with the new optional
        # accounting boundary; the production client accepts ``ledger``.
        if "ledger" not in str(exc):
            raise
        client_context = DeepSeekClient(settings.generation, settings.timeout)
    with client_context as client:
        response = client.chat(messages, **options)
    try:
        content = response["choices"][0]["message"]["content"]
    except (AttributeError, KeyError, IndexError, TypeError) as exc:
        raise ProviderError("document_markdown_invalid") from exc
    if not isinstance(content, str):
        raise ProviderError("document_markdown_invalid")
    return content.strip()


class DocumentSectionWriter:
    """Generate and parse a first-draft Markdown document without A2A assembly."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(
        self, task: DocumentWriteSectionTask, retrieved: RetrievedEvidenceBundle
    ) -> PreparedSectionWrite:
        """Load the writing Skill and assemble the bounded model context."""

        return PreparedSectionWrite(
            assembly=SectionContextAssembler(self.root).assemble(task, retrieved.bundle)
        )

    def generate(
        self, task: DocumentWriteSectionTask, prepared: PreparedSectionWrite
    ) -> GeneratedMarkdown:
        """Call the generation model and parse its Markdown-only response."""

        try:
            markdown = _generate_markdown(
                self.root,
                messages=prepared.assembly.messages,
                max_tokens=task.budget.max_output_tokens,
                temperature=0.7,
                run_id=task.run_id,
            )
            parsed = parse_section_markdown(markdown)
        except MarkdownProtocolError as exc:
            raise ProviderError("document_markdown_invalid") from exc
        return GeneratedMarkdown(
            markdown=markdown,
            parsed=parsed,
            skill_hash=prepared.assembly.skill_hash,
            dropped_context=prepared.assembly.dropped,
        )

    def __call__(
        self, task: DocumentWriteSectionTask, retrieved: RetrievedEvidenceBundle
    ) -> GeneratedMarkdown:
        return self.generate(task, self.prepare(task, retrieved))


class DocumentSectionReviser:
    """Generate a revised Markdown chapter from a prior Artifact and current evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def prepare(
        self, task: DocumentReviseSectionTask, retrieved: RetrievedEvidenceBundle
    ) -> PreparedSectionRevision:
        return PreparedSectionRevision(
            assembly=RevisionContextAssembler(self.root).assemble(task, retrieved.bundle)
        )

    def generate(
        self, task: DocumentReviseSectionTask, prepared: PreparedSectionRevision
    ) -> GeneratedMarkdown:
        try:
            markdown = _generate_markdown(
                self.root,
                messages=prepared.assembly.messages,
                max_tokens=task.budget.max_output_tokens,
                temperature=0.3,
                run_id=task.run_id,
            )
            parsed = parse_section_markdown(markdown)
        except MarkdownProtocolError as exc:
            raise ProviderError("document_markdown_invalid") from exc
        return GeneratedMarkdown(
            markdown=markdown,
            parsed=parsed,
            skill_hash=prepared.assembly.skill_hash,
            dropped_context=prepared.assembly.dropped,
        )

    def __call__(
        self, task: DocumentReviseSectionTask, retrieved: RetrievedEvidenceBundle
    ) -> GeneratedMarkdown:
        return self.generate(task, self.prepare(task, retrieved))


def parse_document_task(text: str) -> DocumentTask:
    """Parse an A2A text payload into one allowed business task.

    The external protocol receives only a stable, safe error code. Detailed
    Pydantic errors remain local to avoid reflecting untrusted payloads in
    A2A status messages.
    """

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DocumentTaskError("invalid_document_task") from exc
    if not isinstance(payload, dict):
        raise DocumentTaskError("invalid_document_task")
    task_type = payload.get("task_type")
    if not isinstance(task_type, str):
        raise DocumentTaskError("invalid_document_task")
    model = {
        "document.write_section": DocumentWriteSectionTask,
        "document.revise_section": DocumentReviseSectionTask,
    }.get(task_type)
    if model is None:
        raise DocumentTaskError("unsupported_document_task")
    try:
        task = model.model_validate(payload)
    except ValidationError as exc:
        raise DocumentTaskError("invalid_document_task") from exc
    _validate_task_encoding(task)
    return task


def validate_document_task_budget(root: Path, task: DocumentTask) -> None:
    """Ensure an A2A task stays within the configured document-task ceilings.

    The controller normally emits the configured profile unchanged.  Lower
    per-task limits remain valid because they are a safe way to constrain an
    individual chapter or an external A2A caller; only a limit exceeding the
    configured teaching budget is rejected.
    """

    settings = load_settings(root)
    model_profile = model_budget(root, settings.generation.model)
    profile = task_budget(
        root,
        settings.generation.model,
        "document_write" if isinstance(task, DocumentWriteSectionTask) else "document_revision",
    )
    if (
        task.budget.context_window_tokens > model_profile.context_window_tokens
        or task.budget.evidence_tokens > (profile.evidence_tokens or 0)
        or task.budget.reserved_input_tokens > (profile.reserved_input_tokens or 0)
        or task.budget.max_output_tokens > profile.max_output_tokens
    ):
        raise DocumentTaskError("document_task_budget_invalid")


class DocumentTaskExecutor(AgentExecutor):
    """Run the protocol portion of the Document Agent task contract."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        evidence_retriever: Callable[[DocumentTask], RetrievedEvidenceBundle] | None = None,
        section_writer: Callable[
            [DocumentWriteSectionTask, RetrievedEvidenceBundle], GeneratedMarkdown
        ]
        | None = None,
        section_reviser: Callable[
            [DocumentReviseSectionTask, RetrievedEvidenceBundle], GeneratedMarkdown
        ]
        | None = None,
    ) -> None:
        self.root = root or Path.cwd()
        self.evidence_retriever = evidence_retriever or DocumentEvidenceRetriever(self.root)
        # Production uses the real write/revision lifecycle. Tests can inject
        # deterministic dependencies and therefore remain offline.
        self.section_writer = (
            section_writer
            if section_writer is not None
            else (None if evidence_retriever is not None else DocumentSectionWriter(self.root))
        )
        self.section_reviser = (
            section_reviser
            if section_reviser is not None
            else (None if evidence_retriever is not None else DocumentSectionReviser(self.root))
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work(
            updater.new_agent_message([new_text_part("document.task.received")])
        )
        try:
            document_task = parse_document_task(context.get_user_input())
            try:
                validate_document_task_budget(self.root, document_task)
            except ConfigError:
                # Offline protocol tests intentionally run without credentials.
                pass
        except (DocumentTaskError, TypeError, ValueError) as exc:
            error_code = str(exc)
            if error_code not in _SAFE_TASK_ERROR_CODES:
                error_code = "invalid_document_task"
            await updater.failed(updater.new_agent_message([new_text_part(error_code)]))
            return

        await updater.start_work(
            updater.new_agent_message([new_text_part("document.task.validated")])
        )
        await updater.start_work(
            updater.new_agent_message([new_text_part("document.evidence.retrieving")])
        )
        try:
            retrieved = await asyncio.to_thread(self.evidence_retriever, document_task)
        except (ConfigError, OSError, ProviderError, RagIndexError, ValueError):
            result = self._blocked_result(
                document_task,
                gaps=["evidence_retrieval_failed", "document_execution_not_enabled"],
                quality_flags=["task_validated"],
            )
            await self._add_result_artifact(updater, result)
            await updater.complete(
                updater.new_agent_message([new_text_part("document.task.completed")])
            )
            return

        await updater.start_work(
            updater.new_agent_message([new_text_part("document.evidence.persisted")])
        )
        generator = (
            self.section_writer
            if isinstance(document_task, DocumentWriteSectionTask)
            else self.section_reviser
        )
        if generator is not None:
            await updater.start_work(
                updater.new_agent_message([new_text_part("document.skill.loaded")])
            )
            generated: GeneratedMarkdown
            if isinstance(generator, (DocumentSectionWriter, DocumentSectionReviser)):
                try:
                    prepared = await asyncio.to_thread(
                        generator.prepare, document_task, retrieved
                    )
                except (
                    ConfigError,
                    DocumentContextError,
                    MissingSkillError,
                    OSError,
                    SkillError,
                    UnicodeError,
                    ValueError,
                ):
                    await self._complete_blocked_generation(
                        updater,
                        document_task,
                        retrieved,
                        gap="document_context_assembly_failed",
                    )
                    return
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.context.assembled")])
                )
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.model.requesting")])
                )
                try:
                    generated = await asyncio.to_thread(
                        generator.generate, document_task, prepared
                    )
                except ProviderError as exc:
                    await self._complete_blocked_generation(
                        updater,
                        document_task,
                        retrieved,
                        gap=(
                            "document_markdown_protocol_failed"
                            if str(exc) == "document_markdown_invalid"
                            else "document_model_generation_failed"
                        ),
                    )
                    return
                except (ConfigError, OSError, ValueError):
                    await self._complete_blocked_generation(
                        updater,
                        document_task,
                        retrieved,
                        gap="document_model_generation_failed",
                    )
                    return
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.model.responded")])
                )
            else:
                # Injected writers are deterministic test or harness adapters.  They
                # own their preparation boundary, while production writes use the
                # explicit prepare/generate lifecycle above.
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.context.assembled")])
                )
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.model.requesting")])
                )
                try:
                    generated = await asyncio.to_thread(
                        generator, document_task, retrieved
                    )
                except (ConfigError, OSError, ProviderError, DocumentContextError, ValueError):
                    await self._complete_blocked_generation(
                        updater,
                        document_task,
                        retrieved,
                        gap="document_model_generation_failed",
                    )
                    return
                await updater.start_work(
                    updater.new_agent_message([new_text_part("document.model.responded")])
                )
            await updater.start_work(
                updater.new_agent_message([new_text_part("document.markdown.parsed")])
            )
            try:
                citations = validate_markdown_evidence(generated.parsed, retrieved.bundle)
                reader_parsed, citations = normalize_reader_citations(
                    generated.parsed, citations
                )
                reader_markdown = render_reader_markdown(reader_parsed, citations)
            except MarkdownProtocolError:
                await self._complete_blocked_generation(
                    updater,
                    document_task,
                    retrieved,
                    gap="document_evidence_validation_failed",
                )
                return
            await updater.start_work(
                updater.new_agent_message([new_text_part("document.evidence.validated")])
            )
            try:
                input_versions = {
                    "skill": generated.skill_hash,
                    "outline": document_task.outline_version,
                }
                if isinstance(document_task, DocumentReviseSectionTask):
                    input_versions["base_draft"] = document_task.base_draft.version
                artifact = persist_section_markdown(
                    self.root,
                    run_id=document_task.run_id,
                    section_id=document_task.section_id,
                    version=document_task.target_draft_version,
                    markdown=reader_markdown,
                    input_versions=input_versions,
                )
                citation_map = persist_citation_map(
                    self.root,
                    artifact=artifact,
                    citations=citations,
                    evidence_gaps=generated.parsed.evidence_gaps,
                )
            except MarkdownProtocolError as exc:
                await self._complete_blocked_generation(
                    updater,
                    document_task,
                    retrieved,
                    gap=(
                        "document_draft_version_conflict"
                        if str(exc) == "chapter_version_conflict"
                        else "document_draft_persistence_failed"
                    ),
                )
                return
            except (OSError, ValueError):
                await self._complete_blocked_generation(
                    updater,
                    document_task,
                    retrieved,
                    gap="document_draft_persistence_failed",
                )
                return
            persisted_event = (
                "document.draft.persisted"
                if isinstance(document_task, DocumentWriteSectionTask)
                else "document.revision.persisted"
            )
            await updater.start_work(updater.new_agent_message([new_text_part(persisted_event)]))
            result = DocumentSectionResult(
                task_type=document_task.task_type,
                status=(
                    "drafted"
                    if isinstance(document_task, DocumentWriteSectionTask)
                    else "revised"
                ),
                run_id=document_task.run_id,
                section_id=document_task.section_id,
                draft_version=document_task.target_draft_version,
                draft_artifact=artifact,
                section_summary=generated.parsed.summary,
                evidence_bundle_ref=str(retrieved.path),
                citation_map_ref=str(citation_map),
                used_source_ids=list(
                    dict.fromkeys(source_id for item in citations for source_id in item.source_ids)
                ),
                used_chunk_ids=list(
                    dict.fromkeys(
                        chunk_id for item in citations for chunk_id in item.supporting_chunk_ids
                    )
                ),
                evidence_gaps=list(generated.parsed.evidence_gaps),
                quality_flags=[
                    "task_validated",
                    "evidence_bundle_persisted",
                    "skill_loaded",
                    "markdown_parsed",
                    (
                        "draft_persisted"
                        if isinstance(document_task, DocumentWriteSectionTask)
                        else "revision_persisted"
                    ),
                    *generated.dropped_context,
                ],
            )
            await self._add_result_artifact(updater, result)
            await updater.complete(
                updater.new_agent_message([new_text_part("document.task.completed")])
            )
            return
        source_ids = list(dict.fromkeys(excerpt.source_id for excerpt in retrieved.bundle.excerpts))
        result = self._blocked_result(
            document_task,
            evidence_bundle_ref=str(retrieved.path),
            used_source_ids=source_ids,
            used_chunk_ids=list(dict.fromkeys(retrieved.bundle.chunk_refs)),
            gaps=["document_execution_not_enabled"],
            quality_flags=[
                "task_validated",
                "evidence_bundle_persisted",
                *retrieved.bundle.quality_flags,
            ],
        )
        await self._add_result_artifact(updater, result)
        await updater.complete(
            updater.new_agent_message([new_text_part("document.task.completed")])
        )

    @staticmethod
    def _blocked_result(
        task: DocumentTask,
        *,
        evidence_bundle_ref: str = "",
        used_source_ids: list[str] | None = None,
        used_chunk_ids: list[str] | None = None,
        gaps: list[str],
        quality_flags: list[str],
    ) -> DocumentSectionResult:
        return DocumentSectionResult(
            task_type=task.task_type,
            status="blocked",
            run_id=task.run_id,
            section_id=task.section_id,
            draft_version=task.target_draft_version,
            evidence_bundle_ref=evidence_bundle_ref,
            used_source_ids=used_source_ids or [],
            used_chunk_ids=used_chunk_ids or [],
            gaps=gaps,
            quality_flags=quality_flags,
        )

    async def _complete_blocked_generation(
        self,
        updater: TaskUpdater,
        task: DocumentTask,
        retrieved: RetrievedEvidenceBundle,
        *,
        gap: str,
    ) -> None:
        """Return a precise, safe business outcome for a visible generation phase."""

        result = self._blocked_result(
            task,
            evidence_bundle_ref=str(retrieved.path),
            gaps=[gap],
            quality_flags=["task_validated", "evidence_bundle_persisted", "skill_loaded"],
        )
        await updater.start_work(
            updater.new_agent_message([new_text_part("document.task.blocked")])
        )
        await self._add_result_artifact(updater, result)
        await updater.complete(
            updater.new_agent_message([new_text_part("document.task.completed")])
        )

    @staticmethod
    async def _add_result_artifact(updater: TaskUpdater, result: DocumentSectionResult) -> None:
        await updater.add_artifact(
            [new_text_part(result.model_dump_json())],
            name="document_task_result",
            metadata={"task_type": result.task_type, "result_status": result.status},
            last_chunk=True,
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
