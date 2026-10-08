"""Run-scoped EvidenceChunk ingestion for the controller's RAG handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_status_report.rag.chroma import ChromaEvidenceIndex, RagIndexError
from ai_status_report.rag.chunking import to_retrieval_chunks
from ai_status_report.rag.glm import GLMClient
from ai_status_report.rag.index_versions import DEFAULT_INDEX_VERSION
from ai_status_report.rag.ingest import load_run_evidence
from ai_status_report.settings import Settings
from ai_status_report.storage.search_results import canonicalize_run_id
from ai_status_report.token_budget.runtime import new_run_ledger


@dataclass(frozen=True)
class IndexedRunEvidence:
    """Counts and version produced by one idempotent run-scoped Chroma upsert."""

    run_id: str
    evidence_chunks: int
    indexed_chunks: int
    embedded_chunks: int
    metadata_updated_chunks: int
    skipped_chunks: int
    index_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "evidence_chunks": self.evidence_chunks,
            "indexed_chunks": self.indexed_chunks,
            "embedded_chunks": self.embedded_chunks,
            "metadata_updated_chunks": self.metadata_updated_chunks,
            "skipped_chunks": self.skipped_chunks,
            "index_version": self.index_version,
        }


def index_run_evidence(
    root: Path,
    run_id: str,
    settings: Settings,
    *,
    index_version: str = DEFAULT_INDEX_VERSION,
) -> IndexedRunEvidence:
    """Embed a run's durable EvidenceChunks before a document task retrieves them.

    Each RetrievalChunk has a stable ID. Existing records with unchanged text
    are reused without another paid embedding request; changed text is
    re-embedded and metadata-only changes use Chroma's update operation. A run
    without retrievable evidence is a business blockage, not an empty RAG
    success: the Document Agent would otherwise receive an empty bundle.
    """

    canonical_run_id = canonicalize_run_id(run_id)
    evidence = load_run_evidence(root, canonical_run_id)
    chunks = [
        retrieval_chunk
        for item in evidence
        for retrieval_chunk in to_retrieval_chunks(
            item,
            run_id=canonical_run_id,
            index_version=index_version,
        )
    ]
    if not chunks:
        raise RagIndexError("no_indexable_evidence")
    with GLMClient(
        settings.embedding,
        settings.timeout,
        ledger=new_run_ledger(canonical_run_id, root=root),
    ) as client:
        index = ChromaEvidenceIndex(
            root,
            index_version=index_version,
            dimensions=settings.dimensions,
            client_mode=settings.chroma_client_mode,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        indexed_chunks = index.upsert(chunks, client)
    return IndexedRunEvidence(
        run_id=canonical_run_id,
        evidence_chunks=len(evidence),
        indexed_chunks=indexed_chunks,
        embedded_chunks=index.last_upsert_stats.embedded,
        metadata_updated_chunks=index.last_upsert_stats.metadata_updated,
        skipped_chunks=index.last_upsert_stats.skipped,
        index_version=index_version,
    )
