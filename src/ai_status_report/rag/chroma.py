"""Chroma persistence and retrieval using externally generated GLM vectors.

Chroma API references:
https://docs.trychroma.com/reference/python/client
https://docs.trychroma.com/reference/python/collection
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import chromadb
import httpx

from ai_status_report.rag.glm import GLMClient
from ai_status_report.rag.index_versions import (
    DEFAULT_INDEX_VERSION,
    distance_space_for,
    is_legacy_read_only,
    is_valid_index_version,
)
from ai_status_report.rag.schemas import RetrievalChunk, RetrievalMatch
from ai_status_report.storage.search_results import canonicalize_run_id


class RagIndexError(RuntimeError):
    """Fixed error codes for index and retrieval boundary failures."""


@dataclass(frozen=True)
class UpsertStats:
    """Observe which records required paid embedding work during an upsert."""

    total: int
    embedded: int
    metadata_updated: int
    skipped: int


class ChromaEvidenceIndex:
    """Persistent Chroma collection for one embedding/index version."""

    embedding_model = "embedding-3"
    def __init__(
        self,
        root: Path,
        *,
        index_version: str = DEFAULT_INDEX_VERSION,
        dimensions: int = 2048,
        client_mode: str = "persistent",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        if not is_valid_index_version(index_version):
            raise RagIndexError("invalid_index_version")
        if dimensions not in {256, 512, 1024, 2048}:
            raise RagIndexError("invalid_embedding_dimensions")
        self.index_version = index_version
        self.dimensions = dimensions
        self.distance_space = distance_space_for(index_version)
        self.last_upsert_stats = UpsertStats(0, 0, 0, 0)
        self.path = root / "data" / "vector_store" / "chroma"
        if client_mode not in {"persistent", "http"}:
            raise RagIndexError("invalid_chroma_client_mode")
        try:
            if client_mode == "persistent":
                self.path.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(self.path))
            else:
                # Chroma documents PersistentClient for local development and
                # testing, and HttpClient for a shared client-server setup.
                # Source: https://docs.trychroma.com/reference/python/client
                self.client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.client.get_or_create_collection(
                name=f"evidence_chunks_{index_version}",
                metadata={
                    "index_version": index_version,
                    "embedding_model": self.embedding_model,
                    "embedding_dimensions": dimensions,
                    "distance_space": self.distance_space,
                },
                # Chroma defaults to squared L2. Persist an explicit metric because
                # query distances and their thresholds are meaningful only within
                # the collection's configured embedding space.
                # Source: https://docs.trychroma.com/docs/collections/configure
                configuration={"hnsw": {"space": self.distance_space}},
            )
        except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise RagIndexError("chroma_client_unavailable") from exc
        expected_metadata = {
            "index_version": index_version,
            "embedding_model": self.embedding_model,
            "embedding_dimensions": dimensions,
        }
        metadata = self.collection.metadata or {}
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RagIndexError("index_metadata_mismatch")
        stored_space = metadata.get("distance_space")
        if not is_legacy_read_only(index_version) and stored_space != self.distance_space:
            raise RagIndexError("index_metadata_mismatch")
        if stored_space is not None and stored_space != self.distance_space:
            raise RagIndexError("index_metadata_mismatch")
        configured_space = (self.collection.configuration or {}).get("hnsw", {}).get("space")
        if configured_space != self.distance_space:
            raise RagIndexError("index_configuration_mismatch")

    def healthcheck(self) -> int:
        """Return collection size after forcing a safe segment read."""

        try:
            return int(self.collection.count())
        except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise RagIndexError("chroma_index_unavailable") from exc

    def _check_health(self) -> None:
        """Force a segment read before a write/query crosses an A2A boundary."""

        self.healthcheck()

    def upsert(
        self,
        chunks: list[RetrievalChunk],
        client: GLMClient,
        *,
        batch_size: int = 16,
    ) -> int:
        if not chunks:
            return 0
        self._check_health()
        if is_legacy_read_only(self.index_version):
            raise RagIndexError("legacy_index_read_only")
        if any(chunk.index_version != self.index_version for chunk in chunks):
            raise RagIndexError("retrieval_chunk_index_version_mismatch")
        if not 1 <= batch_size <= 16:
            raise RagIndexError("invalid_embedding_batch_size")
        existing: dict[str, tuple[str, dict[str, str | int]]] = {}
        try:
            for start in range(0, len(chunks), 1000):
                records = self.collection.get(
                    ids=[chunk.retrieval_chunk_id for chunk in chunks[start : start + 1000]],
                    include=["documents", "metadatas"],
                )
                ids = records.get("ids", [])
                documents = records.get("documents", []) or []
                metadatas = records.get("metadatas", []) or []
                for record_id, document, metadata in zip(ids, documents, metadatas):
                    if isinstance(document, str) and isinstance(metadata, dict):
                        existing[str(record_id)] = (document, metadata)
        except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise RagIndexError("chroma_index_unavailable") from exc

        to_embed = []
        metadata_updates = []
        skipped = 0
        for chunk in chunks:
            record = existing.get(chunk.retrieval_chunk_id)
            desired_metadata = self._metadata(chunk)
            if record is None or record[0] != chunk.text:
                to_embed.append(chunk)
            elif record[1] != desired_metadata:
                metadata_updates.append((chunk.retrieval_chunk_id, desired_metadata))
            else:
                skipped += 1

        for start in range(0, len(metadata_updates), batch_size):
            batch = metadata_updates[start : start + batch_size]
            try:
                self.collection.update(
                    ids=[record_id for record_id, _ in batch],
                    metadatas=[metadata for _, metadata in batch],
                )
            except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
                raise RagIndexError("chroma_index_unavailable") from exc

        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start : start + batch_size]
            embedding_result = client.embed(
                [chunk.text for chunk in batch],
                dimensions=self.dimensions,
            )
            vectors = [row["embedding"] for row in embedding_result["data"]]
            if len(vectors) != len(batch) or any(len(vector) != self.dimensions for vector in vectors):
                raise RagIndexError("embedding_dimension_mismatch")
            try:
                self.collection.upsert(
                    ids=[chunk.retrieval_chunk_id for chunk in batch],
                    embeddings=vectors,
                    documents=[chunk.text for chunk in batch],
                    metadatas=[self._metadata(chunk) for chunk in batch],
                )
            except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
                raise RagIndexError("chroma_index_unavailable") from exc
        self.last_upsert_stats = UpsertStats(
            total=len(chunks),
            embedded=len(to_embed),
            metadata_updated=len(metadata_updates),
            skipped=skipped,
        )
        return len(chunks)

    def query(
        self,
        question: str,
        client: GLMClient,
        *,
        n_results: int = 5,
        run_id: str | None = None,
        verification_status: str | None = None,
        distance_threshold: float | None = None,
    ) -> list[RetrievalMatch]:
        if not question.strip():
            raise RagIndexError("empty_query")
        if not 1 <= n_results <= 50:
            raise RagIndexError("invalid_result_limit")
        if distance_threshold is not None and (
            not math.isfinite(distance_threshold) or distance_threshold < 0
        ):
            raise RagIndexError("invalid_distance_threshold")
        canonical_run_id = canonicalize_run_id(run_id) if run_id else None
        self._check_health()
        query_result = client.embed([question], dimensions=self.dimensions)
        query_vector = query_result["data"][0]["embedding"]
        filters = (
            [{"verification_status": verification_status}]
            if verification_status
            else [{"$or": [{"verification_status": "retrieved"}, {"verification_status": "verified"}]}]
        )
        if canonical_run_id:
            filters.append({"run_id": canonical_run_id})
        where = filters[0] if len(filters) == 1 else {"$and": filters}
        try:
            result = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except (chromadb.errors.ChromaError, httpx.HTTPError, RuntimeError, ValueError) as exc:
            raise RagIndexError("chroma_query_failed") from exc
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if not isinstance(metadata, dict) or not isinstance(document, str):
                raise RagIndexError("invalid_chroma_result")
            matches.append(
                RetrievalMatch(
                    retrieval_chunk_id=str(chunk_id),
                    text=document,
                    distance=float(distance),
                    evidence_chunk_id=str(metadata["evidence_chunk_id"]),
                    source_id=str(metadata["source_id"]),
                    run_id=str(metadata["run_id"]),
                    title=str(metadata.get("title", "")),
                    provider=str(metadata.get("provider", "")),
                    url=str(metadata.get("url", "")),
                    raw_ref=str(metadata.get("raw_ref", "")),
                    content_ref=str(metadata.get("content_ref", "")),
                    verification_status=str(metadata["verification_status"]),
                    position=int(metadata.get("position", 0)),
                    start_offset=int(metadata.get("start_offset", 0)),
                    end_offset=int(metadata.get("end_offset", 0)),
                )
            )
        if distance_threshold is None:
            return matches
        return [match for match in matches if match.distance <= distance_threshold]

    @staticmethod
    def _metadata(chunk: RetrievalChunk) -> dict[str, str | int]:
        return {
            "evidence_chunk_id": chunk.evidence_chunk_id,
            "run_id": chunk.run_id,
            "source_id": chunk.source_id,
            "position": chunk.position,
            "title": chunk.title,
            "url": chunk.url,
            "raw_ref": chunk.raw_ref,
            "content_ref": chunk.content_ref,
            "provider": chunk.provider,
            "start_offset": chunk.start_offset,
            "end_offset": chunk.end_offset,
            "verification_status": chunk.verification_status,
            "index_version": chunk.index_version,
        }
