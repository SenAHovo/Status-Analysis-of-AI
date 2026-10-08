from datetime import UTC, datetime

import pytest

from ai_status_report.rag.chroma import ChromaEvidenceIndex, RagIndexError
from ai_status_report.rag.chunking import to_retrieval_chunks
from ai_status_report.schemas.search import EvidenceChunk


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts, *, dimensions):
        self.calls.append(list(texts))
        return {
            "data": [
                {"index": index, "embedding": [float(index == 0)] + [0.0] * (dimensions - 1)}
                for index, _ in enumerate(texts)
            ]
        }


def _evidence(text: str, source_id: str = "source-1", chunk_id: str = "chunk-1"):
    return EvidenceChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        text=text,
        title="Test source",
        url="https://example.com/source",
        provider="tavily_mcp",
        position=0,
        content_hash="hash-1",
        raw_ref="raw.json",
        content_ref="page.md",
        created_at=datetime.now(UTC),
        verification_status="retrieved",
    )


def test_retrieval_chunks_keep_parent_and_bound_embedding_input():
    chunks = to_retrieval_chunks(_evidence("段落一。" * 400), run_id="run-1")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 900 for chunk in chunks)
    assert all(chunk.evidence_chunk_id == "chunk-1" for chunk in chunks)
    assert all(chunk.content_ref == "page.md" for chunk in chunks)
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))


def test_chroma_upsert_is_idempotent_and_query_filters_run(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)
    assert index.index_version == "v2"
    assert index.distance_space == "cosine"
    assert index.collection.configuration["hnsw"]["space"] == "cosine"
    chunks = to_retrieval_chunks(_evidence("人工智能与电力系统交互。" * 100), run_id="run-1")

    assert index.upsert(chunks, client) == len(chunks)
    assert index.upsert(chunks, client) == len(chunks)
    assert len(client.calls) == 1
    assert index.last_upsert_stats.embedded == 0
    assert index.last_upsert_stats.skipped == len(chunks)
    assert index.collection.count() == len(chunks)


def test_chroma_reembeds_only_new_or_changed_text(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)
    first = to_retrieval_chunks(_evidence("原始内容"), run_id="run-1")
    changed = to_retrieval_chunks(_evidence("更新内容"), run_id="run-1")
    added = to_retrieval_chunks(
        _evidence("新增内容", source_id="source-2", chunk_id="chunk-2"), run_id="run-1"
    )

    index.upsert(first, client)
    index.upsert([*changed, *added], client)

    assert len(client.calls) == 2
    assert client.calls[-1] == ["更新内容", "新增内容"]
    assert index.last_upsert_stats.embedded == 2
    assert index.last_upsert_stats.skipped == 0


def test_chroma_updates_metadata_without_reembedding(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)
    original = to_retrieval_chunks(_evidence("固定内容"), run_id="run-1")
    metadata_changed = [original[0].model_copy(update={"title": "Updated title"})]

    index.upsert(original, client)
    index.upsert(metadata_changed, client)

    assert len(client.calls) == 1
    assert index.last_upsert_stats.embedded == 0
    assert index.last_upsert_stats.metadata_updated == 1

    matches = index.query("电力系统交互", client, n_results=3, run_id="run-1")
    assert matches
    assert all(match.run_id == "run-1" for match in matches)
    assert all(match.content_ref == "page.md" for match in matches)


def test_chroma_rejects_existing_collection_metadata_mismatch(tmp_path):
    ChromaEvidenceIndex(tmp_path, dimensions=2048)

    with pytest.raises(RagIndexError, match="index_metadata_mismatch"):
        ChromaEvidenceIndex(tmp_path, dimensions=1024)


def test_chroma_rejects_existing_collection_configuration_mismatch(tmp_path):
    index = ChromaEvidenceIndex(tmp_path)
    index.client.delete_collection(index.collection.name)
    index.client.get_or_create_collection(
        name="evidence_chunks_v2",
        metadata={
            "index_version": "v2",
            "embedding_model": "embedding-3",
            "embedding_dimensions": 2048,
            "distance_space": "cosine",
        },
        configuration={"hnsw": {"space": "l2"}},
    )

    with pytest.raises(RagIndexError, match="index_configuration_mismatch"):
        ChromaEvidenceIndex(tmp_path)


def test_chroma_rejects_v2_collection_without_metric_metadata(tmp_path):
    index = ChromaEvidenceIndex(tmp_path)
    index.client.delete_collection(index.collection.name)
    index.client.get_or_create_collection(
        name="evidence_chunks_v2",
        metadata={
            "index_version": "v2",
            "embedding_model": "embedding-3",
            "embedding_dimensions": 2048,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )

    with pytest.raises(RagIndexError, match="index_metadata_mismatch"):
        ChromaEvidenceIndex(tmp_path)


def test_chroma_opens_legacy_v1_l2_collection_without_metric_metadata(tmp_path):
    seed = ChromaEvidenceIndex(tmp_path)
    seed.client.delete_collection(seed.collection.name)
    seed.client.get_or_create_collection(
        name="evidence_chunks_v1",
        metadata={
            "index_version": "v1",
            "embedding_model": "embedding-3",
            "embedding_dimensions": 2048,
        },
        configuration={"hnsw": {"space": "l2"}},
    )
    index = ChromaEvidenceIndex(tmp_path, index_version="v1")

    assert index.distance_space == "l2"
    assert index.collection.configuration["hnsw"]["space"] == "l2"


def test_chroma_rejects_legacy_index_writes_and_version_mismatched_chunks(tmp_path):
    client = FakeEmbeddingClient()
    legacy_index = ChromaEvidenceIndex(tmp_path, index_version="v1")
    legacy_chunks = to_retrieval_chunks(_evidence("历史资料"), run_id="run-1", index_version="v1")
    with pytest.raises(RagIndexError, match="legacy_index_read_only"):
        legacy_index.upsert(legacy_chunks, client)

    current_index = ChromaEvidenceIndex(tmp_path)
    with pytest.raises(RagIndexError, match="retrieval_chunk_index_version_mismatch"):
        current_index.upsert(legacy_chunks, client)


def test_chroma_rejects_overlong_index_version(tmp_path):
    with pytest.raises(RagIndexError, match="invalid_index_version"):
        ChromaEvidenceIndex(tmp_path, index_version="v" * 65)


def test_chroma_query_canonicalizes_run_id(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)
    chunks = to_retrieval_chunks(_evidence("交互内容"), run_id="run test/001")

    index.upsert(chunks, client)

    matches = index.query("交互", client, run_id="run test/001")

    assert len(matches) == 1
    assert matches[0].run_id == "run-test-001"


def test_chroma_query_filters_by_distance_threshold(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **_kwargs):
            metadata = {
                "evidence_chunk_id": "chunk-1",
                "source_id": "source-1",
                "run_id": "run-1",
                "verification_status": "retrieved",
            }
            return {
                "ids": [["near", "far"]],
                "documents": [["near text", "far text"]],
                "metadatas": [[metadata, metadata]],
                "distances": [[0.2, 0.8]],
            }

    index.collection = FakeCollection()
    matches = index.query("question", client, n_results=2, distance_threshold=0.5)

    assert [match.retrieval_chunk_id for match in matches] == ["near"]


def test_chroma_http_client_is_used_for_shared_agent_store(monkeypatch, tmp_path):
    local = ChromaEvidenceIndex(tmp_path)
    calls = []

    def fake_http_client(*, host, port):
        calls.append((host, port))
        return local.client

    monkeypatch.setattr("ai_status_report.rag.chroma.chromadb.HttpClient", fake_http_client)
    index = ChromaEvidenceIndex(tmp_path, client_mode="http", host="127.0.0.1", port=8000)

    assert index.collection.name == "evidence_chunks_v2"
    assert calls == [("127.0.0.1", 8000)]
    assert index.healthcheck() == 0


def test_chroma_http_client_failure_is_a_safe_rag_error(monkeypatch, tmp_path):
    def unavailable(*, host, port):
        raise RuntimeError("connection details must not escape")

    monkeypatch.setattr("ai_status_report.rag.chroma.chromadb.HttpClient", unavailable)

    with pytest.raises(RagIndexError, match="chroma_client_unavailable"):
        ChromaEvidenceIndex(tmp_path, client_mode="http")


def test_chroma_rejects_invalid_distance_threshold(tmp_path):
    client = FakeEmbeddingClient()
    index = ChromaEvidenceIndex(tmp_path)
    with pytest.raises(RagIndexError, match="invalid_distance_threshold"):
        index.query("question", client, distance_threshold=-0.1)
