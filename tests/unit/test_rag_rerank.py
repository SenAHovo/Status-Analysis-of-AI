import pytest

from ai_status_report.model.client import ProviderError
from ai_status_report.rag.rerank import rerank_matches
from ai_status_report.rag.schemas import RetrievalMatch


def make_match(chunk_id: str, text: str) -> RetrievalMatch:
    return RetrievalMatch(
        retrieval_chunk_id=chunk_id,
        text=text,
        distance=0.5,
        evidence_chunk_id=f"e-{chunk_id}",
        source_id="source-1",
        run_id="run-1",
        verification_status="retrieved",
    )


class FakeRerankClient:
    def __init__(self, response):
        self.response = response
        self.documents = None

    def rerank(self, query, documents, *, top_n):
        self.documents = (query, documents, top_n)
        return self.response


def test_rerank_preserves_match_identity_and_reorders():
    client = FakeRerankClient(
        {"results": [{"index": 1, "relevance_score": 0.91}, {"index": 0, "relevance_score": 0.2}]}
    )
    matches = [make_match("a", "甲"), make_match("b", "乙")]

    ranked = rerank_matches("问题", matches, client, top_n=2)

    assert [item.retrieval_chunk_id for item in ranked] == ["b", "a"]
    assert ranked[0].rerank_score == 0.91
    assert client.documents == ("问题", ["甲", "乙"], 2)


def test_rerank_rejects_invalid_provider_indexes():
    client = FakeRerankClient({"results": [{"index": 3, "relevance_score": 0.9}]})
    with pytest.raises(ProviderError, match="rerank_schema"):
        rerank_matches("问题", [make_match("a", "甲")], client)
