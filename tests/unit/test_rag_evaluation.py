from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_status_report.rag.evaluation import (
    EvaluationError,
    evaluate_retrieval,
    load_evaluation_set,
)
from ai_status_report.rag.schemas import RetrievalMatch


def _match(source_id: str, distance: float) -> RetrievalMatch:
    return RetrievalMatch(
        retrieval_chunk_id=f"retrieval-{source_id}",
        evidence_chunk_id=f"evidence-{source_id}",
        source_id=source_id,
        run_id="run-eval-001",
        text=f"Evidence text from {source_id}.",
        distance=distance,
        verification_status="retrieved",
    )


class _Index:
    def __init__(self, matches: list[RetrievalMatch]) -> None:
        self.matches = matches
        self.calls: list[dict[str, Any]] = []

    def query(self, question: str, client: Any, **kwargs: Any) -> list[RetrievalMatch]:
        self.calls.append({"question": question, **kwargs})
        return self.matches


def _evaluation_set() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "name": "unit-evaluation",
        "run_id": "run-eval-001",
        "index_version": "v2",
        "cases": [
            {
                "case_id": "first",
                "query": "first question",
                "expected_source_ids": ["expected"],
            },
            {
                "case_id": "second",
                "query": "second question",
                "expected_source_ids": ["other"],
            },
        ],
    }


def test_evaluate_retrieval_compares_one_candidate_set_per_case() -> None:
    index = _Index([_match("other", 0.1), _match("expected", 0.4)])

    result = evaluate_retrieval(
        _evaluation_set(),
        index,
        object(),
        n_results=12,
        distance_threshold=0.2,
        rerank=lambda _query, matches, _client: list(reversed(matches)),
    )

    assert [call["question"] for call in index.calls] == ["first question", "second question"]
    assert all(call["n_results"] == 12 for call in index.calls)
    assert result["metrics"]["vector"]["source_hit_at_1"] == 0.5
    assert result["metrics"]["vector"]["mean_reciprocal_rank"] == 0.75
    assert result["metrics"]["threshold"]["source_hit_at_1"] == 0.5
    assert result["metrics"]["rerank"]["source_hit_at_1"] == 0.5
    assert result["cases"][0]["modes"]["rerank"][0]["source_id"] == "expected"
    assert result["cases"][0]["modes"]["threshold"][0]["source_id"] == "other"


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_evaluate_retrieval_rejects_invalid_distance_threshold(value: float) -> None:
    with pytest.raises(EvaluationError, match="invalid_distance_threshold"):
        evaluate_retrieval(_evaluation_set(), _Index([]), object(), n_results=1, distance_threshold=value)


def test_load_evaluation_set_rejects_duplicate_case_id(tmp_path: Path) -> None:
    payload = _evaluation_set()
    payload["cases"].append(payload["cases"][0])
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="invalid_evaluation_set"):
        load_evaluation_set(path)
