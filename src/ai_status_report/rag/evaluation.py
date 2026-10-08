"""Small, reviewable retrieval evaluation sets and automated comparisons."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ai_status_report.rag.rerank import rerank_matches
from ai_status_report.rag.schemas import RetrievalMatch

EVALUATION_SCHEMA_VERSION = "1"
_METRIC_CUTOFFS = (1, 3, 5)


class EvaluationError(ValueError):
    """Fixed errors for evaluation-set and result-contract boundaries."""


class RetrievalIndex(Protocol):
    def query(self, question: str, client: Any, **kwargs: Any) -> list[RetrievalMatch]: ...


def load_evaluation_set(path: Path) -> dict[str, Any]:
    """Load a human-reviewed source-level evaluation set."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise EvaluationError("invalid_evaluation_set") from None
    if not isinstance(payload, dict) or payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("invalid_evaluation_set")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise EvaluationError("invalid_evaluation_set")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise EvaluationError("invalid_evaluation_set")
    if not isinstance(payload.get("index_version"), str) or not payload["index_version"].strip():
        raise EvaluationError("invalid_evaluation_set")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("invalid_evaluation_set")
    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise EvaluationError("invalid_evaluation_set")
        case_id = case.get("case_id")
        query = case.get("query")
        expected = case.get("expected_source_ids")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_ids
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(expected, list)
            or not expected
            or any(not isinstance(source_id, str) or not source_id for source_id in expected)
        ):
            raise EvaluationError("invalid_evaluation_set")
        seen_ids.add(case_id)
    return payload


def evaluate_retrieval(
    evaluation_set: dict[str, Any],
    index: RetrievalIndex,
    client: Any,
    *,
    n_results: int,
    distance_threshold: float | None = None,
    rerank: Callable[[str, list[RetrievalMatch], Any], list[RetrievalMatch]] = rerank_matches,
) -> dict[str, Any]:
    """Run one vector call per case, then compare local threshold and rerank modes."""

    if not 1 <= n_results <= 50:
        raise EvaluationError("invalid_result_limit")
    if distance_threshold is not None and (
        not math.isfinite(distance_threshold) or distance_threshold < 0
    ):
        raise EvaluationError("invalid_distance_threshold")

    case_results = []
    mode_results: dict[str, list[dict[str, Any]]] = {"vector": [], "rerank": []}
    if distance_threshold is not None:
        mode_results["threshold"] = []

    for case in evaluation_set["cases"]:
        vector_matches = index.query(
            case["query"],
            client,
            n_results=n_results,
            run_id=evaluation_set["run_id"],
        )
        reranked_matches = rerank(case["query"], vector_matches, client)
        expected_source_ids = set(case["expected_source_ids"])
        modes = {"vector": vector_matches, "rerank": reranked_matches}
        if distance_threshold is not None:
            modes["threshold"] = [
                match for match in vector_matches if match.distance <= distance_threshold
            ]
        result = {
            "case_id": case["case_id"],
            "query": case["query"],
            "expected_source_ids": case["expected_source_ids"],
            "review_note": case.get("review_note", ""),
            "modes": {},
        }
        for name, matches in modes.items():
            rendered = _render_matches(matches, expected_source_ids)
            result["modes"][name] = rendered
            mode_results[name].append(rendered)
        case_results.append(result)

    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evaluation_set": evaluation_set["name"],
        "run_id": evaluation_set["run_id"],
        "index_version": evaluation_set["index_version"],
        "n_results": n_results,
        "distance_threshold": distance_threshold,
        "cases": case_results,
        "metrics": {name: _metrics(results) for name, results in mode_results.items()},
    }


def _render_matches(
    matches: list[RetrievalMatch], expected_source_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "retrieval_chunk_id": match.retrieval_chunk_id,
            "source_id": match.source_id,
            "title": match.title,
            "url": match.url,
            "distance": match.distance,
            "rerank_score": match.rerank_score,
            "is_expected_source": match.source_id in expected_source_ids,
            "text_excerpt": " ".join(match.text.split())[:240],
        }
        for rank, match in enumerate(matches, start=1)
    ]


def _metrics(cases: list[list[dict[str, Any]]]) -> dict[str, float | int]:
    total = len(cases)
    ranks = [
        next((item["rank"] for item in matches if item["is_expected_source"]), None)
        for matches in cases
    ]
    return {
        "cases": total,
        **{
            f"source_hit_at_{cutoff}": round(
                sum(rank is not None and rank <= cutoff for rank in ranks) / total, 4
            )
            for cutoff in _METRIC_CUTOFFS
        },
        "mean_reciprocal_rank": round(
            sum(1 / rank if rank is not None else 0 for rank in ranks) / total, 4
        ),
    }
