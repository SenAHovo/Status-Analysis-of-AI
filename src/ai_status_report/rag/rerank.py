"""GLM Rerank integration for retrieved evidence candidates."""

from __future__ import annotations

from ai_status_report.model.client import ProviderError
from ai_status_report.rag.glm import GLMClient
from ai_status_report.rag.schemas import RetrievalMatch


def rerank_matches(
    query: str,
    matches: list[RetrievalMatch],
    client: GLMClient,
    *,
    top_n: int = 0,
) -> list[RetrievalMatch]:
    """Return matches in GLM relevance order while preserving provenance."""

    if not matches:
        return []
    if top_n < 0 or top_n > len(matches):
        raise ProviderError("rerank_top_n_bounds")
    response = client.rerank(query, [match.text for match in matches], top_n=top_n)
    rows = response.get("results")
    if not isinstance(rows, list) or not rows:
        raise ProviderError("rerank_empty_results")
    ranked: list[RetrievalMatch] = []
    seen: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ProviderError("rerank_schema")
        try:
            index = int(row["index"])
            score = float(row["relevance_score"])
        except (KeyError, TypeError, ValueError):
            raise ProviderError("rerank_schema") from None
        if index in seen or not 0 <= index < len(matches):
            raise ProviderError("rerank_schema")
        seen.add(index)
        ranked.append(matches[index].model_copy(update={"rerank_score": score}))
    return ranked
