"""GLM embedding and rerank clients."""

import math

from ai_status_report.model.client import ProviderClient, ProviderError
from ai_status_report.token_budget.estimator import estimate_tokens


class GLMClient(ProviderClient):
    def embed(self, texts: list[str], dimensions: int) -> dict:
        if not texts or len(texts) > 16 or any(not t or len(t) > 1000 for t in texts):
            raise ProviderError("embedding_input_bounds")
        result = self.accounted_post(
            "/embeddings",
            {"model": self.profile.model, "input": texts, "dimensions": dimensions},
            estimate=sum(estimate_tokens(text) for text in texts),
        )
        rows = result.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise ProviderError("embedding_count")
        try:
            if sorted(row["index"] for row in rows) != list(range(len(texts))):
                raise ProviderError("embedding_indices")
            for row in rows:
                vector = row["embedding"]
                if len(vector) != dimensions or not all(
                    type(v) in (int, float) and math.isfinite(v) for v in vector
                ):
                    raise ProviderError("embedding_values")
        except (KeyError, TypeError):
            raise ProviderError("embedding_schema") from None
        result["data"] = sorted(rows, key=lambda row: row["index"])
        return result

    def rerank(self, query: str, documents: list[str], *, top_n: int = 0) -> dict:
        """Score candidate documents with GLM Rerank.

        Official API: https://docs.bigmodel.cn/api-reference/模型-api/文本重排序
        """

        if not query.strip() or len(query) > 4096:
            raise ProviderError("rerank_query_bounds")
        if not 1 <= len(documents) <= 128 or any(
            not document or len(document) > 4096 for document in documents
        ):
            raise ProviderError("rerank_document_bounds")
        if top_n < 0 or top_n > len(documents):
            raise ProviderError("rerank_top_n_bounds")
        result = self.accounted_post(
            "/rerank",
            {
                "model": "rerank",
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": False,
            },
            estimate=estimate_tokens(query) + sum(estimate_tokens(document) for document in documents),
        )
        rows = result.get("results")
        if not isinstance(rows, list) or len(rows) > len(documents):
            raise ProviderError("rerank_schema")
        try:
            indexes = [int(row["index"]) for row in rows]
            scores = [float(row["relevance_score"]) for row in rows]
        except (KeyError, TypeError, ValueError):
            raise ProviderError("rerank_schema") from None
        if (
            len(indexes) != len(set(indexes))
            or any(index < 0 or index >= len(documents) for index in indexes)
            or any(not math.isfinite(score) for score in scores)
        ):
            raise ProviderError("rerank_schema")
        return result
