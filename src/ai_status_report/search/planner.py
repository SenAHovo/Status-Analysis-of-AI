"""Model-generated and deterministic SearchPlan construction."""

from __future__ import annotations

import hashlib
import json

from ai_status_report.model.client import DeepSeekClient, ProviderError
from ai_status_report.model.router import ResponseCache
from ai_status_report.model.structured import StructuredOutputError, parse_structured, schema_hint
from ai_status_report.schemas.common import new_id
from ai_status_report.schemas.search import (
    SearchLimits,
    SearchPlan,
    SearchQuery,
    SearchRouteDecision,
)
from ai_status_report.search.routing import choose_provider


def _plan_cache_key(model: str, messages: list[dict], options: dict) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "options": options},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_plan(task: dict[str, object]) -> SearchPlan:
    query = str(task.get("query", "")).strip()
    preference = str(task.get("source_preference", "auto"))
    evidence_requirement = str(task.get("evidence_requirement", "none"))
    if evidence_requirement not in {"none", "retrievable"}:
        evidence_requirement = "none"
    if evidence_requirement == "retrievable" and preference == "auto":
        route_providers = ("tavily",)
        route_reason = "retrievable_evidence_required"
    else:
        route = choose_provider(
            query=query,
            preference=preference,
            critical=bool(task.get("critical", False)),
        )
        route_providers = route.providers
        route_reason = route.reason
    try:
        requested_budget = int(task.get("max_output_tokens", 65536))
    except (TypeError, ValueError):
        requested_budget = 65536
    requested_budget = min(max(requested_budget, 1024), 384000)
    try:
        max_results = min(max(int(task.get("max_results", 5)), 1), 10)
    except (TypeError, ValueError):
        max_results = 5
    return SearchPlan(
        plan_id=new_id("plan"),
        original_query=query,
        queries=[SearchQuery(query=query, purpose="原始研究问题")],
        providers=list(route_providers),
        evidence_requirement=evidence_requirement,
        limits=SearchLimits(
            max_results_per_query=max_results,
            max_output_tokens=requested_budget,
            max_parallel_providers=min(len(route_providers), 2),
        ),
        need_cross_validation=len(route_providers) > 1,
        reason=route_reason,
        source="deterministic",
    )


class SearchPlanner:
    def __init__(
        self,
        client: DeepSeekClient | None,
        cache: ResponseCache | None = None,
        *,
        max_output_tokens: int = 512,
    ):
        self.client = client
        self.cache = cache
        self.max_output_tokens = max_output_tokens

    def plan(self, task: dict[str, object], *, allow_model: bool = True) -> SearchPlan:
        deterministic = deterministic_plan(task)
        if not allow_model or self.client is None:
            return deterministic
        query = str(task.get("query", "")).strip()
        evidence_requirement = deterministic.evidence_requirement
        messages = [
            {
                "role": "system",
                "content": (
                    "你是网络搜索 Agent 的搜索路由器。只输出 JSON，不输出 Markdown。"
                    "根据用户研究问题选择一个最适合的搜索 Provider。只能选择 tavily 或 deepseek。"
                    "中文主题可优先 deepseek，英文论文、官方文档和全球协议可优先 tavily。"
                    "当下游要求可检索正文证据时，必须选择 tavily：当前只有它会抓取并持久化网页正文，"
                    "供后续 EvidenceChunk 和 RAG 使用。"
                    f"输出契约：{schema_hint(SearchRouteDecision)}。"
                    '示例：{"provider":"tavily","reason":"需要英文官方文档"}。'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"研究问题：{query}\n"
                    f"evidence_requirement：{evidence_requirement}"
                ),
            },
        ]
        options = {
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "max_tokens": self.max_output_tokens,
        }
        key = _plan_cache_key(self.client.profile.model, messages, options)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                try:
                    return SearchPlan.model_validate(cached).model_copy(update={"source": "model_cache"})
                except (TypeError, ValueError):
                    return deterministic.model_copy(
                        update={"source": "deterministic_fallback", "validation_warnings": ["model_plan_fallback:cache_schema_mismatch"]}
                    )
        try:
            response = self.client.chat(messages, **options)
            content = response["choices"][0]["message"]["content"]
            decision = parse_structured(content, SearchRouteDecision)
            plan = deterministic.model_copy(
                update={
                    "providers": [decision.provider],
                    "need_cross_validation": False,
                    "reason": decision.reason,
                    "source": "model",
                    "original_query": query,
                    "evidence_requirement": evidence_requirement,
                }
            )
        except StructuredOutputError as exc:
            return deterministic.model_copy(
                update={
                    "source": "deterministic_fallback",
                    "validation_warnings": [f"model_plan_fallback:{exc!s}"],
                }
            )
        except ProviderError:
            return deterministic.model_copy(
                update={
                    "source": "deterministic_fallback",
                    "validation_warnings": ["model_plan_fallback:provider_error"],
                }
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return deterministic.model_copy(
                update={
                    "source": "deterministic_fallback",
                    "validation_warnings": ["model_plan_fallback:invalid_model_response"],
                }
            )
        if self.cache is not None:
            self.cache.put(key, plan.model_dump(mode="json"))
        return plan
