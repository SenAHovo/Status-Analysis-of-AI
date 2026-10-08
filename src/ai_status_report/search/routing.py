"""Low-cost deterministic provider routing for research searches."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchRoute:
    providers: tuple[str, ...]
    reason: str


def choose_provider(*, query: str, preference: str = "auto", critical: bool = False) -> SearchRoute:
    if preference in {"tavily", "deepseek"}:
        return SearchRoute((preference,), "explicit_preference")
    if critical:
        return SearchRoute(("tavily", "deepseek"), "critical_fact_cross_check")
    if any(word in query.lower() for word in ("最新", "近期", "当前", "实时")):
        return SearchRoute(("deepseek",), "freshness_keyword")
    return SearchRoute(("tavily",), "default_research_search")
