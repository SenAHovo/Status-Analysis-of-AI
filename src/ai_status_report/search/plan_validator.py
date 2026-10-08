"""Program-side authorization and limit validation for SearchPlan."""

from __future__ import annotations

from ai_status_report.schemas.search import SearchLimits, SearchPlan


class SearchPlanError(ValueError):
    """Safe plan validation error code."""


def validate_plan(
    plan: SearchPlan,
    task: dict[str, object],
    *,
    available_providers: set[str],
    configured_budget: int,
) -> SearchPlan:
    explicit = str(task.get("source_preference", "auto"))
    evidence_requirement = str(task.get("evidence_requirement", "none"))
    if evidence_requirement not in {"none", "retrievable"}:
        raise SearchPlanError("invalid_evidence_requirement")
    if evidence_requirement == "retrievable" and "tavily" not in available_providers:
        raise SearchPlanError("retrievable_evidence_provider_not_configured")
    if evidence_requirement == "retrievable" and explicit == "deepseek":
        raise SearchPlanError("retrievable_evidence_conflicts_with_provider")

    if explicit == "both":
        required = {"tavily", "deepseek"}
        if not required.issubset(available_providers):
            raise SearchPlanError("requested_providers_not_configured")
        providers = ["tavily", "deepseek"]
    elif explicit in {"tavily", "deepseek"}:
        if explicit not in available_providers:
            raise SearchPlanError("requested_provider_not_configured")
        providers = [explicit]
    elif evidence_requirement == "retrievable":
        providers = ["tavily"]
    else:
        providers = list(
            dict.fromkeys(provider for provider in plan.providers if provider in available_providers)
        )
        if not providers:
            providers = [provider for provider in ("tavily", "deepseek") if provider in available_providers]
            if not providers:
                raise SearchPlanError("no_search_provider_available")
        providers = providers[:2]

    try:
        requested_results = max(int(task.get("max_results", 5)), 1)
        requested_budget = int(task.get("max_output_tokens", configured_budget))
    except (TypeError, ValueError):
        raise SearchPlanError("invalid_search_limits") from None
    max_results = min(plan.limits.max_results_per_query, requested_results, 10)
    if requested_budget < 1024:
        raise SearchPlanError("invalid_search_budget")
    max_tokens = min(plan.limits.max_output_tokens, requested_budget, configured_budget, 384000)
    if max_tokens < 1024:
        raise SearchPlanError("invalid_search_budget")
    queries = list(dict.fromkeys(plan.queries))[: plan.limits.max_queries]
    if not queries:
        raise SearchPlanError("empty_search_plan")
    warnings = list(plan.validation_warnings)
    if providers != plan.providers:
        warnings.append(
            "provider_required_for_retrievable_evidence"
            if evidence_requirement == "retrievable"
            else "provider_reduced_to_available"
        )
    return plan.model_copy(
        update={
            "original_query": str(task.get("query", plan.original_query)),
            "queries": queries,
            "providers": providers,
            "evidence_requirement": evidence_requirement,
            "limits": SearchLimits(
                max_queries=len(queries),
                max_results_per_query=max_results,
                max_output_tokens=max_tokens,
                max_parallel_providers=min(plan.limits.max_parallel_providers, len(providers)),
            ),
            "need_cross_validation": plan.need_cross_validation and len(providers) > 1,
            "validation_warnings": sorted(set(warnings)),
        }
    )
