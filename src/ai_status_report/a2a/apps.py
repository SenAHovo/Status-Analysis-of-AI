"""Local HTTP applications for the two specialist agents."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from dotenv import load_dotenv

from ai_status_report.a2a.server import SpecialistExecutor, build_a2a_app, build_agent_card
from ai_status_report.agents.document import DocumentTaskExecutor
from ai_status_report.mcp.client import TavilyMCPClient, TavilyRateLimitedError, TavilyRequestPacer
from ai_status_report.model.client import DeepSeekClient
from ai_status_report.model.router import ResponseCache
from ai_status_report.model.search_cache import WebSearchCache, cached_web_search
from ai_status_report.schemas.search import ResearchReport
from ai_status_report.search.adapters import deepseek_report, tavily_report
from ai_status_report.search.extraction import normalize_extract_response, select_extract_sources
from ai_status_report.search.merge import merge_reports
from ai_status_report.search.plan_validator import SearchPlanError, validate_plan
from ai_status_report.search.planner import SearchPlanner
from ai_status_report.search.synthesis import synthesize_report
from ai_status_report.settings import ConfigError, load_search_settings
from ai_status_report.storage.search_results import (
    canonicalize_url,
    persist_evidence_chunks,
    persist_raw_search_response,
    persist_research_report,
    persist_web_content,
)
from ai_status_report.token_budget.allocator import SEARCH_REQUESTS
from ai_status_report.token_budget.config import task_budget
from ai_status_report.token_budget.runtime import new_run_ledger

load_dotenv()


def network_search_app():
    card = build_agent_card(
        name="Network Search Agent",
        description="Collects and organizes research sources.",
        url="http://127.0.0.1:8001/",
    )
    settings = load_search_settings(Path.cwd())
    tavily_key = settings.tavily_api_key
    tavily_url = settings.tavily_mcp_url
    search = None
    tavily = (
        TavilyMCPClient(tavily_url, tavily_key, pacer=TavilyRequestPacer())
        if tavily_key
        else None
    )
    cache = WebSearchCache(Path.cwd() / "data" / "cache" / "web_search.sqlite3")
    plan_cache = ResponseCache(Path.cwd() / "data" / "cache" / "search_plan.sqlite3")
    synthesis_cache = ResponseCache(Path.cwd() / "data" / "cache" / "search_synthesis.sqlite3")

    async def run_search(
        task: dict[str, object],
        *,
        progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        root = Path.cwd()
        run_id = str(task.get("run_id") or "").strip() or None
        if not tavily and not settings.deepseek:
            raise RuntimeError("no_search_provider_configured")
        if not run_id:
            raise RuntimeError("search_run_id_missing")
        ledger = new_run_ledger(run_id, root=root)
        deepseek = (
            DeepSeekClient(settings.deepseek, settings.timeout, ledger=ledger)
            if settings.deepseek
            else None
        )
        planner = SearchPlanner(
            deepseek,
            plan_cache,
            max_output_tokens=task_budget(
                root,
                settings.deepseek.model if settings.deepseek else "deepseek-v4-flash",
                "search_planning",
            ).max_output_tokens,
        )
        try:
            plan = await asyncio.to_thread(planner.plan, task)
            available = {
                provider
                for provider, configured in (("tavily", tavily), ("deepseek", deepseek))
                if configured
            }
            try:
                plan = validate_plan(
                    plan,
                    task,
                    available_providers=available,
                    configured_budget=task_budget(
                        root,
                        settings.deepseek.model if settings.deepseek else "deepseek-v4-flash",
                        "search_execution",
                    ).max_output_tokens,
                )
            except SearchPlanError as exc:
                raise RuntimeError(str(exc)) from None

            async def execute_provider(provider: str) -> tuple[list[ResearchReport], list[str]]:
                reports = []
                raw_refs = []
                for search_query in plan.queries:
                    try:
                        extract_refs: list[str] = []
                        extracted_content_by_source_id: dict[str, str] = {}
                        if provider == "tavily":
                            if tavily is None:
                                raise RuntimeError("tavily_not_configured")
                            async def report_rate_limit(_error, retry_attempt: int, delay: int) -> None:
                                if progress is not None:
                                    await progress(f"tavily_rate_limited_wait:{delay}:{retry_attempt}")

                            result = await tavily.call(
                                "tavily_search",
                                {
                                    "query": search_query.query,
                                    "max_results": plan.limits.max_results_per_query,
                                },
                                on_attempt=lambda: ledger.meter(SEARCH_REQUESTS),
                                on_rate_limit=report_rate_limit,
                            )
                            provider_name = "tavily_mcp"
                            report = tavily_report(search_query.query, result)
                            if report.sources:
                                selected = select_extract_sources(
                                    report.sources,
                                    limit=min(plan.limits.max_results_per_query, 20),
                                )
                                urls = [source.url for source in selected]
                                extracted = None
                                extract_depth = "basic"
                                for depth in ("basic", "advanced"):
                                    try:
                                        extracted = await tavily.call(
                                            "tavily_extract",
                                            {
                                                "urls": urls,
                                                "extract_depth": depth,
                                                "format": "markdown",
                                                "include_images": False,
                                            },
                                            on_attempt=lambda: ledger.meter(SEARCH_REQUESTS),
                                            on_rate_limit=report_rate_limit,
                                        )
                                    except TavilyRateLimitedError:
                                        raise
                                    except RuntimeError:
                                        extracted = None
                                    extract_depth = depth
                                    if normalize_extract_response(extracted):
                                        break
                                extracted_items = normalize_extract_response(extracted)
                                if extracted_items:
                                    extract_ref = str(
                                        persist_raw_search_response(
                                            root,
                                            query=search_query.query,
                                            provider="tavily_mcp_extract",
                                            result=extracted,
                                            run_id=run_id,
                                        )
                                    )
                                    extract_refs.append(extract_ref)
                                    extracted_by_url = {
                                        canonicalize_url(item["url"]): item
                                        for item in extracted_items
                                    }
                                    updated_sources = []
                                    for source in report.sources:
                                        item = extracted_by_url.get(canonicalize_url(source.url))
                                        if item is None:
                                            updated_sources.append(source)
                                            continue
                                        content_ref = persist_web_content(
                                            root,
                                            source=source.model_dump(),
                                            content=item["raw_content"],
                                            extract_ref=extract_ref,
                                            extract_depth=extract_depth,
                                            run_id=run_id,
                                        )
                                        updated_sources.append(
                                            source.model_copy(
                                                update={
                                                    "snippet": item["raw_content"][:1200],
                                                    "content_ref": str(content_ref),
                                                    "verification_status": "retrieved",
                                                }
                                            )
                                        )
                                        extracted_content_by_source_id[source.source_id] = item["raw_content"]
                                    report = report.model_copy(update={"sources": updated_sources})
                        else:
                            if deepseek is None:
                                raise RuntimeError("deepseek_not_configured")
                            result = await asyncio.to_thread(
                                cached_web_search,
                                deepseek,
                                cache,
                                search_query.query,
                                max_output_tokens=plan.limits.max_output_tokens,
                            )
                            provider_name = "deepseek_native_web_search"
                            report = deepseek_report(search_query.query, result)
                        raw_ref = str(
                            persist_raw_search_response(
                                root,
                                query=search_query.query,
                                provider=provider_name,
                                result=result,
                                run_id=run_id,
                            )
                        )
                        chunks = persist_evidence_chunks(
                            root,
                            [source.model_dump() for source in report.sources],
                            raw_ref=extract_refs[-1] if extract_refs else raw_ref,
                            content_by_source_id=extracted_content_by_source_id,
                            run_id=run_id,
                        )
                        report = report.model_copy(
                            update={
                                # All EvidenceChunks remain on disk; the A2A report carries
                                # only the bounded handoff reference list.
                                "evidence_refs": [chunk.chunk_id for chunk in chunks][:100],
                                "sources": [
                                    source.model_copy(update={"raw_ref": raw_ref})
                                    for source in report.sources
                                ],
                                "raw_ref": raw_ref,
                                "raw_refs": [raw_ref, *extract_refs][:10],
                            }
                        )
                        reports.append(report)
                        raw_refs.extend([raw_ref, *extract_refs])
                    except TavilyRateLimitedError as exc:
                        reports.append(
                            ResearchReport(
                                query=search_query.query,
                                provider="tavily_mcp",
                                status="failed",
                                incomplete_reason=str(exc),
                            )
                        )
                        # The bounded retry has already been spent inside the
                        # single tool call. Do not issue later queries for the
                        # same API key while it remains rate limited.
                        break
                    except (ConfigError, RuntimeError, ValueError) as exc:
                        reports.append(
                            ResearchReport(
                                query=search_query.query,
                                provider=(
                                    "tavily_mcp" if provider == "tavily" else "deepseek_native_web_search"
                                ),
                                status="failed",
                                incomplete_reason=str(exc),
                            )
                        )
                return reports, raw_refs

            results = await asyncio.gather(*(execute_provider(provider) for provider in plan.providers))
            reports = [report for provider_reports, _ in results for report in provider_reports]
            raw_refs = [raw_ref for _, provider_refs in results for raw_ref in provider_refs][:10]
            report = merge_reports(reports, plan, raw_refs=raw_refs)
            report = await asyncio.to_thread(
                synthesize_report,
                deepseek,
                synthesis_cache,
                report,
                max_output_tokens=(
                    task_budget(root, settings.deepseek.model, "search_synthesis").max_output_tokens
                    if settings.deepseek else 4096
                ),
            )
            persist_research_report(root, report, run_id=run_id)
            return report.model_dump_json()
        finally:
            if deepseek is not None:
                deepseek.http.close()

    # Keep the A2A endpoint available for a clear protocol-level failure, but
    # never let an unconfigured Search Agent acknowledge a task as completed.
    search = run_search
    return build_a2a_app(
        card=card,
        executor=SpecialistExecutor(card.name, search, progress_updates=True),
    )


def _resolve_search_budget(task: dict[str, object], configured: int) -> int:
    requested = task.get("max_output_tokens", configured)
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        raise ConfigError("invalid search output budget") from None
    if requested < 1024:
        raise ConfigError("search output budget below minimum")
    return min(requested, configured)


def document_generation_app(
    *,
    root: Path | None = None,
    evidence_retriever=None,
):
    card = build_agent_card(
        name="Document Generation Agent",
        description="Drafts and exports report sections.",
        url="http://127.0.0.1:8002/",
    )
    return build_a2a_app(
        card=card,
        executor=DocumentTaskExecutor(root=root, evidence_retriever=evidence_retriever),
    )
