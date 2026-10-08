"""Tavily MCP extraction and source selection helpers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from ai_status_report.schemas.search import ResearchSource

PAPER_BOOST = 0.28
_PAPER_DOMAINS = {
    "arxiv.org",
    "openreview.net",
    "acm.org",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "doi.org",
    "semanticscholar.org",
    "nature.com",
    "science.org",
    "springer.com",
    "sciencedirect.com",
}
_PAPER_TERMS = ("paper", "research", "journal", "论文", "研究", "期刊", "学术")


def _is_paper_source(source: ResearchSource) -> bool:
    host = urlsplit(source.url).netloc.lower().removeprefix("www.")
    domain_match = host in _PAPER_DOMAINS or any(host.endswith(f".{domain}") for domain in _PAPER_DOMAINS)
    text = f"{source.title} {source.url}".lower()
    return domain_match or any(term in text for term in _PAPER_TERMS)


def select_extract_sources(sources: list[ResearchSource], limit: int = 5) -> list[ResearchSource]:
    """Rank candidates with a small paper-source boost before extraction."""

    candidates = [source for source in sources if source.url]
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (PAPER_BOOST if _is_paper_source(item[1]) else 0.0, -item[0]),
        reverse=True,
    )
    return [source for _, source in ranked[:limit]]


def _json_objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for key in ("results", "data", "content"):
            nested = value.get(key)
            if isinstance(nested, list):
                yield from _json_objects(nested)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                try:
                    decoded = json.loads(item)
                except ValueError:
                    continue
                yield from _json_objects(decoded)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    try:
                        decoded = json.loads(text)
                    except ValueError:
                        continue
                    yield from _json_objects(decoded)


def normalize_extract_response(response: Any) -> list[dict[str, str]]:
    """Normalize structured or text MCP content from tavily_extract."""

    normalized: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in _json_objects(response):
        url = item.get("url") or item.get("source_url")
        content = item.get("raw_content") or item.get("content") or item.get("text")
        if not isinstance(url, str) or not isinstance(content, str) or not content.strip():
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        normalized.append(
            {
                "url": url,
                "raw_content": content,
                "title": str(item.get("title") or ""),
                "favicon": str(item.get("favicon") or ""),
            }
        )
    return normalized
