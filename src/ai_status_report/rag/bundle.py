"""Build and persist section-scoped evidence bundles."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from ai_status_report.rag.index_versions import DEFAULT_INDEX_VERSION, is_valid_index_version
from ai_status_report.rag.ingest import load_parent_evidence
from ai_status_report.rag.schemas import RetrievalMatch
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt
from ai_status_report.schemas.search import EvidenceChunk
from ai_status_report.storage.search_results import (
    canonicalize_run_id,
    canonicalize_url,
    run_directory,
)
from ai_status_report.token_budget.estimator import estimate_tokens


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")[:128] or "retrieval"


_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_unicode_escapes(text: str) -> str:
    """Decode literal JSON Unicode escapes without changing ordinary backslashes."""

    return _UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)


def _source_key(match: RetrievalMatch) -> str:
    """Group result chunks by durable source before filling the chapter context."""

    return canonicalize_url(match.url) or f"source_id:{match.source_id}"


def _round_robin_sources(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    """Give each available source an early evidence slot without imposing a cap.

    Chroma still determines relevance within every source. Once every available
    source has contributed its next best result, the next round continues until
    candidates or the chapter evidence budget are exhausted.
    """

    groups: dict[str, list[RetrievalMatch]] = {}
    for match in matches:
        groups.setdefault(_source_key(match), []).append(match)
    ordered: list[RetrievalMatch] = []
    position = 0
    while True:
        round_items = [group[position] for group in groups.values() if position < len(group)]
        if not round_items:
            return ordered
        ordered.extend(round_items)
        position += 1


def build_evidence_bundle(
    *,
    run_id: str,
    section_id: str,
    queries: list[str],
    matches: Iterable[RetrievalMatch],
    budget_tokens: int = 1200,
    index_version: str = DEFAULT_INDEX_VERSION,
    root: Path | None = None,
) -> EvidenceBundle:
    """Select complete, provenance-preserving excerpts for one section.

    Reranked results take precedence when available; vector distance supplies
    a stable order within the same score system. Exact retrieval IDs and
    identical text are removed; distinct chunks from the same source remain
    available because they can contain different evidence. Whole excerpts are
    retained or excluded so citations never point into an unpersisted slice.
    """

    canonical_run_id = canonicalize_run_id(run_id)
    if not section_id.strip() or len(section_id) > 64:
        raise ValueError("invalid_section_id")
    if budget_tokens < 0:
        raise ValueError("invalid_budget_tokens")
    if not is_valid_index_version(index_version):
        raise ValueError("invalid_index_version")
    clean_queries = [query.strip() for query in queries if query.strip()]
    if not clean_queries:
        raise ValueError("queries_required")

    selected: list[Excerpt] = []
    chunk_refs: list[str] = []
    source_metadata: dict[str, str] = {}
    quality_flags: list[str] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    parent_cache: dict[str, EvidenceChunk | None] = {}
    budget_used = 0
    def ranking_key(item: RetrievalMatch) -> tuple[int, float, float, str]:
        """Prefer optional model rerank order without mixing score systems."""

        if item.rerank_score is not None:
            return (0, -item.rerank_score, item.distance, item.retrieval_chunk_id)
        return (1, 0.0, item.distance, item.retrieval_chunk_id)

    ranked = sorted(matches, key=ranking_key)
    unique_matches: list[RetrievalMatch] = []
    for match in ranked:
        retrieval_text = _decode_unicode_escapes(match.text)
        normalized_text = " ".join(retrieval_text.split())
        if match.retrieval_chunk_id in seen_ids or normalized_text in seen_text:
            if "deduplicated" not in quality_flags:
                quality_flags.append("deduplicated")
            continue
        seen_ids.add(match.retrieval_chunk_id)
        seen_text.add(normalized_text)
        unique_matches.append(match)

    for match in _round_robin_sources(unique_matches):
        retrieval_text = _decode_unicode_escapes(match.text)
        parent = None
        if root is not None:
            if match.evidence_chunk_id not in parent_cache:
                try:
                    parent_cache[match.evidence_chunk_id] = load_parent_evidence(
                        root, run_id=canonical_run_id, evidence_chunk_id=match.evidence_chunk_id
                    )
                except (FileNotFoundError, ValueError):
                    parent_cache[match.evidence_chunk_id] = None
            parent = parent_cache[match.evidence_chunk_id]
            if parent is None and "parent_missing" not in quality_flags:
                quality_flags.append("parent_missing")
            elif parent is not None and parent.source_id != match.source_id:
                parent = None
                if "parent_source_mismatch" not in quality_flags:
                    quality_flags.append("parent_source_mismatch")

        excerpt_text = _decode_unicode_escapes(parent.text) if parent is not None else retrieval_text
        estimate = estimate_tokens(excerpt_text)
        if budget_used + estimate > budget_tokens:
            if parent is not None:
                excerpt_text = retrieval_text
                estimate = estimate_tokens(excerpt_text)
                if budget_used + estimate <= budget_tokens:
                    if "parent_budget_fallback" not in quality_flags:
                        quality_flags.append("parent_budget_fallback")
                else:
                    if "budget_clipped" not in quality_flags:
                        quality_flags.append("budget_clipped")
                    continue
            else:
                if "budget_clipped" not in quality_flags:
                    quality_flags.append("budget_clipped")
                continue
        selected.append(
            Excerpt(
                chunk_id=match.retrieval_chunk_id,
                source_id=match.source_id,
                text=excerpt_text,
                page_or_offset=f"{match.start_offset}:{match.end_offset}",
                token_estimate=estimate,
                title=match.title,
                url=match.url,
                provider=match.provider,
                raw_ref=match.raw_ref,
                content_ref=match.content_ref,
                verification_status=match.verification_status,
                distance=match.distance,
                retrieval_chunk_id=match.retrieval_chunk_id,
                parent_chunk_id=match.evidence_chunk_id,
            )
        )
        chunk_refs.append(match.retrieval_chunk_id)
        if match.source_id and match.url:
            source_metadata[match.source_id] = match.url
        budget_used += estimate

    if not selected:
        quality_flags.append("no_excerpts")
    return EvidenceBundle(
        retrieval_id=EvidenceBundle.new_retrieval_id(canonical_run_id, section_id),
        run_id=canonical_run_id,
        section_id=section_id,
        index_version=index_version,
        queries=clean_queries,
        chunk_refs=chunk_refs,
        excerpts=selected,
        source_metadata=source_metadata,
        quality_flags=quality_flags,
        budget_used=budget_used,
    )


def persist_evidence_bundle(root: Path, bundle: EvidenceBundle) -> Path:
    """Persist the exact section input under its isolated run directory."""

    run_id = canonicalize_run_id(bundle.run_id)
    destination = run_directory(root, run_id) / "evidence_bundles" / (
        f"{_safe_filename(bundle.retrieval_id)}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(bundle.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
