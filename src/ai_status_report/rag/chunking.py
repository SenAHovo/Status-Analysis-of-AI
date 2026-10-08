"""Create bounded embedding units from durable EvidenceChunk records."""

from __future__ import annotations

import re

from ai_status_report.rag.index_versions import DEFAULT_INDEX_VERSION, is_valid_index_version
from ai_status_report.rag.schemas import RetrievalChunk
from ai_status_report.schemas.search import EvidenceChunk
from ai_status_report.storage.search_results import canonicalize_run_id

RETRIEVAL_CHUNK_MAX_CHARS = 900


def _split_text(text: str, max_chars: int) -> list[tuple[str, int, int]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[tuple[str, int, int]] = []
    current = ""
    current_start = 0
    cursor = 0
    for paragraph in paragraphs:
        paragraph_start = normalized.find(paragraph, cursor)
        paragraph_end = paragraph_start + len(paragraph)
        cursor = paragraph_end
        if len(paragraph) > max_chars:
            if current:
                chunks.append((current, current_start, paragraph_start))
                current = ""
            for offset in range(0, len(paragraph), max_chars):
                part = paragraph[offset : offset + max_chars]
                chunks.append((part, paragraph_start + offset, paragraph_start + offset + len(part)))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if not current:
            current_start = paragraph_start
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append((current, current_start, paragraph_start))
            current = paragraph
            current_start = paragraph_start
    if current:
        chunks.append((current, current_start, len(normalized)))
    return chunks


def to_retrieval_chunks(
    evidence: EvidenceChunk,
    *,
    run_id: str,
    max_chars: int = RETRIEVAL_CHUNK_MAX_CHARS,
    index_version: str = DEFAULT_INDEX_VERSION,
) -> list[RetrievalChunk]:
    """Split one EvidenceChunk while retaining parent and source references."""

    canonical_run_id = canonicalize_run_id(run_id)
    if not 100 <= max_chars <= 1000:
        raise ValueError("invalid_retrieval_chunk_size")
    if not is_valid_index_version(index_version):
        raise ValueError("invalid_index_version")
    result = []
    for position, (text, start, end) in enumerate(_split_text(evidence.text, max_chars)):
        result.append(
            RetrievalChunk(
                retrieval_chunk_id=f"{canonical_run_id}:{evidence.chunk_id}:{position}",
                evidence_chunk_id=evidence.chunk_id,
                run_id=canonical_run_id,
                source_id=evidence.source_id,
                text=text,
                position=position,
                start_offset=start,
                end_offset=end,
                title=evidence.title,
                url=evidence.url,
                provider=evidence.provider,
                raw_ref=evidence.raw_ref,
                content_ref=evidence.content_ref,
                verification_status=evidence.verification_status,
                index_version=index_version,
            )
        )
    return result
