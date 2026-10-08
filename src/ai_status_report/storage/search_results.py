"""Durable storage for search responses and source metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ai_status_report.schemas.search import EvidenceChunk, ResearchReport


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:80] or "query"


def canonicalize_run_id(run_id: str) -> str:
    """Return the filesystem-safe run identifier used by every data layer."""

    safe_run_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", run_id).strip("-")
    if not safe_run_id or safe_run_id in {".", ".."}:
        raise ValueError("invalid_run_id")
    if len(safe_run_id) > 128:
        raise ValueError("invalid_run_id")
    return safe_run_id


_RUN_LABEL_DIRECTORY = ".run_labels"
_WINDOWS_FORBIDDEN_PATH_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _runs_root(root: Path) -> Path:
    return root / "data" / "runs"


def _run_label_record_path(root: Path, canonical_run_id: str) -> Path:
    return _runs_root(root) / _RUN_LABEL_DIRECTORY / f"{canonical_run_id}.json"


def _topic_directory_label(topic: str) -> str:
    """Return a short display-only topic label safe for Windows and POSIX paths."""

    normalized = unicodedata.normalize("NFKC", topic)
    normalized = _WINDOWS_FORBIDDEN_PATH_CHARS.sub(" ", normalized)
    normalized = " ".join(normalized.split()).strip(". ")
    return normalized[:48].rstrip(". ") or "报告"


def _read_run_label_record(path: Path, canonical_run_id: str) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        directory_name = payload["directory_name"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("run_directory_mapping_invalid") from exc
    if (
        payload.get("run_id") != canonical_run_id
        or not isinstance(directory_name, str)
        or Path(directory_name).name != directory_name
        or not directory_name.startswith(f"{canonical_run_id}__")
        or directory_name in {"", ".", "..", _RUN_LABEL_DIRECTORY}
    ):
        raise ValueError("run_directory_mapping_invalid")
    return directory_name


def run_directory(root: Path, run_id: str) -> Path:
    """Resolve a run's physical directory while preserving its stable run ID.

    Legacy runs have no label record and retain ``data/runs/<run_id>``. New
    teaching runs can register a display label before any worker writes files.
    """

    canonical_run_id = canonicalize_run_id(run_id)
    record = _run_label_record_path(root, canonical_run_id)
    if not record.is_file():
        return _runs_root(root) / canonical_run_id
    return _runs_root(root) / _read_run_label_record(record, canonical_run_id)


def register_run_directory(root: Path, run_id: str, topic: str) -> Path:
    """Register a display label for a new run without changing ``run_id``."""

    canonical_run_id = canonicalize_run_id(run_id)
    legacy_directory = _runs_root(root) / canonical_run_id
    record = _run_label_record_path(root, canonical_run_id)
    if record.is_file():
        return run_directory(root, canonical_run_id)
    if legacy_directory.exists():
        return legacy_directory

    directory_name = f"{canonical_run_id}__{_topic_directory_label(topic)}"
    record.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": canonical_run_id,
        "directory_name": directory_name,
        "topic": topic[:200],
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{canonical_run_id}-", suffix=".tmp", dir=record.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, record)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _runs_root(root) / directory_name


def run_id_for_directory(root: Path, directory_name: str) -> str:
    """Return the stable run ID represented by a direct child of ``data/runs``."""

    if Path(directory_name).name != directory_name:
        raise ValueError("invalid_run_directory")
    registry = _runs_root(root) / _RUN_LABEL_DIRECTORY
    if registry.is_dir():
        for record in registry.glob("*.json"):
            canonical_run_id = record.stem
            if _read_run_label_record(record, canonical_run_id) == directory_name:
                return canonical_run_id
    return canonicalize_run_id(directory_name)


def _run_data_root(root: Path, run_id: str | None) -> Path:
    """Return an isolated data root for one run, or the legacy root for tests."""

    if not run_id:
        return root / "data"
    return run_directory(root, run_id)


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return ""
    query = [(key, value) for key, value in parse_qsl(parts.query) if not key.lower().startswith("utm_")]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


EVIDENCE_CHUNK_MAX_CHARS = 7000


def _split_evidence_text(text: str, *, max_chars: int = EVIDENCE_CHUNK_MAX_CHARS) -> list[str]:
    """Split extracted text into bounded, deterministic, paragraph-aware chunks."""

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                paragraph[index : index + max_chars]
                for index in range(0, len(paragraph), max_chars)
            )
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def persist_evidence_chunks(
    root: Path,
    sources: list[dict[str, Any]],
    *,
    raw_ref: str,
    content_by_source_id: dict[str, str] | None = None,
    run_id: str | None = None,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    content_by_source_id = content_by_source_id or {}
    for source_position, source in enumerate(sources):
        source_id = str(source.get("source_id") or f"source-{source_position}")
        text = str(content_by_source_id.get(source_id) or source.get("snippet") or "").strip()
        url = canonicalize_url(str(source.get("url") or ""))
        verification_status = source.get("verification_status")
        if not text or not url or verification_status not in {"retrieved", "verified"}:
            continue
        for chunk_position, chunk_text in enumerate(_split_evidence_text(text)):
            digest = hashlib.sha256(
                f"{url}\n{source_id}\n{chunk_position}\n{chunk_text}".encode()
            ).hexdigest()
            chunk = EvidenceChunk(
                chunk_id=f"chunk-{digest[:16]}",
                source_id=source_id,
                text=chunk_text,
                title=str(source.get("title") or ""),
                url=url,
                provider=str(source.get("provider") or "unknown"),
                position=chunk_position,
                content_hash=digest,
                raw_ref=raw_ref,
                content_ref=str(source.get("content_ref") or ""),
                created_at=datetime.now(UTC),
                verification_status=verification_status,
            )
            destination = _run_data_root(root, run_id) / "evidence" / f"{chunk.chunk_id}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(chunk.model_dump_json(indent=2) + "\n", encoding="utf-8")
            chunks.append(chunk)
    return chunks


def persist_web_content(
    root: Path,
    *,
    source: dict[str, Any],
    content: str,
    extract_ref: str,
    extract_depth: str,
    run_id: str | None = None,
) -> Path:
    """Persist Tavily Extract content and metadata for later parsing."""

    url = canonicalize_url(str(source.get("url") or ""))
    if not url or not content.strip():
        raise ValueError("invalid_extracted_content")
    digest = hashlib.sha256(f"{url}\n{content}".encode()).hexdigest()[:16]
    destination = _run_data_root(root, run_id) / "raw" / "pages" / f"page-{digest}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    metadata = destination.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "source_id": source.get("source_id", ""),
                "source_url": url,
                "title": source.get("title", ""),
                "provider": "tavily",
                "tool": "tavily_extract",
                "extract_depth": extract_depth,
                "content_format": "markdown",
                "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "extract_ref": extract_ref,
                "content_ref": str(destination),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def persist_search_result(
    root: Path,
    *,
    query: str,
    provider: str,
    result: Any,
    run_id: str | None = None,
) -> Path:
    """Write one immutable, JSON-serializable search record under data/sources."""
    fetched_at = datetime.now(UTC).isoformat()
    digest = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()[:16]
    record = {
        "schema_version": "1",
        "source_id": f"{provider}-{digest}",
        "provider": provider,
        "query": query,
        "fetched_at": fetched_at,
        "result": result,
    }
    destination = _run_data_root(root, run_id) / "sources" / f"{_safe_name(query)}-{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def persist_raw_search_response(
    root: Path,
    *,
    query: str,
    provider: str,
    result: Any,
    run_id: str | None = None,
) -> Path:
    """Persist provider output in the raw/debug layer."""
    fetched_at = datetime.now(UTC).isoformat()
    digest = hashlib.sha256(
        json.dumps(result, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()[:16]
    record = {
        "schema_version": "1",
        "source_id": f"{provider}-{digest}",
        "provider": provider,
        "query": query,
        "fetched_at": fetched_at,
        "result": result,
    }
    destination = _run_data_root(root, run_id) / "raw" / "search" / f"{_safe_name(query)}-{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def persist_research_report(root: Path, report: ResearchReport, *, run_id: str | None = None) -> Path:
    """Persist the compact business handoff separately from raw provider output."""
    payload = report.model_dump(mode="json")
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]
    destination = _run_data_root(root, run_id) / "reports" / f"{_safe_name(report.query)}-{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination
