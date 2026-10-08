"""Load run-scoped EvidenceChunk files for indexing."""

from __future__ import annotations

import json
from pathlib import Path

from ai_status_report.schemas.search import EvidenceChunk
from ai_status_report.storage.search_results import canonicalize_run_id, run_directory


def load_run_evidence(root: Path, run_id: str) -> list[EvidenceChunk]:
    """Load evidence and recover page references from the same run when needed."""

    canonical_run_id = canonicalize_run_id(run_id)
    run_root = run_directory(root, canonical_run_id)
    page_refs: dict[str, str] = {}
    for metadata_path in (run_root / "raw" / "pages").glob("*.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_id = str(metadata.get("source_id") or "")
        content_ref = str(metadata.get("content_ref") or "")
        if source_id and content_ref:
            page_refs[source_id] = content_ref
    result = []
    for evidence_path in sorted((run_root / "evidence").glob("*.json")):
        evidence = EvidenceChunk.model_validate_json(evidence_path.read_text(encoding="utf-8"))
        if not evidence.content_ref and evidence.source_id in page_refs:
            evidence = evidence.model_copy(update={"content_ref": page_refs[evidence.source_id]})
        result.append(evidence)
    return result


def load_parent_evidence(root: Path, *, run_id: str, evidence_chunk_id: str) -> EvidenceChunk:
    """Load one parent EvidenceChunk within the requested run boundary."""

    canonical_run_id = canonicalize_run_id(run_id)
    if not evidence_chunk_id or Path(evidence_chunk_id).name != evidence_chunk_id:
        raise ValueError("invalid_evidence_chunk_id")
    evidence_path = (
        root
        / "data"
        / "runs"
        / canonical_run_id
        / "evidence"
        / f"{evidence_chunk_id}.json"
    )
    if not evidence_path.is_file():
        raise FileNotFoundError("evidence_chunk_not_found")
    evidence = EvidenceChunk.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    if evidence.chunk_id != evidence_chunk_id:
        raise ValueError("evidence_chunk_id_mismatch")
    return evidence
