"""Run-scoped, durable records for controller review and evidence supplements."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ai_status_report.schemas.document import DocumentSectionResult
from ai_status_report.schemas.review import ReviewDecision, SupplementPlan
from ai_status_report.storage.search_results import run_directory

_SAFE_SECTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _section_directory(section_id: str) -> str:
    """Return the reviewed section's safe audit directory name."""

    if not _SAFE_SECTION_ID.fullmatch(section_id) or section_id in {".", ".."}:
        raise ValueError("review_audit_section_invalid")
    return section_id


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("review_audit_conflict")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


def persist_review_audit(
    root: Path,
    *,
    result: DocumentSectionResult,
    decision: ReviewDecision,
    plan: SupplementPlan | None,
) -> Path:
    """Persist the review outcome before LangGraph routes to another node."""

    path = (
        run_directory(root, result.run_id)
        / "reviews"
        / _section_directory(result.section_id)
        / f"review-v{result.draft_version}-r{decision.review_round}.json"
    )
    revision_effect = None
    if int(result.draft_version) > 1:
        chapter_dir = run_directory(root, result.run_id) / "chapters" / result.section_id
        previous = chapter_dir / f"v{int(result.draft_version) - 1}.md"
        current = chapter_dir / f"v{result.draft_version}.md"
        if previous.is_file() and current.is_file():
            old_text = previous.read_text(encoding="utf-8")
            new_text = current.read_text(encoding="utf-8")
            revision_effect = {
                "base_version": str(int(result.draft_version) - 1),
                "target_version": result.draft_version,
                "hash_changed": old_text != new_text,
                "character_delta": len(new_text) - len(old_text),
                "citation_count_before": len(re.findall(r"\[\d{1,3}\]", old_text)),
                "citation_count_after": len(re.findall(r"\[\d{1,3}\]", new_text)),
                "limitation_text_removed": any(
                    marker in old_text and marker not in new_text
                    for marker in ("需要说明", "尚缺少", "缺少独立交叉验证", "证据不足")
                ),
            }
    return _write_json(
        path,
        {
            "schema_version": "1",
            "section_id": result.section_id,
            "draft_version": result.draft_version,
            "decision": decision.model_dump(mode="json"),
            "document_result_ref": {
                "draft_artifact": result.draft_artifact.model_dump(mode="json")
                if result.draft_artifact
                else None,
                "citation_map_ref": result.citation_map_ref,
                "evidence_bundle_ref": result.evidence_bundle_ref,
            },
            "supplement_plan": plan.model_dump(mode="json") if plan else None,
            "revision_effect": revision_effect,
        },
    )


def persist_supplement_audit(
    root: Path,
    *,
    plan: SupplementPlan,
    report_refs: list[str],
    new_source_ids: list[str],
    new_chunk_ids: list[str],
) -> Path:
    """Persist a compact, replayable record of the completed supplement round."""

    path = (
        run_directory(root, plan.run_id)
        / "supplements"
        / _section_directory(plan.section_id)
        / f"round-{plan.supplement_round}.json"
    )
    return _write_json(
        path,
        {
            "schema_version": "1",
            "plan": plan.model_dump(mode="json"),
            "report_refs": report_refs,
            "evidence_delta": {
                "new_source_ids": new_source_ids,
                "new_chunk_ids": new_chunk_ids,
            },
        },
    )
