"""Deterministic assembly of approved chapter Markdown artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ai_status_report.documents.markdown import (
    MarkdownProtocolError,
    validate_persisted_chapter_artifact,
)
from ai_status_report.schemas.report import ArtifactRef
from ai_status_report.schemas.report_spec import ReportSpec, ReportTitle, default_report_spec
from ai_status_report.storage.search_results import canonicalize_run_id, run_directory

_CITATION = re.compile(r"\[(\d+)\]")


def assemble_report(
    root: Path,
    *,
    run_id: str,
    chapter_artifacts: dict[str, ArtifactRef],
    spec: ReportSpec | None = None,
    report_title: str | None = None,
) -> ArtifactRef:
    """Combine exactly the four fixed chapters and deduplicate references by URL."""

    safe_run_id = canonicalize_run_id(run_id)
    report_spec = spec or default_report_spec()
    if set(chapter_artifacts) != {item.section_id for item in report_spec.sections}:
        raise ValueError("report assembly requires exactly one artifact for every report section")
    blocks: list[str] = []
    references: list[tuple[str, str]] = []
    reference_numbers: dict[str, int] = {}
    for section in report_spec.sections:
        artifact = chapter_artifacts[section.section_id]
        if artifact.run_id != safe_run_id:
            raise ValueError("report_chapter_artifact_unavailable")
        try:
            path, _ = validate_persisted_chapter_artifact(
                root,
                artifact=artifact,
                section_id=section.section_id,
            )
        except MarkdownProtocolError as exc:
            raise ValueError("report_chapter_artifact_integrity_failed") from exc
        text = path.read_text(encoding="utf-8")
        parts = text.split("## 参考来源", 1)
        body = parts[0].rstrip()
        chapter_refs = parts[1] if len(parts) == 2 else ""
        local_map: dict[int, int] = {}
        for match in re.finditer(r"^\[(\d+)\]\s+(.+?)(?:。)?(?:\n|$)", chapter_refs, re.MULTILINE):
            label = match.group(2).strip()
            url_match = re.search(r"https?://\S+", label)
            key = url_match.group(0).rstrip("。") if url_match else label
            if key not in reference_numbers:
                reference_numbers[key] = len(references) + 1
                references.append((label.replace(key, "").rstrip("。 "), key if url_match else ""))
            local_map[int(match.group(1))] = reference_numbers[key]
        def replace_citation(match: re.Match[str], mapping: dict[int, int] = local_map) -> str:
            number = int(match.group(1))
            return f"[{mapping.get(number, number)}]"

        body = _CITATION.sub(replace_citation, body)
        blocks.append(body)
    title = ReportTitle(title=report_title or "人工智能现状分析报告").title
    lines = [f"# {title}", ""]
    for block in blocks:
        lines.extend([block, ""])
    lines.extend(["## 统一参考来源", ""])
    lines.extend(f"[{index}] {label}" + (f"。{url}" if url else "。") for index, (label, url) in enumerate(references, 1))
    content = "\n".join(lines).rstrip() + "\n"
    content_bytes = content.encode("utf-8")
    digest = hashlib.sha256(content_bytes).hexdigest()
    path = run_directory(root, safe_run_id) / "reports" / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content_bytes:
        raise ValueError("report_artifact_conflict")
    if not path.exists():
        path.write_bytes(content_bytes)
    manifest = path.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "run_id": safe_run_id,
                "sections": [section.section_id for section in report_spec.sections],
                "title": title,
                "content_hash": digest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return ArtifactRef(
        artifact_id=f"artifact-report-{digest[:32]}",
        run_id=safe_run_id,
        version="1",
        kind="report_markdown",
        mime_type="text/markdown",
        size=path.stat().st_size,
        hash=digest,
        access_ref=str(path),
        input_versions={
            section.section_id: chapter_artifacts[section.section_id].version
            for section in report_spec.sections
        },
    )
