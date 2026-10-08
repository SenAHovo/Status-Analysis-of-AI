import hashlib

import pytest

from ai_status_report.documents.markdown import persist_section_markdown
from ai_status_report.documents.report_assembly import assemble_report


def test_assembly_orders_four_chapters_and_deduplicates_references(tmp_path):
    artifacts = {}
    for section_id, title in (("background", "背景"), ("current-status", "现状"), ("trends", "趋势"), ("recommendations", "建议")):
        artifacts[section_id] = persist_section_markdown(
            tmp_path, run_id="run-assembly", section_id=section_id, version="1",
            markdown=f"# {title}\n\n正文 [{1}]。\n\n## 参考来源\n\n[1] 统一来源。https://example.com/source\n",
            input_versions={},
        )
    artifact = assemble_report(tmp_path, run_id="run-assembly", chapter_artifacts=artifacts)
    content = (tmp_path / "data" / "runs" / "run-assembly" / "reports" / "report.md").read_text(encoding="utf-8")
    assert artifact.kind == "report_markdown"
    assert artifact.hash == hashlib.sha256((tmp_path / "data" / "runs" / "run-assembly" / "reports" / "report.md").read_bytes()).hexdigest()
    assert [content.index(f"# {title}") for title in ("背景", "现状", "趋势", "建议")] == sorted(content.index(f"# {title}") for title in ("背景", "现状", "趋势", "建议"))
    assert content.count("https://example.com/source") == 1
    assert content.count("[1]") == 5


def test_assembly_uses_validated_report_title(tmp_path):
    artifacts = {}
    for section_id, title in (("background", "背景"), ("current-status", "现状"), ("trends", "趋势"), ("recommendations", "建议")):
        artifacts[section_id] = persist_section_markdown(
            tmp_path, run_id="run-title", section_id=section_id, version="1",
            markdown=f"# {title}\n\n正文。\n\n## 参考来源\n\n本章未使用正文引用。\n",
            input_versions={},
        )
    assemble_report(
        tmp_path,
        run_id="run-title",
        chapter_artifacts=artifacts,
        report_title="智能体交易安全：现状、趋势与建议",
    )
    content = (tmp_path / "data" / "runs" / "run-title" / "reports" / "report.md").read_text(encoding="utf-8")
    assert content.startswith("# 智能体交易安全：现状、趋势与建议\n")


def test_assembly_rejects_a_forged_chapter_artifact(tmp_path):
    artifacts = {}
    for section_id in ("background", "current-status", "trends", "recommendations"):
        artifacts[section_id] = persist_section_markdown(
            tmp_path,
            run_id="run-assembly-integrity",
            section_id=section_id,
            version="1",
            markdown=(
                f"# {section_id}\n\n正文。\n\n## 参考来源\n\n"
                "本章未使用正文引用。\n"
            ),
            input_versions={},
        )
    artifacts["trends"] = artifacts["trends"].model_copy(update={"hash": "0" * 64})

    with pytest.raises(ValueError, match="report_chapter_artifact_integrity_failed"):
        assemble_report(
            tmp_path,
            run_id="run-assembly-integrity",
            chapter_artifacts=artifacts,
        )


def test_assembly_rejects_an_existing_report_with_different_line_endings(tmp_path):
    artifacts = {}
    for section_id in ("background", "current-status", "trends", "recommendations"):
        artifacts[section_id] = persist_section_markdown(
            tmp_path,
            run_id="run-assembly-line-endings",
            section_id=section_id,
            version="1",
            markdown=f"# {section_id}\n\n正文。\n\n## 参考来源\n\n本章未使用正文引用。\n",
            input_versions={},
        )
    assemble_report(
        tmp_path,
        run_id="run-assembly-line-endings",
        chapter_artifacts=artifacts,
    )
    report_path = tmp_path / "data" / "runs" / "run-assembly-line-endings" / "reports" / "report.md"
    report_path.write_bytes(report_path.read_bytes().replace(b"\n", b"\r\n"))

    with pytest.raises(ValueError, match="report_artifact_conflict"):
        assemble_report(
            tmp_path,
            run_id="run-assembly-line-endings",
            chapter_artifacts=artifacts,
        )
