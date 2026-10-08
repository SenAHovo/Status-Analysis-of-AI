import hashlib

import pytest

from ai_status_report.documents.pdf_export import (
    _css,
    _html,
    _is_allowed_resource,
    export_report_pdf,
)
from ai_status_report.schemas.report import ArtifactRef


def _report_artifact(tmp_path) -> ArtifactRef:
    report = tmp_path / "data" / "runs" / "run-pdf-001" / "reports" / "report.md"
    report.parent.mkdir(parents=True)
    content = b"# Report\n\nBody.\n"
    report.write_bytes(content)
    return ArtifactRef(
        artifact_id="artifact-report-pdf-test",
        run_id="run-pdf-001",
        version="1",
        kind="report_markdown",
        mime_type="text/markdown",
        size=len(content),
        hash=hashlib.sha256(content).hexdigest(),
        access_ref=str(report),
    )


def test_pdf_export_rejects_an_output_outside_the_report_directory(tmp_path):
    with pytest.raises(ValueError, match="report_pdf_output_path_invalid"):
        export_report_pdf(
            tmp_path,
            report_artifact=_report_artifact(tmp_path),
            output_pdf=tmp_path / "outside.pdf",
        )


def test_pdf_export_rejects_invalid_utf8_before_starting_browser(tmp_path):
    artifact = _report_artifact(tmp_path)
    report = tmp_path / "data" / "runs" / "run-pdf-001" / "reports" / "report.md"
    invalid = b"# Report\n\n\xff\xfe"
    report.write_bytes(invalid)
    artifact = artifact.model_copy(
        update={"size": len(invalid), "hash": hashlib.sha256(invalid).hexdigest()}
    )

    with pytest.raises(ValueError, match="report_markdown_invalid_encoding"):
        export_report_pdf(tmp_path, report_artifact=artifact)

    assert not report.with_suffix(".pdf").exists()


def test_pdf_resource_policy_blocks_network_and_outside_files(tmp_path):
    report_directory = tmp_path / "data" / "runs" / "run-pdf-001" / "reports"
    assert _is_allowed_resource("data:text/plain,report", report_directory=report_directory)
    assert not _is_allowed_resource("https://example.com/image.png", report_directory=report_directory)
    assert not _is_allowed_resource((tmp_path / "outside.png").as_uri(), report_directory=report_directory)
    assert _is_allowed_resource((report_directory / "inside.png").as_uri(), report_directory=report_directory)


def test_pdf_css_prefers_times_for_latin_and_songti_for_cjk():
    assert "font-family: 'Times New Roman', 'SimSun', '宋体'" in _css()


def test_pdf_css_keeps_long_cover_title_and_toc_together_on_first_page():
    css = _css()

    assert "body > h1:first-child" in css
    assert "font-size: 20pt" in css
    assert "page-break-inside: avoid" in css
    assert "page-break-after: avoid" in css
    assert ".toc { margin: 0 auto 18pt; max-width: 92%; font-size: 9.5pt" in css


def test_pdf_html_adds_title_page_toc_and_reference_line_breaks(tmp_path):
    rendered = _html(
        "# 报告标题\n\n# 背景\n\n## 本章摘要\n\n**一、技术形态：从对话能力到可执行任务**\n\n**二、工具调用：从接口连接到任务执行**\n\n正文。\n\n## 正文\n\n# 现状\n\n正文。\n\n## 统一参考来源\n\n[1] 来源一。https://example.com\n[2] 来源二。https://example.org\n",
        tmp_path,
    )

    assert "<h1 id='report-title'>报告标题</h1>" in rendered
    assert "<nav class='toc'" in rendered
    assert "href='#toc-1'>一、背景</a>" in rendered
    assert "href='#toc-6'>二、现状</a>" in rendered
    assert "href='#toc-2'>1、本章摘要</a>" in rendered
    assert "href='#toc-5'>2、正文</a>" in rendered
    assert "href='#toc-3'>（1）技术形态：从对话能力到可执行任务</a>" in rendered
    assert "href='#toc-4'>（2）工具调用：从接口连接到任务执行</a>" in rendered
    assert "<ol class='toc-level-1'>" in rendered
    assert "<li><a href='#toc-1'>一、背景</a>" in rendered
    assert "<h1 id='toc-1'>一、背景</h1>" in rendered
    assert "<h2 id='toc-2'>1、本章摘要</h2>" in rendered
    assert "<h3 id='toc-3'>（1）技术形态：从对话能力到可执行任务</h3>" in rendered
    assert "<h3 id='toc-4'>（2）工具调用：从接口连接到任务执行</h3>" in rendered
    assert "<div class='references'>" in rendered
    assert "<p>[1] 来源一。https://example.com</p>" in rendered
    assert "<p>[2] 来源二。https://example.org</p>" in rendered
