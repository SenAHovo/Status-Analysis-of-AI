from pathlib import Path

import pytest

from ai_status_report.documents.markdown import (
    MarkdownProtocolError,
    extract_reader_section_summary,
    normalize_reader_citations,
    parse_section_markdown,
    persist_citation_map,
    persist_section_markdown,
    render_reader_markdown,
    validate_markdown_evidence,
    validate_persisted_chapter_artifact,
)
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt
from ai_status_report.storage.search_results import register_run_directory


def test_parse_section_markdown_extracts_protocol_fields_and_citations():
    parsed = parse_section_markdown(
        """# 二、现状

## 本章摘要

本章说明当前部署现状。

## 正文

已有企业将智能体用于流程编排。[1]
同一证据可在本章不同段落引用。[1]

## 证据缺口

- 缺少跨行业成本数据。
"""
    )

    assert parsed.title == "二、现状"
    assert parsed.summary == "本章说明当前部署现状。"
    assert parsed.citation_numbers == (1,)
    assert parsed.evidence_gaps[0].description == "缺少跨行业成本数据。"
    assert parsed.evidence_gaps[0].severity == "soft"


def test_extract_reader_section_summary_does_not_require_controller_only_gap_section():
    assert extract_reader_section_summary(
        """# 二、现状

## 本章摘要

已批准的章节摘要。

## 正文

章节正文。[1]

## 参考来源

[1] 来源。https://example.com/source
"""
    ) == "已批准的章节摘要。"


def test_validate_and_persist_section_markdown(tmp_path):
    markdown = """# 二、现状

## 本章摘要

摘要。

## 正文

事实。[1]

## 证据缺口

无
"""
    parsed = parse_section_markdown(markdown)
    bundle = EvidenceBundle(
        retrieval_id="retrieval-1",
        run_id="run-1",
        section_id="s1",
        excerpts=[Excerpt(chunk_id="chunk-1", source_id="source-a", text="evidence")],
    )
    citations = validate_markdown_evidence(parsed, bundle)
    reader_markdown = render_reader_markdown(parsed, citations)
    artifact = persist_section_markdown(
        tmp_path,
        run_id="run-1",
        section_id="s1",
        version="1",
        markdown=reader_markdown,
        input_versions={"skill": "abc"},
    )

    assert artifact.kind == "chapter_markdown"
    assert artifact.access_ref.endswith("data\\runs\\run-1\\chapters\\s1\\v1.md")
    assert artifact.size > 0
    citation_map = persist_citation_map(
        tmp_path,
        artifact=artifact,
        citations=citations,
        evidence_gaps=parsed.evidence_gaps,
    )
    assert citation_map.is_file()
    persisted = Path(artifact.access_ref).read_text(encoding="utf-8")
    assert "## 参考来源" in persisted
    assert "[1] 未命名来源。" in persisted
    assert "## 证据缺口" not in persisted


def test_validate_persisted_chapter_uses_topic_labelled_run_directory(tmp_path):
    run_id = "teach-labelled"
    labelled_directory = register_run_directory(tmp_path, run_id, "智能体交易安全")
    artifact = persist_section_markdown(
        tmp_path,
        run_id=run_id,
        section_id="background",
        version="1",
        markdown="# 一、背景\n\n## 本章摘要\n\n摘要。\n\n## 正文\n\n正文。\n",
        input_versions={},
    )

    path, content_hash = validate_persisted_chapter_artifact(
        tmp_path,
        artifact=artifact,
        section_id="background",
    )

    assert path.parent.parent.parent == labelled_directory
    assert content_hash == artifact.hash


def test_same_source_chunks_share_one_reader_citation_and_keep_internal_locators(tmp_path):
    markdown = """# 二、现状

## 本章摘要

摘要。

## 正文

同一报告的不同段落共同支持该结论。[1]

## 证据缺口

- [hard] 缺少量化效果数据。
"""
    parsed = parse_section_markdown(markdown)
    bundle = EvidenceBundle(
        retrieval_id="retrieval-1",
        run_id="run-1",
        section_id="s1",
        excerpts=[
            Excerpt(
                chunk_id="chunk-1",
                source_id="source-a",
                title="同一份报告",
                url="https://example.com/report?utm_source=test",
                text="first evidence",
            ),
            Excerpt(
                chunk_id="chunk-2",
                source_id="source-a",
                title="同一份报告",
                url="https://example.com/report",
                text="second evidence",
            ),
        ],
    )
    citations = validate_markdown_evidence(parsed, bundle)
    assert len(citations) == 1
    assert citations[0].number == 1
    assert citations[0].supporting_chunk_ids == ("chunk-1", "chunk-2")

    artifact = persist_section_markdown(
        tmp_path,
        run_id="run-1",
        section_id="s1",
        version="1",
        markdown=render_reader_markdown(parsed, citations),
        input_versions={},
    )
    citation_map = persist_citation_map(
        tmp_path,
        artifact=artifact,
        citations=citations,
        evidence_gaps=parsed.evidence_gaps,
    )
    reader_markdown = Path(artifact.access_ref).read_text(encoding="utf-8")
    payload = citation_map.read_text(encoding="utf-8")

    assert reader_markdown.count("[1] 同一份报告。https://example.com/report") == 1
    assert "[2]" not in reader_markdown
    assert "证据缺口" not in reader_markdown
    assert '"supporting_chunk_ids": [' in payload
    assert '"chunk-1"' in payload
    assert '"chunk-2"' in payload
    assert "缺少量化效果数据。" in payload


def test_reader_citations_are_compacted_after_model_uses_a_bundle_subset():
    parsed = parse_section_markdown(
        "# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n结论。[2][3]\n\n## 证据缺口\n无"
    )
    bundle = EvidenceBundle(
        retrieval_id="retrieval-1",
        run_id="run-1",
        section_id="s1",
        excerpts=[
            Excerpt(chunk_id="chunk-1", source_id="source-a", text="unused"),
            Excerpt(chunk_id="chunk-2", source_id="source-b", text="used first"),
            Excerpt(chunk_id="chunk-3", source_id="source-c", text="used second"),
        ],
    )
    reader_parsed, reader_citations = normalize_reader_citations(
        parsed, validate_markdown_evidence(parsed, bundle)
    )

    assert reader_parsed.body == "结论。[1][2]"
    assert [binding.number for binding in reader_citations] == [1, 2]
    assert [binding.source_ids for binding in reader_citations] == [("source-b",), ("source-c",)]


def test_validate_markdown_evidence_rejects_unknown_reference():
    parsed = parse_section_markdown(
        "# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n事实。[2]\n\n## 证据缺口\n无"
    )
    bundle = EvidenceBundle(retrieval_id="retrieval-1", run_id="run-1", section_id="s1")
    with pytest.raises(MarkdownProtocolError, match="markdown_unknown_citation_number"):
        validate_markdown_evidence(parsed, bundle)


def test_parse_section_markdown_rejects_legacy_internal_citation():
    with pytest.raises(MarkdownProtocolError, match="markdown_internal_citation_forbidden"):
        parse_section_markdown(
            "# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n事实。[source-a|chunk-b]\n\n## 证据缺口\n无"
        )


def test_persist_section_markdown_rejects_conflicting_same_version(tmp_path):
    original = "# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n正文\n\n## 证据缺口\n无"
    persist_section_markdown(
        tmp_path,
        run_id="run-1",
        section_id="s1",
        version="1",
        markdown=original,
        input_versions={},
    )

    with pytest.raises(MarkdownProtocolError, match="chapter_version_conflict"):
        persist_section_markdown(
            tmp_path,
            run_id="run-1",
            section_id="s1",
            version="1",
            markdown=original + "\n新增内容。",
            input_versions={},
        )


def test_parse_section_markdown_rejects_protocol_code_fences():
    with pytest.raises(MarkdownProtocolError, match="markdown_code_fence_forbidden"):
        parse_section_markdown(
            "```markdown\n# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n正文\n\n## 证据缺口\n无\n```"
        )


@pytest.mark.parametrize(
    "markdown, code",
    [
        ("## 本章摘要\n内容", "markdown_missing_title"),
        ("# 标题\n\n## 正文\n内容\n\n## 证据缺口\n无", "markdown_missing_本章摘要"),
        ("# 标题\n\n## 本章摘要\n摘要\n\n## 正文\n内容", "markdown_missing_证据缺口"),
    ],
)
def test_parse_section_markdown_rejects_missing_required_sections(markdown, code):
    with pytest.raises(MarkdownProtocolError, match=code):
        parse_section_markdown(markdown)
