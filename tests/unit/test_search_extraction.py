from types import SimpleNamespace

from ai_status_report.schemas.search import ResearchSource
from ai_status_report.search.extraction import normalize_extract_response, select_extract_sources
from ai_status_report.storage.search_results import persist_web_content


def _source(source_id: str, url: str, title: str) -> ResearchSource:
    return ResearchSource(
        source_id=source_id,
        provider="tavily_mcp",
        url=url,
        title=title,
        retrieved_at="2026-09-10T00:00:00Z",
    )


def test_paper_sources_receive_small_selection_boost():
    sources = [
        _source("news", "https://example.com/news", "Industry news"),
        _source("paper", "https://arxiv.org/abs/1234.5678", "Research paper"),
    ]

    selected = select_extract_sources(sources, limit=1)

    assert [source.source_id for source in selected] == ["paper"]


def test_extract_response_normalizes_structured_content():
    response = {
        "results": [
            {"url": "https://example.com/article", "raw_content": "# Article\n正文"},
            {"url": "https://example.com/article", "raw_content": "duplicate"},
        ]
    }

    assert normalize_extract_response(response) == [
        {
            "url": "https://example.com/article",
            "raw_content": "# Article\n正文",
            "title": "",
            "favicon": "",
        }
    ]


def test_extract_response_normalizes_mcp_text_content():
    response = [
        SimpleNamespace(
            text='{"results":[{"url":"https://example.com/article","raw_content":"正文"}]}'
        )
    ]

    assert normalize_extract_response(response)[0]["raw_content"] == "正文"


def test_web_content_persistence_keeps_trace_metadata(tmp_path):
    destination = persist_web_content(
        tmp_path,
        source={
            "source_id": "source-1",
            "url": "https://example.com/article?utm_source=test",
            "title": "Article",
        },
        content="# Article\n正文",
        extract_ref="data/raw/search/extract.json",
        extract_depth="basic",
    )

    assert destination.exists()
    assert destination.suffix == ".md"
    metadata = destination.with_suffix(".json").read_text(encoding="utf-8")
    assert '"tool": "tavily_extract"' in metadata
    assert '"source_url": "https://example.com/article"' in metadata
