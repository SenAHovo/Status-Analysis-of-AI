
from ai_status_report.search.adapters import deepseek_report, tavily_report


def test_tavily_result_is_normalized():
    report = tavily_report("q", {"answer": "summary", "results": [{"title": "T", "url": "https://e", "content": "C"}]})
    assert report.sources[0].provider == "tavily_mcp"
    assert report.sources[0].url == "https://e"
    assert report.sources[0].verification_status == "retrieved"


def test_tavily_without_sources_is_partial():
    report = tavily_report("q", {"answer": "summary", "results": []})

    assert report.status == "partial"
    assert report.incomplete_reason == "no_citations"


def test_tavily_source_ids_are_query_scoped():
    first = tavily_report("query one", {"answer": "a", "results": [{"url": "https://one", "content": "A"}]})
    second = tavily_report("query two", {"answer": "b", "results": [{"url": "https://two", "content": "B"}]})

    assert first.sources[0].source_id != second.sources[0].source_id


def test_deepseek_message_is_normalized():
    report = deepseek_report(
        "q",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"summary":"summary","sources":[{"title":"T","url":"https://e","snippet":"C"}]}',
                        }
                    ],
                }
            ],
        },
    )
    assert report.provider == "deepseek_native_web_search"
    assert report.summary == "summary"
    assert report.status == "completed"
    assert report.sources[0].verification_status == "reported"


def test_deepseek_rejects_non_http_source_urls():
    report = deepseek_report(
        "q",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [
                        {
                            "type": "output_text",
                            "text": '{"summary":"summary","sources":[{"url":"javascript:bad","snippet":"C"}]}',
                        }
                    ],
                }
            ],
        },
    )

    assert report.sources == []
    assert report.status == "partial"


def test_deepseek_commentary_is_not_a_completed_report():
    report = deepseek_report(
        "q",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "intermediate"}],
                }
            ],
        },
    )
    assert report.status == "partial"
    assert report.summary == ""


def test_deepseek_completed_without_citations_is_partial():
    report = deepseek_report(
        "q",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": [{"type": "output_text", "text": "summary"}],
                }
            ],
        },
    )
    assert report.status == "partial"
    assert report.incomplete_reason == "no_citations"


def test_deepseek_incomplete_response_preserves_reason():
    report = deepseek_report(
        "q",
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
    )
    assert report.status == "partial"
    assert report.incomplete_reason == "max_output_tokens"


def test_deepseek_malformed_nested_response_becomes_partial():
    report = deepseek_report(
        "q",
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "phase": "final_answer",
                    "content": "malformed-content",
                }
            ],
            "incomplete_details": "malformed-details",
        },
    )
    assert report.status == "partial"
    assert report.incomplete_reason == "invalid_provider_response"
