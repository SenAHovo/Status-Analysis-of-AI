from ai_status_report.model.search_cache import WebSearchCache


def test_web_search_cache_key_changes_with_query_and_options(tmp_path):
    cache = WebSearchCache(tmp_path / "search.sqlite")
    first = cache.key(model="m", query="a", options={"max_output_tokens": 10})
    second = cache.key(model="m", query="b", options={"max_output_tokens": 10})
    assert first != second
    cache.put(first, {"status": "completed"})
    assert cache.get(first)["status"] == "completed"
