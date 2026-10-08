from ai_status_report.search.routing import choose_provider


def test_search_route_defaults_to_tavily():
    assert choose_provider(query="人工智能行业概况").providers == ("tavily",)


def test_search_route_can_cross_check_critical_facts():
    assert choose_provider(query="市场规模", critical=True).providers == ("tavily", "deepseek")
