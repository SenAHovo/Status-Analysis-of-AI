from ai_status_report.a2a.apps import document_generation_app, network_search_app


def test_two_specialist_apps_are_separate_local_a2a_services():
    assert network_search_app() is not document_generation_app()
