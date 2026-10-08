from ai_status_report.storage.search_results import (
    persist_evidence_chunks,
    register_run_directory,
    run_directory,
    run_id_for_directory,
)


def test_only_retrieved_sources_become_evidence_chunks(tmp_path):
    chunks = persist_evidence_chunks(
        tmp_path,
        [
            {
                "source_id": "reported",
                "provider": "deepseek_native_web_search",
                "title": "Candidate",
                "url": "https://example.com/candidate",
                "snippet": "Model-reported excerpt",
                "verification_status": "reported",
            },
            {
                "source_id": "retrieved",
                "provider": "tavily_mcp",
                "title": "Retrieved",
                "url": "https://example.com/retrieved",
                "snippet": "Tool-retrieved excerpt",
                "content_ref": "page.md",
                "verification_status": "retrieved",
            },
        ],
        raw_ref="raw.json",
    )

    assert [chunk.source_id for chunk in chunks] == ["retrieved"]
    assert chunks[0].verification_status == "retrieved"
    assert chunks[0].content_ref == "page.md"
    assert len(list((tmp_path / "data" / "evidence").glob("*.json"))) == 1


def test_invalid_url_never_becomes_evidence(tmp_path):
    chunks = persist_evidence_chunks(
        tmp_path,
        [
            {
                "source_id": "bad",
                "provider": "tavily_mcp",
                "url": "not-a-url",
                "snippet": "content",
                "verification_status": "retrieved",
            }
        ],
        raw_ref="raw.json",
    )

    assert chunks == []


def test_long_retrieved_content_is_split_into_bounded_chunks(tmp_path):
    content = "\n\n".join([f"段落 {index}：" + "内容" * 1200 for index in range(3)])
    chunks = persist_evidence_chunks(
        tmp_path,
        [
            {
                "source_id": "long-source",
                "provider": "tavily_mcp",
                "title": "Long page",
                "url": "https://example.com/long",
                "snippet": "short fallback",
                "verification_status": "retrieved",
            }
        ],
        raw_ref="extract.json",
        content_by_source_id={"long-source": content},
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 7000 for chunk in chunks)
    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
    assert len(list((tmp_path / "data" / "evidence").glob("*.json"))) == len(chunks)


def test_run_id_groups_evidence_under_isolated_data_root(tmp_path):
    chunks = persist_evidence_chunks(
        tmp_path,
        [
            {
                "source_id": "run-source",
                "provider": "tavily_mcp",
                "url": "https://example.com/run",
                "snippet": "run content",
                "verification_status": "retrieved",
            }
        ],
        raw_ref="run-extract.json",
        run_id="run-test-001",
    )

    assert len(chunks) == 1
    assert (tmp_path / "data" / "runs" / "run-test-001" / "evidence" / f"{chunks[0].chunk_id}.json").exists()
    assert not (tmp_path / "data" / "evidence").exists()


def test_topic_labelled_directory_preserves_stable_run_id(tmp_path):
    run_id = "teach-20260920T072109689333Z"
    directory = register_run_directory(tmp_path, run_id, "人工智能现状分析报告：/不安全字符")

    assert directory.name == "teach-20260920T072109689333Z__人工智能现状分析报告 不安全字符"
    assert run_directory(tmp_path, run_id) == directory
    assert run_id_for_directory(tmp_path, directory.name) == run_id


def test_legacy_run_directory_stays_compatible_without_a_label_record(tmp_path):
    run_id = "legacy-run-001"

    assert run_directory(tmp_path, run_id) == tmp_path / "data" / "runs" / run_id
    assert run_id_for_directory(tmp_path, run_id) == run_id


def test_run_directory_rejects_mapping_to_another_run(tmp_path):
    registry = tmp_path / "data" / "runs" / ".run_labels"
    registry.mkdir(parents=True)
    (registry / "run-a.json").write_text(
        '{"run_id":"run-a","directory_name":"run-b__其他主题"}\n',
        encoding="utf-8",
    )

    try:
        run_directory(tmp_path, "run-a")
    except ValueError as exc:
        assert str(exc) == "run_directory_mapping_invalid"
    else:
        raise AssertionError("cross-run mapping must be rejected")
