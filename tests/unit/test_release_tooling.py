import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_snapshot_omits_evidence_text_and_search_reports(tmp_path):
    snapshot = _load_script("prepare_public_run_snapshot")
    source = tmp_path / "source-run"
    target = tmp_path / "public-run"
    (source / "evidence").mkdir(parents=True)
    (source / "raw").mkdir()
    (source / "reports").mkdir()
    (source / "chapters").mkdir()
    (source / "evidence" / "chunk.json").write_text(
        json.dumps(
            {
                "source_id": "source-1",
                "title": "A source",
                "url": "https://example.test/source?token=private-value&topic=report",
                "provider": "tavily_mcp",
                "verification_status": "retrieved",
                "content_hash": "a" * 64,
                "text": "third-party source body must not be copied",
            }
        ),
        encoding="utf-8",
    )
    (source / "raw" / "page.json").write_text('{"content":"private source body"}', encoding="utf-8")
    (source / "reports" / "query-search.json").write_text('{"snippet":"source body"}', encoding="utf-8")
    (source / "reports" / "report.md").write_text("# Report", encoding="utf-8")
    (source / "reports" / "report.pdf").write_bytes(b"%PDF")
    (source / "chapters" / "section.md").write_text("# Chapter", encoding="utf-8")
    (source / "traces").mkdir()
    (source / "traces" / "trace.jsonl").write_text(
        json.dumps(
            {
                "summary": json.dumps(
                    {
                        "sources": [
                            {
                                "title": "A source",
                                "url": "https://example.test/source?token=private-value&topic=report",
                                "snippet": "third-party source body must not be copied",
                            }
                        ]
                    }
                )
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = snapshot.prepare_snapshot(source, target)

    assert result["source_count"] == 1
    assert not (target / "raw").exists()
    assert not (target / "evidence").exists()
    assert not (target / "evidence_bundles").exists()
    assert not (target / "reports" / "query-search.json").exists()
    assert (target / "reports" / "report.md").is_file()
    assert (target / "reports" / "report.pdf").is_file()
    trace = (target / "traces" / "trace.jsonl").read_text(encoding="utf-8")
    assert "third-party source body" not in trace
    assert "[not-published]" in trace
    assert json.loads((target / "sources.json").read_text(encoding="utf-8")) == [
        {
            "source_id": "source-1",
            "title": "A source",
            "url": "https://example.test/source?topic=report",
            "provider": "tavily_mcp",
            "verification_status": "retrieved",
            "content_sha256": "a" * 64,
        }
    ]


def test_public_secret_scan_ignores_named_test_placeholder_and_detects_key_shape():
    audit = _load_script("audit_secrets")

    assert audit._generic_hits(b"sk-p0-placeholder-only") == 0
    assert audit._generic_hits(b"sk-" + b"1234567890abcdefghijklmnop") == 1
    assert audit._generic_hits(b"GLM_EMBEDDING_API_KEY=" + b"actual-value-123") == 1
    assert audit._generic_hits(b"GLM_EMBEDDING_API_KEY=test-placeholder-value") == 0
