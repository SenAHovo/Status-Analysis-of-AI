import json

from ai_status_report.rag.ingest import load_run_evidence
from ai_status_report.storage.search_results import persist_evidence_chunks


def test_load_run_evidence_recovers_page_reference(tmp_path):
    run_root = tmp_path / "data" / "runs" / "run-1"
    (run_root / "evidence").mkdir(parents=True)
    (run_root / "raw" / "pages").mkdir(parents=True)
    (run_root / "raw" / "pages" / "page.json").write_text(
        json.dumps({"source_id": "source-1", "content_ref": "page.md"}),
        encoding="utf-8",
    )
    (run_root / "evidence" / "chunk.json").write_text(
        json.dumps(
            {
                "chunk_id": "chunk-1",
                "source_id": "source-1",
                "text": "content",
                "provider": "tavily_mcp",
                "content_hash": "hash",
                "raw_ref": "raw.json",
                "created_at": "2026-09-10T00:00:00Z",
                "verification_status": "retrieved",
            }
        ),
        encoding="utf-8",
    )

    evidence = load_run_evidence(tmp_path, "run-1")

    assert evidence[0].content_ref == "page.md"


def test_storage_and_ingest_share_run_id_canonicalization(tmp_path):
    persist_evidence_chunks(
        tmp_path,
        [
            {
                "source_id": "source-1",
                "provider": "tavily_mcp",
                "url": "https://example.com/source",
                "snippet": "content",
                "verification_status": "retrieved",
            }
        ],
        raw_ref="raw.json",
        run_id="run test/001",
    )

    evidence = load_run_evidence(tmp_path, "run test/001")

    assert len(evidence) == 1
    assert (tmp_path / "data" / "runs" / "run-test-001").is_dir()
