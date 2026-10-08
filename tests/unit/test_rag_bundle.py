import json

from ai_status_report.rag.bundle import build_evidence_bundle, persist_evidence_bundle
from ai_status_report.rag.schemas import RetrievalMatch


def match(
    chunk_id: str,
    text: str,
    distance: float,
    source_id: str = "src-1",
    url: str = "https://example.com/a",
):
    return RetrievalMatch(
        retrieval_chunk_id=chunk_id,
        text=text,
        distance=distance,
        evidence_chunk_id=f"e-{chunk_id}" if len(chunk_id) <= 158 else "e-parent",
        source_id=source_id,
        run_id="run-1",
        title="title",
        url=url,
        raw_ref="raw.json",
        content_ref="page.md",
        verification_status="retrieved",
    )


def test_bundle_deduplicates_and_preserves_provenance():
    bundle = build_evidence_bundle(
        run_id="run/1",
        section_id="s1",
        queries=["  研究问题  "],
        matches=[match("b", "相同内容", 0.2), match("a", "相同内容", 0.1)],
        budget_tokens=100,
    )
    assert bundle.run_id == "run-1"
    assert bundle.index_version == "v2"
    assert bundle.chunk_refs == ["a"]
    assert bundle.excerpts[0].content_ref == "page.md"
    assert "deduplicated" in bundle.quality_flags


def test_bundle_clips_whole_excerpts_to_budget():
    bundle = build_evidence_bundle(
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[match("a", "第一段证据", 0.1), match("b", "第二段证据", 0.2)],
        budget_tokens=8,
    )
    assert len(bundle.excerpts) == 1
    assert bundle.budget_used <= 8
    assert "budget_clipped" in bundle.quality_flags


def test_bundle_rotates_available_sources_without_limiting_source_chunk_count():
    bundle = build_evidence_bundle(
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[
            match("a1", "来源一片段一", 0.1, source_id="src-1", url="https://example.com/one"),
            match("a2", "来源一片段二", 0.2, source_id="src-1", url="https://example.com/one"),
            match("a3", "来源一片段三", 0.3, source_id="src-1", url="https://example.com/one"),
            match("b1", "来源二片段一", 0.4, source_id="src-2", url="https://example.com/two"),
            match("b2", "来源二片段二", 0.5, source_id="src-2", url="https://example.com/two"),
        ],
        budget_tokens=100,
    )

    assert bundle.chunk_refs == ["a1", "b1", "a2", "b2", "a3"]
    assert [excerpt.source_id for excerpt in bundle.excerpts] == [
        "src-1",
        "src-2",
        "src-1",
        "src-2",
        "src-1",
    ]


def test_bundle_persistence_is_run_scoped(tmp_path):
    bundle = build_evidence_bundle(
        run_id="run-1", section_id="s1", queries=["q"], matches=[], budget_tokens=10
    )
    path = persist_evidence_bundle(tmp_path, bundle)
    assert path.parent.name == "evidence_bundles"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-1"
    assert payload["quality_flags"] == ["no_excerpts"]


def test_retrieval_id_stays_within_contract_for_long_inputs():
    bundle = build_evidence_bundle(
        run_id="r" * 128,
        section_id="s" * 64,
        queries=["q"],
        matches=[],
        budget_tokens=10,
    )
    assert len(bundle.retrieval_id) <= 128


def test_bundle_fetches_parent_evidence_when_budget_allows(tmp_path):
    evidence_dir = tmp_path / "data" / "runs" / "run-1" / "evidence"
    evidence_dir.mkdir(parents=True)
    parent = {
        "schema_version": "1",
        "chunk_id": "e-a",
        "source_id": "src-1",
        "text": "父级证据包含更完整的上下文。",
        "provider": "tavily",
        "content_hash": "hash",
        "created_at": "2026-09-11T00:00:00Z",
        "verification_status": "retrieved",
    }
    (evidence_dir / "e-a.json").write_text(json.dumps(parent), encoding="utf-8")
    bundle = build_evidence_bundle(
        root=tmp_path,
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[match("a", "检索片段", 0.1)],
        budget_tokens=100,
    )
    assert bundle.excerpts[0].text == parent["text"]
    assert bundle.excerpts[0].parent_chunk_id == "e-a"


def test_bundle_falls_back_to_retrieval_chunk_when_parent_exceeds_budget(tmp_path):
    evidence_dir = tmp_path / "data" / "runs" / "run-1" / "evidence"
    evidence_dir.mkdir(parents=True)
    parent = {
        "schema_version": "1",
        "chunk_id": "e-a",
        "source_id": "src-1",
        "text": "父" * 100,
        "provider": "tavily",
        "content_hash": "hash",
        "created_at": "2026-09-11T00:00:00Z",
        "verification_status": "retrieved",
    }
    (evidence_dir / "e-a.json").write_text(json.dumps(parent), encoding="utf-8")
    bundle = build_evidence_bundle(
        root=tmp_path,
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[match("a", "检索片段", 0.1)],
        budget_tokens=20,
    )
    assert bundle.excerpts[0].text == "检索片段"
    assert "parent_budget_fallback" in bundle.quality_flags


def test_bundle_accepts_retrieval_contract_identifier_lengths():
    long_match = match("r" * 200, "证据", 0.1, source_id="s" * 160)
    bundle = build_evidence_bundle(
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[long_match],
        budget_tokens=20,
    )
    assert bundle.excerpts[0].chunk_id == "r" * 200
    assert bundle.excerpts[0].source_id == "s" * 160


def test_bundle_decodes_literal_unicode_escapes_for_readability():
    bundle = build_evidence_bundle(
        run_id="run-1",
        section_id="s1",
        queries=["q"],
        matches=[match("a", r"\u8fd9\u662f\u5185\u5bb9\u3002", 0.1)],
        budget_tokens=20,
    )
    assert bundle.excerpts[0].text == "这是内容。"
