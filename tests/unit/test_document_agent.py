"""Document Agent task parsing tests without an A2A HTTP server."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import ai_status_report.agents.document as document_agent
from ai_status_report.agents.document import (
    DocumentEvidenceRetriever,
    DocumentSectionReviser,
    DocumentSectionWriter,
    RetrievedEvidenceBundle,
    parse_document_task,
)
from ai_status_report.context.document import DocumentContextError
from ai_status_report.rag.schemas import RetrievalMatch
from ai_status_report.schemas.document import DocumentReviseSectionTask, DocumentWriteSectionTask
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt


def write_payload() -> dict[str, object]:
    return {
        "task_type": "document.write_section",
        "run_id": "run-doc-001",
        "research_report_ref": "data/runs/run-doc-001/reports/research.json",
        "research_summary": "研究摘要。",
        "outline_ref": "data/runs/run-doc-001/outlines/v001.json",
        "outline_version": "1",
        "section_id": "s1",
        "section_title": "智能体交互方式",
        "section_goal": "说明主要交互模式。",
        "retrieval_queries": ["智能体交互方式"],
        "budget": {
            "context_window_tokens": 16000,
            "evidence_tokens": 4000,
            "reserved_input_tokens": 3000,
            "max_output_tokens": 3000,
        },
    }


def test_parse_document_task_accepts_a_valid_write_task():
    task = parse_document_task(json.dumps(write_payload(), ensure_ascii=False))

    assert isinstance(task, DocumentWriteSectionTask)
    assert task.section_id == "s1"


def test_parse_document_task_returns_safe_codes_for_invalid_payloads():
    with pytest.raises(ValueError, match="invalid_document_task"):
        parse_document_task("not json")
    for task_type in ([], {"unexpected": "object"}):
        with pytest.raises(ValueError, match="invalid_document_task"):
            parse_document_task(json.dumps({"task_type": task_type}))
    with pytest.raises(ValueError, match="unsupported_document_task"):
        parse_document_task(json.dumps({"task_type": "research.search"}))


def test_parse_document_task_rejects_a_likely_encoding_corrupted_query():
    payload = write_payload() | {"retrieval_queries": ["????????"]}

    with pytest.raises(ValueError, match="input_encoding_corrupted"):
        parse_document_task(json.dumps(payload))


def test_parse_document_task_accepts_a_valid_revision_task():
    payload = write_payload() | {
        "task_type": "document.revise_section",
        "base_draft": {
            "artifact_id": "artifact-run-doc-001-chapter_markdown-1",
            "run_id": "run-doc-001",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "access_ref": "data/runs/run-doc-001/chapters/s1/v001.md",
        },
        "target_draft_version": "2",
        "review_issues": [
            {
                "issue_id": "issue-1",
                "section_id": "s1",
                "draft_version": "1",
                "type": "citation",
                "requested_change": "补充引用。",
            }
        ],
        "writing_constraints": ["写作风格：analytical", "使用场景：student"],
    }
    payload.pop("research_summary")
    payload.pop("section_goal")

    task = parse_document_task(json.dumps(payload, ensure_ascii=False))

    assert isinstance(task, DocumentReviseSectionTask)
    assert task.target_draft_version == "2"


def test_document_evidence_retriever_uses_d05_query_and_persistence(monkeypatch, tmp_path):
    queries_seen: list[str] = []

    class FakeGLMClient:
        def __init__(self, embedding, timeout, *, ledger):
            assert embedding == "embedding-config"
            assert timeout == 20
            assert ledger.run_id == "run-doc-001"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    class FakeIndex:
        index_version = "v2"

        def __init__(self, root, *, dimensions):
            assert root == tmp_path
            assert dimensions == 2048

        def query(self, question, client, *, n_results, run_id):
            assert isinstance(client, FakeGLMClient)
            assert n_results == 32
            assert run_id == "run-doc-001"
            queries_seen.append(question)
            return [
                RetrievalMatch(
                    retrieval_chunk_id=f"retrieval-{len(queries_seen)}",
                    text=f"Evidence for {question}.",
                    distance=0.2,
                    evidence_chunk_id=f"evidence-{len(queries_seen)}",
                    source_id=f"source-{len(queries_seen)}",
                    run_id="run-doc-001",
                    provider="tavily_mcp",
                    url="https://example.com/evidence",
                    verification_status="retrieved",
                    end_offset=20,
                )
            ]

    monkeypatch.setattr(
        document_agent,
        "load_settings",
        lambda root: SimpleNamespace(embedding="embedding-config", timeout=20, dimensions=2048),
    )
    monkeypatch.setattr(document_agent, "GLMClient", FakeGLMClient)
    monkeypatch.setattr(document_agent, "ChromaEvidenceIndex", FakeIndex)
    task = parse_document_task(
        json.dumps(
            write_payload() | {"retrieval_queries": ["智能体协作", "协议交互"]}, ensure_ascii=False
        )
    )

    retrieved = DocumentEvidenceRetriever(tmp_path)(task)

    assert queries_seen == ["智能体协作", "协议交互"]
    assert retrieved.path.is_file()
    assert retrieved.path.parent == tmp_path / "data" / "runs" / "run-doc-001" / "evidence_bundles"
    assert retrieved.bundle.index_version == "v2"
    assert retrieved.bundle.chunk_refs == ["retrieval-1", "retrieval-2"]


def test_document_section_writer_uses_skill_context_and_markdown_protocol(monkeypatch, tmp_path):
    skill_file = tmp_path / "skills" / "report-writing" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: report-writing\ndescription: test guidance\n---\n\n必须使用现有证据。\n",
        encoding="utf-8",
    )
    task = parse_document_task(json.dumps(write_payload(), ensure_ascii=False))
    retrieved = RetrievedEvidenceBundle(
        bundle=EvidenceBundle(
            retrieval_id="retrieval-1",
            run_id="run-doc-001",
            section_id="s1",
            excerpts=[Excerpt(chunk_id="chunk-1", source_id="source-1", text="可追溯证据。")],
        ),
        path=tmp_path / "bundle.json",
    )
    calls: list[tuple[list[dict], dict]] = []

    class FakeDeepSeekClient:
        def __init__(self, generation, timeout):
            assert generation == "generation-config"
            assert timeout == 20

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def chat(self, messages, **options):
            calls.append((messages, options))
            return {
                "choices": [
                    {
                        "message": {
                            "content": "# 智能体交互方式\n\n## 本章摘要\n\n摘要。\n\n## 正文\n\n结论。[1]\n\n## 证据缺口\n\n无"
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        document_agent,
        "load_settings",
        lambda root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(document_agent, "DeepSeekClient", FakeDeepSeekClient)

    writer = DocumentSectionWriter(tmp_path)
    generated = writer.generate(task, writer.prepare(task, retrieved))

    assert generated.parsed.summary == "摘要。"
    assert generated.parsed.citation_numbers == (1,)
    assert "必须使用现有证据。" in calls[0][0][0]["content"]
    assert "reader_citation=[1]" in calls[0][0][1]["content"]
    assert calls[0][1] == {
        "max_tokens": 3000,
        "thinking": {"type": "disabled"},
        "temperature": 0.7,
    }


def test_document_section_reviser_uses_base_artifact_and_revision_skill(monkeypatch, tmp_path):
    """A revision context includes the reviewed v1 chapter and current evidence."""

    skill_file = tmp_path / "skills" / "report-revision" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        "---\nname: report-revision\ndescription: test revision guidance\n---\n\n必须处理审核意见。\n",
        encoding="utf-8",
    )
    outline_path = tmp_path / "data" / "runs" / "run-doc-001" / "outlines" / "v001.json"
    outline_path.parent.mkdir(parents=True)
    outline_path.write_text(
        json.dumps(
            {
                "outline_version": "1",
                "sections": [{"section_id": "s1", "title": "智能体交互方式"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    base_path = tmp_path / "data" / "runs" / "run-doc-001" / "chapters" / "s1" / "v1.md"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("# 智能体交互方式\n\n旧版章节正文。\n", encoding="utf-8")
    payload = write_payload() | {
        "task_type": "document.revise_section",
        "base_draft": {
            "artifact_id": "artifact-run-doc-001-chapter-1",
            "run_id": "run-doc-001",
            "version": "1",
            "kind": "chapter_markdown",
                "mime_type": "text/markdown",
                "size": base_path.stat().st_size,
                "hash": hashlib.sha256(base_path.read_bytes()).hexdigest(),
            "access_ref": str(base_path),
        },
        "target_draft_version": "2",
        "review_issues": [
            {
                "issue_id": "issue-1",
                "section_id": "s1",
                "draft_version": "1",
                "type": "citation",
                "requested_change": "补充可追溯引用。",
            }
        ],
        "writing_constraints": ["写作风格：analytical", "使用场景：student"],
    }
    payload.pop("research_summary")
    payload.pop("section_goal")
    task = parse_document_task(json.dumps(payload, ensure_ascii=False))
    assert isinstance(task, DocumentReviseSectionTask)
    retrieved = RetrievedEvidenceBundle(
        bundle=EvidenceBundle(
            retrieval_id="retrieval-2",
            run_id="run-doc-001",
            section_id="s1",
            excerpts=[Excerpt(chunk_id="chunk-1", source_id="source-1", text="新证据。")],
        ),
        path=tmp_path / "bundle.json",
    )
    calls: list[tuple[list[dict], dict]] = []

    class FakeDeepSeekClient:
        def __init__(self, generation, timeout):
            assert generation == "generation-config"
            assert timeout == 20

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def chat(self, messages, **options):
            calls.append((messages, options))
            return {
                "choices": [
                    {
                        "message": {
                            "content": "# 智能体交互方式\n\n## 本章摘要\n\n修订摘要。\n\n## 正文\n\n补充后的结论。[1]\n\n## 证据缺口\n\n无"
                        }
                    }
                ]
            }

    monkeypatch.setattr(
        document_agent,
        "load_settings",
        lambda root: SimpleNamespace(generation="generation-config", timeout=20),
    )
    monkeypatch.setattr(document_agent, "DeepSeekClient", FakeDeepSeekClient)

    reviser = DocumentSectionReviser(tmp_path)
    generated = reviser.generate(task, reviser.prepare(task, retrieved))

    assert generated.parsed.summary == "修订摘要。"
    assert "必须处理审核意见。" in calls[0][0][0]["content"]
    assert "旧版章节正文。" in calls[0][0][1]["content"]
    assert "补充可追溯引用。" in calls[0][0][1]["content"]
    assert "写作风格：analytical" in calls[0][0][1]["content"]
    assert "使用场景：student" in calls[0][0][1]["content"]
    assert calls[0][1] == {
        "max_tokens": 3000,
        "thinking": {"type": "disabled"},
        "temperature": 0.3,
    }
    base_path.write_text("# 篡改的章节\n", encoding="utf-8")
    with pytest.raises(DocumentContextError, match="revision_base_draft_unavailable"):
        reviser.prepare(task, retrieved)
