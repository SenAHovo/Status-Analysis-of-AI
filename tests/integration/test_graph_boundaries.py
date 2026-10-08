import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from a2a.helpers import new_text_part
from a2a.types import Artifact, StreamResponse, TaskArtifactUpdateEvent

from ai_status_report.documents.markdown import persist_section_markdown
from ai_status_report.harness.graph import (
    _validate_document_result,
    _verified_chapter_artifact_hash,
    build_graph,
)
from ai_status_report.model.router import ModelRoute
from ai_status_report.rag.chroma import RagIndexError
from ai_status_report.schemas.document import (
    DocumentBudget,
    DocumentSectionResult,
    DocumentWriteSectionTask,
)
from ai_status_report.schemas.intent import Intent
from ai_status_report.schemas.report import ArtifactRef


def _persist_chapter_artifact(
    root: Path, *, run_id: str, section_id: str, version: str, markdown: str
) -> dict[str, object]:
    """Create the local immutable file a real Document Agent would return."""

    path = root / "data" / "runs" / run_id / "chapters" / section_id / f"v{version}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = markdown.encode("utf-8")
    path.write_bytes(content)
    return {
        "artifact_id": f"artifact-{run_id}-{section_id}-{version}",
        "run_id": run_id,
        "version": version,
        "kind": "chapter_markdown",
        "mime_type": "text/markdown",
        "size": len(content),
        "hash": hashlib.sha256(content).hexdigest(),
        "access_ref": str(path),
    }
from ai_status_report.schemas.review import ReviewDecision


def test_graph_reaches_protocol_dispatch_boundary(tmp_path):
    calls = []

    def classify(text):
        calls.append("classify")
        return "report_request"

    result = asyncio.run(
        build_graph(classify=classify, root=tmp_path).ainvoke(
            {"run_id": "run-1", "user_input": "分析人工智能现状"}
        )
    )

    assert calls == ["classify"]
    assert result["phase"] == "awaiting_a2a_dispatch"


def test_graph_accepts_model_route_intent_object(tmp_path):
    result = asyncio.run(
        build_graph(
            classify=lambda _text: ModelRoute(
                intent=Intent.REPORT_REQUEST,
                reason="model-confirmed",
                confidence=0.9,
            ),
            root=tmp_path,
        ).ainvoke({"run_id": "run-model-route", "user_input": "分析人工智能现状"})
    )

    assert result["intent"] == "report_request"
    assert result["phase"] == "awaiting_a2a_dispatch"


def test_whole_report_graph_assembles_four_verified_chapters_and_exports_pdf(monkeypatch, tmp_path):
    search_sections = []
    document_contexts = []
    document_constraints = []

    async def fake_send_text_task(url, text):
        payload = json.loads(text)
        if payload["task_type"] == "research.search":
            search_sections.append(payload["query"])
            yield json.dumps(
                {
                    "query": payload["query"],
                    "provider": "test",
                    "status": "completed",
                    "summary": "检索摘要。",
                    "sources": [],
                },
                ensure_ascii=False,
            )
            return
        document_contexts.append(payload["recent_section_summaries"])
        document_constraints.append(payload["writing_constraints"])
        section_id = payload["section_id"]
        artifact = persist_section_markdown(
            tmp_path,
            run_id=payload["run_id"],
            section_id=section_id,
            version="1",
            markdown=(
                f"# {payload['section_title']}\n\n## 本章摘要\n\n"
                f"approved-summary-{section_id}\n\n## 正文\n\n"
                f"{section_id}正文。[1]\n\n## 参考来源\n\n"
                "[1] 统一来源。https://example.com/source\n"
            ),
            input_versions={},
        )
        yield json.dumps(
            {
                "task_type": "document.write_section",
                "status": "drafted",
                "run_id": payload["run_id"],
                "section_id": section_id,
                "draft_version": "1",
                "draft_artifact": artifact.model_dump(mode="json"),
                "section_summary": "untrusted-a2a-summary",
                "evidence_bundle_ref": "data/runs/test/evidence_bundles/bundle.json",
                "citation_map_ref": "data/runs/test/chapters/citation-map.json",
            },
            ensure_ascii=False,
        )

    def fake_export(root, *, report_artifact, output_pdf=None):
        output = tmp_path / "data" / "runs" / report_artifact.run_id / "reports" / "report.pdf"
        output.write_bytes(b"%PDF-test")
        return ArtifactRef(
            artifact_id="artifact-pdf-test",
            run_id=report_artifact.run_id,
            version="1",
            kind="report_pdf",
            mime_type="application/pdf",
            size=output.stat().st_size,
            hash=hashlib.sha256(output.read_bytes()).hexdigest(),
            access_ref=str(output),
            input_versions={"report_markdown": report_artifact.version},
        )

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        whole_report=True,
        evidence_indexer=lambda root, run_id: {"run_id": run_id, "index_version": "v2"},
        evidence_reviewer=lambda task, result, review_round: ReviewDecision(
            decision="approve",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            review_round=review_round,
            reason="approved",
        ),
    )
    monkeypatch.setattr("ai_status_report.harness.graph.export_report_pdf", fake_export)

    result = asyncio.run(
        graph.ainvoke(
            {
                "run_id": "run-whole-001",
                "user_input": "分析人工智能",
                "research_brief": {
                    "writing_style": "analytical",
                    "audience": "student",
                },
            }
        )
    )

    assert result["phase"] == "report_pdf_exported"
    assert len(search_sections) == 4
    assert [len(context) for context in document_contexts] == [0, 1, 2, 3]
    assert all("写作风格：深度分析" in items for items in document_constraints)
    assert all("使用场景：学生学习" in items for items in document_constraints)
    assert all(not any("资料偏好" in item or "篇幅目标" in item for item in items)
               for items in document_constraints)
    assert document_contexts[1][0]["summary"] == "approved-summary-background"
    assert "untrusted-a2a-summary" not in json.dumps(document_contexts, ensure_ascii=False)
    assert list(result["approved_chapter_artifacts"]) == [
        "background",
        "current-status",
        "trends",
        "recommendations",
    ]
    assert Path(result["report_markdown_artifact"]["access_ref"]).is_file()
    assert Path(result["report_pdf_artifact"]["access_ref"]).is_file()


def test_controller_rejects_document_artifact_for_a_different_dispatched_task():
    task = DocumentWriteSectionTask(
        run_id="run-expected",
        research_report_ref="data/runs/run-expected/reports/research.json",
        research_summary="研究摘要。",
        outline_ref="data/runs/run-expected/outlines/v1.json",
        outline_version="1",
        section_id="expected-section",
        section_title="预期章节",
        section_goal="预期目标。",
        retrieval_queries=["预期查询"],
        budget=DocumentBudget(
            context_window_tokens=16000,
            evidence_tokens=4000,
            reserved_input_tokens=3000,
            max_output_tokens=3000,
        ),
    )
    result = DocumentSectionResult(
        task_type="document.write_section",
        status="drafted",
        run_id="run-other",
        section_id="expected-section",
        draft_version="1",
        draft_artifact={
            "artifact_id": "artifact-other",
            "run_id": "run-other",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "access_ref": "data/runs/run-other/chapters/expected-section/v1.md",
        },
        evidence_bundle_ref="data/runs/run-other/evidence_bundles/bundle.json",
        citation_map_ref="data/runs/run-other/chapters/expected-section/v1.citations.json",
    )

    with pytest.raises(RuntimeError, match="a2a_document_result_mismatch"):
        _validate_document_result(result, task)


def test_controller_rejects_a_revision_artifact_with_a_forged_hash(tmp_path):
    artifact = _persist_chapter_artifact(
        tmp_path,
        run_id="run-integrity-001",
        section_id="current-status",
        version="2",
        markdown="# 修订稿\n\n实际章节内容。\n",
    )
    artifact["hash"] = "0" * 64

    with pytest.raises(RuntimeError, match="a2a_document_artifact_integrity_failed"):
        _verified_chapter_artifact_hash(
            tmp_path,
            artifact=ArtifactRef.model_validate(artifact),
            section_id="current-status",
        )


def test_graph_ends_non_report_without_protocol_calls(tmp_path):
    calls = []
    graph = build_graph(
        classify=lambda text: "conversation",
        root=tmp_path,
    )

    result = asyncio.run(graph.ainvoke({"run_id": "run-1", "user_input": "你好"}))

    assert result["phase"] == "classified"
    assert calls == []


def test_graph_can_invoke_injected_a2a_client_boundary(monkeypatch, tmp_path):
    async def fake_send_text_task(url, text):
        yield '{"query":"test","provider":"tavily_mcp","status":"completed","sources":[]}'

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        root=tmp_path,
    )

    result = asyncio.run(graph.ainvoke({"run_id": "run-1", "user_input": "test search task"}))

    assert result["phase"] == "search_completed"
    assert result["research_report"]["provider"] == "tavily_mcp"
    assert result["a2a_trace"][0]["kind"] == "outbound_task"
    assert result["a2a_trace"][0]["agent"] == "search_agent"
    assert json.loads(result["a2a_trace"][0]["text"])["task_type"] == "research.search"


def test_graph_marks_partial_search_as_partial(tmp_path):
    async def fake_send_text_task(url, text):
        yield '{"query":"test","provider":"deepseek_native_web_search","status":"partial","sources":[],"plan_source":"deterministic_fallback","providers":["deepseek"],"raw_ref":"raw.json"}'

    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        root=tmp_path,
    )
    import ai_status_report.harness.graph as graph_module

    original = graph_module.send_text_task
    graph_module.send_text_task = fake_send_text_task
    try:
        result = asyncio.run(graph.ainvoke({"run_id": "run-2", "user_input": "test"}))
    finally:
        graph_module.send_text_task = original
    assert result["phase"] == "search_partial"
    assert "search.plan.fallback" in result["events"]
    assert "search.route.selected" in result["events"]
    assert "search.results.partial" in result["events"]
    assert "search.results.merged" not in result["events"]
    assert "search.persisted" in result["events"]


def test_graph_marks_failed_search_without_rag_dispatch(tmp_path):
    async def fake_send_text_task(url, text):
        yield (
            '{"query":"test","provider":"tavily_mcp","status":"failed",'
            '"incomplete_reason":"tavily_rate_limited","sources":[]}'
        )

    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
    )
    import ai_status_report.harness.graph as graph_module

    original = graph_module.send_text_task
    graph_module.send_text_task = fake_send_text_task
    try:
        result = asyncio.run(graph.ainvoke({"run_id": "run-rate-limited", "user_input": "test"}))
    finally:
        graph_module.send_text_task = original

    assert result["phase"] == "search_failed"
    assert "search.results.failed" in result["events"]
    assert "rag.index.started" not in result["events"]


def test_graph_dispatches_document_task_and_persists_controller_trace(monkeypatch, tmp_path):
    search_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "summary": "搜索 Agent 已交付可用于章节撰写的研究摘要。",
        "sources": [],
    }
    document_result = {
        "task_type": "document.write_section",
        "status": "drafted",
        "run_id": "run-controller-doc-001",
        "section_id": "current-status",
        "draft_version": "1",
        "draft_artifact": {
            "artifact_id": "artifact-chapter-1",
            "run_id": "run-controller-doc-001",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "access_ref": "data/runs/run-controller-doc-001/chapters/current-status/v1.md",
        },
        "evidence_bundle_ref": "data/runs/run-controller-doc-001/evidence_bundles/bundle.json",
        "citation_map_ref": "data/runs/run-controller-doc-001/chapters/current-status/v1.citations.json",
    }
    document_result["draft_artifact"] = _persist_chapter_artifact(
        tmp_path,
        run_id="run-controller-doc-001",
        section_id="current-status",
        version="1",
        markdown="# 当前发展现状\n\n正文。\n",
    )

    def artifact_response(*, task_id, content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id="context-1",
                artifact=Artifact(
                    artifact_id=f"artifact-{task_id}",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    async def fake_send_text_task(url, text):
        if url.endswith(":8001/"):
            assert '"task_type": "research.search"' in text
            yield artifact_response(task_id="search-task", content=search_report)
            return
        assert url.endswith(":8002/")
        assert '"task_type":"document.write_section"' in text
        yield artifact_response(task_id="document-task", content=document_result)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    indexed_runs = []

    def fake_evidence_indexer(root, run_id):
        indexed_runs.append((root, run_id))
        return {
            "run_id": run_id,
            "evidence_chunks": 3,
            "indexed_chunks": 9,
            "index_version": "v2",
        }

    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=fake_evidence_indexer,
        evidence_reviewer=lambda task, result, review_round: ReviewDecision(
            decision="approve",
            run_id=result.run_id,
            section_id=result.section_id,
            draft_version=result.draft_version,
            review_round=review_round,
            reason="model review approved the chapter",
            source="model",
        ),
    )

    result = asyncio.run(
        graph.ainvoke({"run_id": "run-controller-doc-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "document_approved"
    assert result["document_result"]["status"] == "drafted"
    assert "review.model" in result["events"]
    assert indexed_runs == [(tmp_path, "run-controller-doc-001")]
    assert result["rag_index"] == {
        "status": "completed",
        "run_id": "run-controller-doc-001",
        "evidence_chunks": 3,
        "indexed_chunks": 9,
        "index_version": "v2",
    }
    assert [
        item["task_type"] for item in result["a2a_trace"] if item["kind"] == "outbound_task"
    ] == [
        "research.search",
        "document.write_section",
    ]
    assert [item["kind"] for item in result["a2a_trace"] if item["kind"] == "artifact_update"] == [
        "artifact_update",
        "artifact_update",
    ]
    assert (
        sum(event["event_type"] == "a2a.artifact_update" for event in result["execution_trace"])
        == 2
    )
    assert [event["sequence"] for event in result["execution_trace"]] == list(
        range(1, len(result["execution_trace"]) + 1)
    )
    trace_path = Path(result["trace_path"])
    assert trace_path.is_file()
    assert len(trace_path.read_text(encoding="utf-8").strip().splitlines()) == len(
        result["execution_trace"]
    )
    a2a_trace_path = Path(result["a2a_trace_path"])
    assert a2a_trace_path.is_file()
    assert len(a2a_trace_path.read_text(encoding="utf-8").strip().splitlines()) == len(
        result["a2a_trace"]
    )
    assert Path(result["document_task"]["research_report_ref"]).is_file()
    assert Path(result["document_task"]["outline_ref"]).is_file()


def test_graph_dispatches_supplement_then_revision_and_reapproves(monkeypatch, tmp_path):
    search_calls = []
    initial_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "summary": "初始研究摘要",
        "sources": [],
    }
    supplement_report = {
        "query": "智能体交互方式 实际采用率 数据",
        "provider": "tavily_mcp",
        "status": "completed",
        "summary": "补证研究摘要",
        "sources": [],
    }
    document_result = {
        "task_type": "document.write_section",
        "status": "drafted",
        "run_id": "run-supplement-001",
        "section_id": "current-status",
        "draft_version": "1",
        "draft_artifact": _persist_chapter_artifact(
            tmp_path,
            run_id="run-supplement-001",
            section_id="current-status",
            version="1",
            markdown="# 初稿\n\n初稿正文。\n",
        ),
        "evidence_bundle_ref": "data/runs/run-supplement-001/evidence_bundles/bundle.json",
        "citation_map_ref": "data/runs/run-supplement-001/chapters/current-status/v1.citations.json",
        "evidence_gaps": [
            {
                    "gap_id": "gap-adoption",
                    "kind": "content_issue",
                    "severity": "hard",
                "description": "缺少实际采用率数据",
                "suggested_queries": [
                    "智能体交互方式 实际采用率 数据",
                    "智能体交互方式 行业应用 案例",
                ],
            }
        ],
    }
    revised_document_result = {
        **document_result,
        "task_type": "document.revise_section",
        "status": "revised",
        "draft_version": "2",
        "draft_artifact": _persist_chapter_artifact(
            tmp_path,
            run_id="run-supplement-001",
            section_id="current-status",
            version="2",
            markdown="# 修订稿\n\n补入采用率资料后的正文。\n",
        ),
        "evidence_gaps": [],
    }

    def artifact_response(*, task_id, content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id="context-1",
                artifact=Artifact(
                    artifact_id=f"artifact-{task_id}",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    def persist_evidence_chunk(index: int) -> None:
        directory = tmp_path / "data" / "runs" / "run-supplement-001" / "evidence"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"chunk-{index}.json").write_text(
            json.dumps(
                {"chunk_id": f"chunk-{index}", "source_id": f"source-{index}"},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    async def fake_send_text_task(url, text):
        payload = json.loads(text)
        if url.endswith(":8002/"):
            result = (
                document_result
                if payload["task_type"] == "document.write_section"
                else revised_document_result
            )
            yield artifact_response(task_id="document-task", content=result)
            return
        search_calls.append(payload)
        persist_evidence_chunk(len(search_calls))
        report = initial_report if len(search_calls) == 1 else supplement_report
        yield artifact_response(task_id=f"search-task-{len(search_calls)}", content=report)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    indexed_runs = []

    def fake_evidence_indexer(root, run_id):
        indexed_runs.append(run_id)
        return {"run_id": run_id, "evidence_chunks": 2, "indexed_chunks": 2, "index_version": "v2"}

    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=fake_evidence_indexer,
    )
    result = asyncio.run(
        graph.ainvoke({"run_id": "run-supplement-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "document_approved"
    assert result["review_decision"]["decision"] == "approve"
    assert result["supplement_round"] == 1
    assert result["revision_round"] == 1
    assert len(search_calls) == 3  # initial search plus two bounded supplement queries
    assert all(call["max_results"] == 5 for call in search_calls[1:])
    assert indexed_runs == ["run-supplement-001", "run-supplement-001"]
    assert "supplement.search.completed" in result["events"]
    assert "document.revised" in result["events"]
    assert [
        json.loads(task)["task_type"]
        for task in result["a2a_tasks"]
    ] == [
        "research.search",
        "document.write_section",
        "research.search",
        "research.search",
        "document.revise_section",
    ]
    revision_task = json.loads(result["a2a_tasks"][-1])
    assert revision_task["base_draft"]["version"] == "1"
    assert revision_task["target_draft_version"] == "2"


def test_graph_does_not_review_or_approve_a_blocked_document_result(monkeypatch, tmp_path):
    search_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "sources": [],
    }
    document_result = {
        "task_type": "document.write_section",
        "status": "blocked",
        "run_id": "run-document-blocked-001",
        "section_id": "current-status",
        "draft_version": "1",
        "gaps": ["document_model_generation_failed"],
    }

    def artifact_response(*, task_id, content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id="context-1",
                artifact=Artifact(
                    artifact_id=f"artifact-{task_id}",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    async def fake_send_text_task(url, text):
        if url.endswith(":8001/"):
            yield artifact_response(task_id="search-task", content=search_report)
            return
        yield artifact_response(task_id="document-task", content=document_result)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=lambda root, run_id: {"run_id": run_id, "index_version": "v2"},
    )
    result = asyncio.run(
        graph.ainvoke({"run_id": "run-document-blocked-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "document_blocked"
    assert "review_decision" not in result
    assert "review.approve" not in result["events"]


def test_graph_stops_before_revision_when_supplement_adds_no_new_evidence(monkeypatch, tmp_path):
    """A repeated search result cannot masquerade as evidence-backed revision."""

    report = {"query": "智能体采用率", "provider": "tavily_mcp", "status": "completed"}
    calls = []
    document_result = {
        "task_type": "document.write_section",
        "status": "drafted",
        "run_id": "run-no-delta-001",
        "section_id": "current-status",
        "draft_version": "1",
        "draft_artifact": {
            "artifact_id": "artifact-1",
            "run_id": "run-no-delta-001",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
            "access_ref": "data/runs/run-no-delta-001/chapters/current-status/v1.md",
        },
        "evidence_bundle_ref": "data/runs/run-no-delta-001/evidence_bundles/bundle.json",
        "citation_map_ref": "data/runs/run-no-delta-001/chapters/current-status/v1.citations.json",
        "evidence_gaps": [
            {"gap_id": "gap-adoption", "kind": "content_issue", "severity": "hard", "description": "缺少采用率数据"}
        ],
    }
    document_result["draft_artifact"] = _persist_chapter_artifact(
        tmp_path,
        run_id="run-no-delta-001",
        section_id="current-status",
        version="1",
        markdown="# 当前发展现状\n\n正文。\n",
    )

    def artifact_response(content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="task-1",
                context_id="context-1",
                artifact=Artifact(
                    artifact_id="artifact-1",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    async def fake_send_text_task(url, text):
        payload = json.loads(text)
        calls.append(payload["task_type"])
        yield artifact_response(document_result if url.endswith(":8002/") else report)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    result = asyncio.run(
        build_graph(
            classify=lambda text: "report_request",
            search_agent_url="http://127.0.0.1:8001/",
            document_agent_url="http://127.0.0.1:8002/",
            root=tmp_path,
            evidence_indexer=lambda root, run_id: {"run_id": run_id, "index_version": "v2"},
        ).ainvoke({"run_id": "run-no-delta-001", "user_input": "分析智能体采用率"})
    )

    assert result["phase"] == "supplement_no_new_evidence"
    assert calls == ["research.search", "document.write_section", "research.search"]
    assert Path(result["supplement_audit_ref"]).is_file()
    review_audit = json.loads(Path(result["review_audit_ref"]).read_text(encoding="utf-8"))
    assert review_audit["decision"]["source"] == "deterministic_fallback"
    assert review_audit["decision"]["fallback_reason"] == "review_context_unavailable"


def test_graph_rejects_a_revision_with_the_same_content_hash(monkeypatch, tmp_path):
    report = {"query": "智能体结构", "provider": "tavily_mcp", "status": "completed"}
    document_calls = 0

    def artifact_response(content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="task-1",
                context_id="context-1",
                artifact=Artifact(
                    artifact_id="artifact-1",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    async def fake_send_text_task(url, text):
        nonlocal document_calls
        if url.endswith(":8001/"):
            yield artifact_response(report)
            return
        document_calls += 1
        payload = json.loads(text)
        version = str(document_calls)
        yield artifact_response(
            {
                "task_type": payload["task_type"],
                "status": "drafted" if document_calls == 1 else "revised",
                "run_id": "run-no-change-001",
                "section_id": "current-status",
                "draft_version": version,
                "draft_artifact": _persist_chapter_artifact(
                    tmp_path,
                    run_id="run-no-change-001",
                    section_id="current-status",
                    version=version,
                    markdown="# 智能体结构\n\n没有变化的章节正文。\n",
                ),
                "evidence_bundle_ref": "data/runs/run-no-change-001/evidence_bundles/bundle.json",
                "citation_map_ref": f"data/runs/run-no-change-001/chapters/current-status/v{version}.citations.json",
                "evidence_gaps": [
                    {
                        "gap_id": "gap-style",
                        "kind": "structure_issue",
                        "severity": "soft",
                        "description": "调整章节结构",
                    }
                ],
            }
        )

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    result = asyncio.run(
        build_graph(
            classify=lambda text: "report_request",
            search_agent_url="http://127.0.0.1:8001/",
            document_agent_url="http://127.0.0.1:8002/",
            root=tmp_path,
            evidence_indexer=lambda root, run_id: {"run_id": run_id, "index_version": "v2"},
        ).ainvoke({"run_id": "run-no-change-001", "user_input": "分析智能体结构"})
    )

    assert result["phase"] == "document_revision_no_material_change"
    assert result["revision_round"] == 1
    assert "document.revision.no_material_change" in result["events"]


def test_graph_stops_after_two_document_revision_rounds(monkeypatch, tmp_path):
    """A persistent structural issue cannot create an unbounded revision loop."""

    search_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "sources": [],
    }
    persistent_gap = {
        "gap_id": "gap-structure",
        "kind": "structure_issue",
        "severity": "soft",
        "description": "需要调整章节结构。",
        "suggested_queries": [],
    }
    document_calls = 0

    def artifact_response(*, task_id, content):
        return StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id="context-1",
                artifact=Artifact(
                    artifact_id=f"artifact-{task_id}",
                    name="result",
                    parts=[new_text_part(json.dumps(content, ensure_ascii=False))],
                ),
                last_chunk=True,
            )
        )

    async def fake_send_text_task(url, text):
        nonlocal document_calls
        if url.endswith(":8001/"):
            yield artifact_response(task_id="search-task", content=search_report)
            return
        document_calls += 1
        payload = json.loads(text)
        version = str(document_calls)
        yield artifact_response(
            task_id=f"document-task-{version}",
            content={
                "task_type": payload["task_type"],
                "status": "drafted" if document_calls == 1 else "revised",
                "run_id": "run-revision-limit-001",
                "section_id": "current-status",
                "draft_version": version,
                    "draft_artifact": _persist_chapter_artifact(
                        tmp_path,
                        run_id="run-revision-limit-001",
                        section_id="current-status",
                        version=version,
                        markdown=f"# 智能体交互方式\n\n第 {version} 版章节正文。\n",
                    ),
                "evidence_bundle_ref": "data/runs/run-revision-limit-001/evidence_bundles/bundle.json",
                "citation_map_ref": f"data/runs/run-revision-limit-001/chapters/current-status/v{version}.citations.json",
                "evidence_gaps": [persistent_gap],
            },
        )

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=lambda root, run_id: {"run_id": run_id, "index_version": "v2"},
    )

    result = asyncio.run(
        graph.ainvoke({"run_id": "run-revision-limit-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "review_limit_reached"
    assert result["revision_round"] == 2
    assert document_calls == 3  # initial draft plus exactly two revisions
    assert result["events"].count("document.revised") == 2
    assert "review.limit_reached" in result["events"]


def test_graph_blocks_document_dispatch_when_rag_indexing_fails(monkeypatch, tmp_path):
    search_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "summary": "搜索 Agent 已交付研究摘要。",
        "sources": [],
    }

    async def fake_send_text_task(url, text):
        assert url.endswith(":8001/")
        yield json.dumps(search_report, ensure_ascii=False)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=lambda root, run_id: (_ for _ in ()).throw(RagIndexError("unavailable")),
    )

    result = asyncio.run(
        graph.ainvoke({"run_id": "run-index-failure-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "rag_index_blocked"
    assert result["rag_index"] == {"status": "failed", "reason": "rag_indexing_failed"}
    assert "document_task" not in result
    assert "rag.index.failed" in result["events"]


def test_graph_exposes_no_indexable_evidence_reason(monkeypatch, tmp_path):
    search_report = {
        "query": "智能体交互方式",
        "provider": "tavily_mcp",
        "status": "completed",
        "summary": "搜索 Agent 已交付研究摘要。",
        "sources": [],
    }

    async def fake_send_text_task(url, text):
        yield json.dumps(search_report, ensure_ascii=False)

    monkeypatch.setattr("ai_status_report.harness.graph.send_text_task", fake_send_text_task)
    graph = build_graph(
        classify=lambda text: "report_request",
        search_agent_url="http://127.0.0.1:8001/",
        document_agent_url="http://127.0.0.1:8002/",
        root=tmp_path,
        evidence_indexer=lambda root, run_id: (_ for _ in ()).throw(RagIndexError("no_indexable_evidence")),
    )

    result = asyncio.run(
        graph.ainvoke({"run_id": "run-no-evidence-001", "user_input": "分析智能体交互方式"})
    )

    assert result["phase"] == "rag_index_blocked"
    assert result["rag_index"] == {"status": "failed", "reason": "no_indexable_evidence"}
    assert result["execution_trace"][-1]["refs"] == {"reason": "no_indexable_evidence"}
