"""Document Agent A2A lifecycle tests without model or RAG calls."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from a2a.helpers import new_text_message
from a2a.server.events import EventQueueLegacy
from a2a.types import Role, Task, TaskArtifactUpdateEvent, TaskState, TaskStatusUpdateEvent
from google.protobuf.json_format import MessageToDict
from starlette.testclient import TestClient

from ai_status_report.a2a.apps import document_generation_app
from ai_status_report.agents.document import (
    DocumentTaskExecutor,
    GeneratedMarkdown,
    RetrievedEvidenceBundle,
)
from ai_status_report.documents.markdown import parse_section_markdown
from ai_status_report.rag.bundle import persist_evidence_bundle
from ai_status_report.rag.chroma import RagIndexError
from ai_status_report.schemas.evidence import EvidenceBundle, Excerpt


def document_task_json() -> str:
    return json.dumps(
        {
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
        },
        ensure_ascii=False,
    )


def persisted_evidence_retriever(root: Path):
    def retrieve(task):
        bundle = EvidenceBundle(
            retrieval_id=f"retrieval-{task.run_id}-{task.section_id}",
            run_id=task.run_id,
            section_id=task.section_id,
            queries=task.retrieval_queries or [task.section_title],
            chunk_refs=["retrieval-chunk-1"],
            excerpts=[
                Excerpt(
                    chunk_id="retrieval-chunk-1",
                    source_id="source-1",
                    text="A persistent evidence excerpt.",
                    token_estimate=6,
                    url="https://example.com/evidence",
                    retrieval_chunk_id="retrieval-chunk-1",
                    parent_chunk_id="evidence-chunk-1",
                )
            ],
            source_metadata={"source-1": "https://example.com/evidence"},
            budget_used=6,
        )
        return RetrievedEvidenceBundle(
            bundle=bundle,
            path=persist_evidence_bundle(root, bundle),
        )

    return retrieve


def test_document_executor_emits_persisted_evidence_result_lifecycle(tmp_path):
    async def collect_events():
        message = new_text_message(document_task_json(), role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: document_task_json(),
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor(
            evidence_retriever=persisted_evidence_retriever(tmp_path)
        ).execute(context, queue)
        return [await queue.dequeue_event() for _ in range(7)]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], Task)
    assert all(isinstance(event, TaskStatusUpdateEvent) for event in events[1:5])
    assert isinstance(events[5], TaskArtifactUpdateEvent)
    assert isinstance(events[6], TaskStatusUpdateEvent)
    artifact = MessageToDict(events[5], preserving_proto_field_name=True)["artifact"]
    result = json.loads(artifact["parts"][0]["text"])
    assert result["status"] == "blocked"
    assert result["gaps"] == ["document_execution_not_enabled"]
    assert Path(result["evidence_bundle_ref"]).is_file()
    assert result["used_source_ids"] == ["source-1"]
    assert result["used_chunk_ids"] == ["retrieval-chunk-1"]


def test_document_executor_persists_markdown_and_returns_drafted_artifact(tmp_path):
    def writer(task, retrieved):
        markdown = """# 智能体交互方式

## 本章摘要

说明主要交互模式。

## 正文

证据支持的结论。[1]

## 证据缺口

- [hard] 缺少跨行业量化效果数据。
"""
        return GeneratedMarkdown(markdown, parse_section_markdown(markdown), "skill-hash", ())

    async def collect_events():
        message = new_text_message(document_task_json(), role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: document_task_json(),
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor(
            root=tmp_path,
            evidence_retriever=persisted_evidence_retriever(tmp_path),
            section_writer=writer,
        ).execute(context, queue)
        return [await queue.dequeue_event() for _ in range(14)]

    events = asyncio.run(collect_events())
    artifact = MessageToDict(events[12], preserving_proto_field_name=True)["artifact"]
    result = json.loads(artifact["parts"][0]["text"])

    assert result["status"] == "drafted"
    assert result["section_summary"] == "说明主要交互模式。"
    assert result["used_source_ids"] == ["source-1"]
    assert result["used_chunk_ids"] == ["retrieval-chunk-1"]
    assert Path(result["draft_artifact"]["access_ref"]).is_file()
    assert Path(result["citation_map_ref"]).is_file()
    markdown = Path(result["draft_artifact"]["access_ref"]).read_text(encoding="utf-8")
    assert "证据支持的结论。[1]" in markdown
    assert "## 参考来源" in markdown
    assert "## 证据缺口" not in markdown
    assert result["evidence_gaps"] == [
        {
                "schema_version": "1",
                "gap_id": result["evidence_gaps"][0]["gap_id"],
                "kind": "evidence_limitation",
                "severity": "hard",
            "description": "缺少跨行业量化效果数据。",
            "preferred_source_types": [],
            "suggested_queries": [],
            "resolution_status": "open",
            "attempt_count": 0,
        }
    ]
    status_texts = [
        MessageToDict(event, preserving_proto_field_name=True)["status"]["message"]["parts"][0][
            "text"
        ]
        for event in events[1:12]
    ]
    assert status_texts == [
        "document.task.received",
        "document.task.validated",
        "document.evidence.retrieving",
        "document.evidence.persisted",
        "document.skill.loaded",
        "document.context.assembled",
        "document.model.requesting",
        "document.model.responded",
        "document.markdown.parsed",
        "document.evidence.validated",
        "document.draft.persisted",
    ]
    assert events[13].status.state == TaskState.Value("TASK_STATE_COMPLETED")


def test_document_executor_persists_revised_markdown_and_returns_revision_artifact(tmp_path):
    base_path = tmp_path / "data" / "runs" / "run-doc-001" / "chapters" / "s1" / "v1.md"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("# 智能体交互方式\n\n旧版正文。\n", encoding="utf-8")
    payload = json.loads(document_task_json()) | {
        "task_type": "document.revise_section",
        "base_draft": {
            "artifact_id": "artifact-chapter-v1",
            "run_id": "run-doc-001",
            "version": "1",
            "kind": "chapter_markdown",
            "mime_type": "text/markdown",
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
    }
    payload.pop("research_summary")
    payload.pop("section_goal")
    revision_task = json.dumps(payload, ensure_ascii=False)

    def reviser(task, retrieved):
        assert task.base_draft.access_ref == str(base_path)
        assert task.target_draft_version == "2"
        markdown = """# 智能体交互方式

## 本章摘要

补充引用后的修订摘要。

## 正文

修订后的可追溯结论。[1]

## 证据缺口

无
"""
        return GeneratedMarkdown(markdown, parse_section_markdown(markdown), "revision-skill", ())

    async def collect_events():
        message = new_text_message(revision_task, role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-2",
            context_id="context-2",
            get_user_input=lambda: revision_task,
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor(
            root=tmp_path,
            evidence_retriever=persisted_evidence_retriever(tmp_path),
            section_reviser=reviser,
        ).execute(context, queue)
        return [await queue.dequeue_event() for _ in range(14)]

    events = asyncio.run(collect_events())
    artifact = MessageToDict(events[12], preserving_proto_field_name=True)["artifact"]
    result = json.loads(artifact["parts"][0]["text"])

    assert result["task_type"] == "document.revise_section"
    assert result["status"] == "revised"
    assert result["draft_version"] == "2"
    assert result["quality_flags"][-1] == "revision_persisted"
    assert Path(result["draft_artifact"]["access_ref"]).read_text(encoding="utf-8").startswith(
        "# 智能体交互方式"
    )
    status_texts = [
        MessageToDict(event, preserving_proto_field_name=True)["status"]["message"]["parts"][0][
            "text"
        ]
        for event in events[1:12]
    ]
    assert status_texts[-1] == "document.revision.persisted"
    assert events[13].status.state == TaskState.Value("TASK_STATE_COMPLETED")


def test_document_executor_fails_invalid_task_without_an_artifact():
    async def collect_events():
        message = new_text_message("{}", role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: "{}",
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor().execute(context, queue)
        return [await queue.dequeue_event() for _ in range(3)]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], Task)
    assert all(isinstance(event, TaskStatusUpdateEvent) for event in events[1:])
    final = MessageToDict(events[2], preserving_proto_field_name=True)
    assert final["status"]["state"] == "TASK_STATE_FAILED"


def test_document_executor_rejects_encoding_corruption_before_evidence_retrieval():
    calls = []
    payload = json.loads(document_task_json())
    payload["retrieval_queries"] = ["????????"]
    corrupted_task = json.dumps(payload)

    async def collect_events():
        message = new_text_message(corrupted_task, role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: corrupted_task,
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor(evidence_retriever=lambda task: calls.append(task)).execute(
            context, queue
        )
        return [await queue.dequeue_event() for _ in range(3)]

    events = asyncio.run(collect_events())

    assert calls == []
    final = MessageToDict(events[2], preserving_proto_field_name=True)
    assert final["status"]["state"] == "TASK_STATE_FAILED"
    assert final["status"]["message"]["parts"][0]["text"] == "input_encoding_corrupted"


def test_document_executor_returns_a_structured_result_when_evidence_retrieval_fails():
    def retrieval_fails(task):
        raise RagIndexError("index_unavailable")

    async def collect_events():
        message = new_text_message(document_task_json(), role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: document_task_json(),
        )
        queue = EventQueueLegacy()
        await DocumentTaskExecutor(evidence_retriever=retrieval_fails).execute(context, queue)
        return [await queue.dequeue_event() for _ in range(6)]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], Task)
    assert all(isinstance(event, TaskStatusUpdateEvent) for event in events[1:4])
    artifact = MessageToDict(events[4], preserving_proto_field_name=True)["artifact"]
    result = json.loads(artifact["parts"][0]["text"])
    assert result["status"] == "blocked"
    assert result["evidence_bundle_ref"] == ""
    assert result["gaps"] == ["evidence_retrieval_failed", "document_execution_not_enabled"]
    terminal = MessageToDict(events[5], preserving_proto_field_name=True)
    assert terminal["status"]["state"] == "TASK_STATE_COMPLETED"


def test_document_agent_http_endpoint_streams_persisted_evidence_result(tmp_path):
    message = MessageToDict(
        new_text_message(document_task_json(), role=Role.Value("ROLE_USER")),
        preserving_proto_field_name=True,
    )
    payload = {
        "jsonrpc": "2.0",
        "id": "document-stream-1",
        "method": "SendStreamingMessage",
        "params": {"message": message},
    }

    with TestClient(
        document_generation_app(evidence_retriever=persisted_evidence_retriever(tmp_path))
    ) as client:
        response = client.post(
            "/",
            json=payload,
            headers={"A2A-Version": "1.0"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"task"' in response.text
    assert '"artifactUpdate"' in response.text
    assert "evidence_bundle_persisted" in response.text


def test_document_agent_http_endpoint_returns_safe_code_for_unhashable_task_type():
    message = MessageToDict(
        new_text_message('{"task_type": []}', role=Role.Value("ROLE_USER")),
        preserving_proto_field_name=True,
    )
    payload = {
        "jsonrpc": "2.0",
        "id": "document-stream-invalid-type",
        "method": "SendStreamingMessage",
        "params": {"message": message},
    }

    with TestClient(document_generation_app()) as client:
        response = client.post(
            "/",
            json=payload,
            headers={"A2A-Version": "1.0"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "invalid_document_task" in response.text
    assert "unhashable type" not in response.text
    assert "TASK_STATE_FAILED" in response.text
