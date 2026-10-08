import asyncio
from types import SimpleNamespace

from a2a.helpers import new_text_message
from a2a.server.events import EventQueueLegacy
from a2a.types import (
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from google.protobuf.json_format import MessageToDict
from starlette.testclient import TestClient

from ai_status_report.a2a.server import (
    SpecialistExecutor,
    build_a2a_app,
    build_agent_card,
)


def test_a2a_agent_card_and_jsonrpc_endpoint_are_real_sdk_routes():
    card = build_agent_card(
        name="Network Search Agent",
        description="Searches research sources.",
        url="http://testserver/",
    )
    app = build_a2a_app(card=card, executor=SpecialistExecutor(card.name))

    with TestClient(app) as client:
        card_response = client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        assert card_response.json()["name"] == "Network Search Agent"
        assert card_response.json()["capabilities"]["streaming"] is True


def test_a2a_streaming_endpoint_returns_sse_events_in_order():
    card = build_agent_card(
        name="Network Search Agent",
        description="Searches research sources.",
        url="http://testserver/",
    )
    app = build_a2a_app(card=card, executor=SpecialistExecutor(card.name))
    message = MessageToDict(
        new_text_message("test", role=Role.Value("ROLE_USER")),
        preserving_proto_field_name=True,
    )
    payload = {
        "jsonrpc": "2.0",
        "id": "stream-1",
        "method": "SendStreamingMessage",
        "params": {"message": message},
    }

    with TestClient(app) as client:
        response = client.post(
            "/",
            json=payload,
            headers={"A2A-Version": "1.0"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert body.index('"task"') < body.index('"statusUpdate"')
    assert body.index('"statusUpdate"') < body.index('"artifactUpdate"')


def test_specialist_executor_emits_task_lifecycle_in_order():
    async def collect_events():
        message = new_text_message("test", role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
        )
        queue = EventQueueLegacy()
        await SpecialistExecutor("Network Search Agent").execute(context, queue)
        return [await queue.dequeue_event() for _ in range(4)]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], Task)
    assert isinstance(events[1], TaskStatusUpdateEvent)
    assert isinstance(events[2], TaskArtifactUpdateEvent)
    assert isinstance(events[3], TaskStatusUpdateEvent)


def test_specialist_executor_does_not_reemit_existing_task():
    async def collect_events():
        message = new_text_message("test", role=Role.Value("ROLE_USER"))
        existing_task = Task(
            id="task-1",
            context_id="context-1",
            status=TaskStatus(state=TaskState.Value("TASK_STATE_WORKING")),
        )
        context = SimpleNamespace(
            current_task=existing_task,
            message=message,
            task_id="task-1",
            context_id="context-1",
        )
        queue = EventQueueLegacy()
        await SpecialistExecutor("Network Search Agent").execute(context, queue)
        return [await queue.dequeue_event() for _ in range(3)]

    events = asyncio.run(collect_events())

    assert isinstance(events[0], TaskStatusUpdateEvent)
    assert isinstance(events[1], TaskArtifactUpdateEvent)
    assert isinstance(events[2], TaskStatusUpdateEvent)


def test_specialist_executor_emits_controlled_progress_for_search_agent():
    async def search(_task, *, progress):
        await progress("tavily_rate_limited_wait:12:2")
        return "done"

    async def collect_events():
        text = '{"task_type":"research.search","query":"test"}'
        message = new_text_message(text, role=Role.Value("ROLE_USER"))
        context = SimpleNamespace(
            current_task=None,
            message=message,
            task_id="task-1",
            context_id="context-1",
            get_user_input=lambda: text,
        )
        queue = EventQueueLegacy()
        await SpecialistExecutor("Network Search Agent", search, progress_updates=True).execute(context, queue)
        return [await queue.dequeue_event() for _ in range(5)]

    events = asyncio.run(collect_events())

    assert isinstance(events[2], TaskStatusUpdateEvent)
    assert "tavily_rate_limited_wait:12:2" in str(events[2])
