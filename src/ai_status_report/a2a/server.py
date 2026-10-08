"""A2A HTTP server components for the two specialist agents.

This module uses the official ``a2a-sdk`` request handler and route builders;
it does not replace A2A messages with an internal Python callback.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from a2a.helpers import new_task_from_user_message, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, Role, TaskState
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
from starlette.applications import Starlette


class SpecialistExecutor(AgentExecutor):
    """Small protocol-level executor used until specialist logic is connected."""

    def __init__(
        self,
        agent_name: str,
        search: Callable[..., Awaitable[object]] | None = None,
        *,
        progress_updates: bool = False,
    ):
        self.agent_name = agent_name
        self.search = search
        self.progress_updates = progress_updates

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # A2A v1 requires the initial Task before any status or artifact event.
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        if self.search and self.progress_updates:
            async def progress(code: str) -> None:
                await updater.update_status(
                    TaskState.TASK_STATE_WORKING,
                    message=new_text_message(code, role=Role.Value("ROLE_AGENT")),
                )

            result = await self.search(_parse_search_task(context.get_user_input()), progress=progress)
        elif self.search:
            result = await self.search(_parse_search_task(context.get_user_input()))
        else:
            result = f"{self.agent_name} received the A2A task."
        await updater.add_artifact(
            [new_text_part(str(result))],
            name="acknowledgement",
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()


def _parse_search_task(text: str) -> dict[str, object]:
    try:
        task = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_search_task") from exc
    if not isinstance(task, dict) or task.get("task_type") != "research.search":
        raise ValueError("invalid_search_task")
    query = task.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("invalid_search_query")
    return task


def build_agent_card(*, name: str, description: str, url: str) -> AgentCard:
    return AgentCard(
        name=name,
        description=description,
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        # The teaching demo exposes TaskStatusUpdateEvent and
        # TaskArtifactUpdateEvent through the official A2A SSE binding.
        # Source: https://a2a-protocol.org/latest/topics/streaming-and-async/
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", protocol_version="1.0", url=url)
        ],
        skills=[],
    )


def build_a2a_app(*, card: AgentCard, executor: AgentExecutor) -> Starlette:
    """Expose the official Agent Card and JSON-RPC A2A endpoint."""
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    routes = create_agent_card_routes(
        card,
        card_url=AGENT_CARD_WELL_KNOWN_PATH,
    )
    routes.extend(create_jsonrpc_routes(handler, rpc_url="/"))
    return Starlette(routes=routes)
