"""Official A2A SDK client wrapper for local specialist agents."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx
from a2a.client import Client, ClientCallContext, ClientConfig, ClientFactory
from a2a.helpers import get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

_PROGRESS_CODE = re.compile(r"^tavily_rate_limited_wait:(?:[1-9]|[1-8][0-9]|90):2$")


async def send_text_task(base_url: str, text: str) -> AsyncIterator[object]:
    """Resolve an Agent Card and send a real A2A message over HTTP JSON-RPC."""
    # Local A2A services must bypass inherited HTTP proxy settings. Without
    # this, a machine-level proxy can turn 127.0.0.1 AgentCard requests into
    # 502 responses before the request reaches Uvicorn.
    # A specialist may perform a paid web search before it emits the final
    # Artifact. Keep the protocol client alive for the bounded provider
    # timeout instead of relying on the SDK/httpx default timeout.
    config = ClientConfig(
        httpx_client=httpx.AsyncClient(
            trust_env=False,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )
    )
    client: Client | None = None
    try:
        client = await ClientFactory(config).create_from_url(base_url)
        # The official SDK resolves the Agent Card before selecting JSON-RPC/HTTP.
        # Source: https://a2a-protocol.org/latest/sdk/python/api/a2a.client.html
        message = new_text_message(text, role=Role.Value("ROLE_USER"))
        request = SendMessageRequest(message=message)
        context = ClientCallContext(
            service_parameters={"A2A-Version": "1.0"},
            timeout=120.0,
        )
        async for response in client.send_message(request, context=context):
            yield response
    finally:
        if client is not None:
            await client.close()


def response_text(response: object) -> str:
    """Extract text from an SDK StreamResponse while keeping test doubles usable."""
    if isinstance(response, str):
        return response
    return get_stream_response_text(response)


def describe_response(response: object) -> dict[str, object]:
    """Return a compact, human-readable trace entry for one A2A stream event."""

    if isinstance(response, str):
        return {"kind": "text", "text": response}
    payload_kind = response.WhichOneof("payload")
    if payload_kind == "task":
        return {
            "kind": "task",
            "task_id": response.task.id,
            "context_id": response.task.context_id,
            "state": TaskState.Name(response.task.status.state),
            "artifact_count": len(response.task.artifacts),
        }
    if payload_kind == "status_update":
        text = response_text(response)
        return {
            "kind": "status_update",
            "task_id": response.status_update.task_id,
            "context_id": response.status_update.context_id,
            "state": TaskState.Name(response.status_update.status.state),
            "text": text,
            "progress_code": text if _PROGRESS_CODE.fullmatch(text) else "",
        }
    if payload_kind == "artifact_update":
        artifact = response.artifact_update.artifact
        return {
            "kind": "artifact_update",
            "task_id": response.artifact_update.task_id,
            "context_id": response.artifact_update.context_id,
            "artifact_id": artifact.artifact_id,
            "name": artifact.name,
            "append": response.artifact_update.append,
            "last_chunk": response.artifact_update.last_chunk,
            "text": response_text(response),
        }
    if payload_kind == "message":
        return {"kind": "message", "text": response_text(response)}
    return {"kind": "unknown"}
