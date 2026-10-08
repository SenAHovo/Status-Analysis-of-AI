"""Official MCP client wrapper for remote search tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPError(RuntimeError):
    """Safe MCP boundary error without credentials or response bodies."""


class TavilyRateLimitedError(MCPError):
    """A safe, retryable Tavily rate-limit signal from the MCP boundary."""

    def __init__(self, tool_name: str, retry_after_seconds: int | None = None) -> None:
        super().__init__("tavily_rate_limited")
        self.tool_name = tool_name
        self.retry_after_seconds = retry_after_seconds


class TavilyRequestPacer:
    """Serialize Tavily tool calls from one Search Agent process.

    Tavily applies a key-level request limit.  The pacer intentionally lives
    with the long-running Search Agent application, rather than in a graph
    run, so concurrent teaching runs cannot burst through the same API key.
    """

    def __init__(self, minimum_interval_seconds: float = 1.0) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("invalid_tavily_request_interval")
        self.minimum_interval_seconds = minimum_interval_seconds
        self._lock = asyncio.Lock()
        self._next_allowed_at = 0.0

    async def wait_turn(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = max(0.0, self._next_allowed_at - loop.time())
            if delay:
                await asyncio.sleep(delay)
            self._next_allowed_at = loop.time() + self.minimum_interval_seconds

    async def defer(self, seconds: int) -> None:
        """Prevent any same-process call before the provider-directed delay."""

        if seconds < 0:
            raise ValueError("invalid_tavily_retry_delay")
        async with self._lock:
            self._next_allowed_at = max(
                self._next_allowed_at,
                asyncio.get_running_loop().time() + seconds,
            )


AttemptHook = Callable[[], object]
RetryHook = Callable[[TavilyRateLimitedError, int, int], Awaitable[None] | None]


def _bounded_retry_after(value: object) -> int | None:
    """Accept only a small numeric retry delay from structured MCP content."""

    if isinstance(value, bool):
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None
    return seconds if 1 <= seconds <= 90 else None


def _rate_limit_metadata(value: object) -> tuple[bool, int | None]:
    """Read structured status fields without retaining provider error text."""

    if isinstance(value, dict):
        status = value.get("status")
        if status == 429 or status == "429":
            return True, _bounded_retry_after(
                value.get("retry_after", value.get("retry_after_seconds"))
            )
        for nested in value.values():
            limited, retry_after = _rate_limit_metadata(nested)
            if limited:
                return limited, retry_after
    elif isinstance(value, list):
        for nested in value:
            limited, retry_after = _rate_limit_metadata(nested)
            if limited:
                return limited, retry_after
    elif hasattr(value, "text"):
        text = value.text
        if isinstance(text, str):
            try:
                return _rate_limit_metadata(json.loads(text))
            except json.JSONDecodeError:
                pass
    return False, None


class TavilyMCPClient:
    """Discover and call Tavily tools through Streamable HTTP MCP."""

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: float = 60,
        *,
        pacer: TavilyRequestPacer | None = None,
        max_attempts: int = 2,
        fallback_retry_after_seconds: int = 60,
    ) -> None:
        if not api_key:
            raise MCPError("missing_tavily_api_key")
        if max_attempts < 1 or fallback_retry_after_seconds < 0:
            raise ValueError("invalid_tavily_retry_policy")
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.pacer = pacer or TavilyRequestPacer()
        self.max_attempts = max_attempts
        self.fallback_retry_after_seconds = fallback_retry_after_seconds

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with (  # noqa: SIM117
            httpx.AsyncClient(
            headers=headers,
            timeout=self.timeout,
            follow_redirects=False,
            trust_env=False,
            ) as http_client,
            streamable_http_client(self.url, http_client=http_client) as streams,
        ):
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session

    async def list_tools(self) -> list[str]:
        async with self.session() as session:
            result = await session.list_tools()
            return [tool.name for tool in result.tools]

    async def _call_once(self, name: str, arguments: dict[str, Any]) -> Any:
        async with self.session() as session:
            tools = await session.list_tools()
            available = {tool.name for tool in tools.tools}
            if name not in available:
                raise MCPError("tool_not_advertised")
            result = await session.call_tool(name, arguments=arguments)
            payload = result.structuredContent or result.content
            rate_limited, retry_after = _rate_limit_metadata(payload)
            if rate_limited:
                raise TavilyRateLimitedError(name, retry_after)
            if result.isError:
                raise MCPError("tool_call_failed")
            return payload

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        on_attempt: AttemptHook | None = None,
        on_rate_limit: RetryHook | None = None,
    ) -> Any:
        """Call one tool with bounded, provider-aware retry for HTTP 429.

        A retry never repeats an earlier successful search/extract operation.
        The fallback delay is used only when Remote MCP did not expose the
        upstream ``Retry-After`` value in its structured result.
        """

        for attempt in range(1, self.max_attempts + 1):
            await self.pacer.wait_turn()
            if on_attempt is not None:
                on_attempt()
            try:
                return await self._call_once(name, arguments)
            except TavilyRateLimitedError as exc:
                if attempt >= self.max_attempts:
                    raise
                delay = exc.retry_after_seconds or self.fallback_retry_after_seconds
                await self.pacer.defer(delay)
                if on_rate_limit is not None:
                    callback_result = on_rate_limit(exc, attempt + 1, delay)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        raise MCPError("tavily_retry_exhausted")
