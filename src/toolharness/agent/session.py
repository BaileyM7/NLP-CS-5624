"""Tool sessions: how the agent runner reaches the MCP tool server for one episode."""
from __future__ import annotations

from contextlib import AsyncExitStack, contextmanager
from typing import Any, Iterator, Protocol

import httpx2
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import MCPServer

from ..llm.client import tools_from_mcp


class ToolServerError(Exception):
    """The tool server couldn't serve a call (it's down, or the episode IDs were rejected). Not a tool result."""


SERVER_ERRORS = (MCPError, httpx2.HTTPError, OSError)


def as_server_error(error: Exception) -> ToolServerError | None:
    """A ToolServerError for an MCP or connection failure, also when anyio wrapped it in an ExceptionGroup."""
    while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    if isinstance(error, SERVER_ERRORS):
        return ToolServerError(f"{type(error).__name__}: {error}")
    if isinstance(error, ExceptionGroup) and error.subgroup(SERVER_ERRORS) is not None:
        return ToolServerError(str(error))
    return None


@contextmanager
def server_errors() -> Iterator[None]:
    """Re-raise MCP and connection failures as ToolServerError; anything else passes through."""
    try:
        yield
    except Exception as error:
        wrapped = as_server_error(error)
        if wrapped is None:
            raise
        raise wrapped from error


class ToolSession(Protocol):
    async def list_tools(self) -> list[dict]: ...

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]: ...


def episode_headers(episode_id: str, scenario_id: str, seed: int) -> dict[str, str]:
    """The HTTP headers that tell the tool server which episode a call belongs to."""
    return {"X-Episode-Id": episode_id, "X-Scenario-Id": scenario_id, "X-Fault-Seed": str(seed)}


class _McpSession:
    """Shared tool-calling code; subclasses open the MCP client and may add request _meta."""

    def __init__(self):
        self.stack = AsyncExitStack()
        self.client: Client | None = None
        self.meta: dict[str, str] | None = None

    async def _open(self) -> Client:
        raise NotImplementedError

    async def __aenter__(self):
        try:
            with server_errors():
                self.client = await self._open()
        except ToolServerError:
            await self.stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc) -> None:
        with server_errors():
            await self.stack.aclose()

    async def list_tools(self) -> list[dict]:
        """The server's tools as OpenAI function tools."""
        with server_errors():
            result = await self.client.list_tools()
        return tools_from_mcp(result.tools)

    async def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run one tool call; returns the response text and whether it is an error result."""
        with server_errors():
            result = await self.client.call_tool(name, args, meta=self.meta)
        text = result.content[0].text if result.content else ""
        return text, result.is_error


class HttpToolSession(_McpSession):
    """One episode's session with the running tool server, identified by HTTP headers."""

    def __init__(self, url: str, episode_id: str, scenario_id: str, seed: int):
        super().__init__()
        self.url = url
        self.headers = episode_headers(episode_id, scenario_id, seed)

    async def _open(self) -> Client:
        http = await self.stack.enter_async_context(
            httpx2.AsyncClient(headers=self.headers, timeout=httpx2.Timeout(30, read=300))
        )
        return await self.stack.enter_async_context(Client(streamable_http_client(self.url, http_client=http)))


class InMemoryToolSession(_McpSession):
    """One episode's session with an in-process server; the episode keys travel as request _meta (for tests)."""

    def __init__(self, server: MCPServer, episode_id: str, scenario_id: str, seed: int):
        super().__init__()
        self.server = server
        self.meta = {key.lower(): value for key, value in episode_headers(episode_id, scenario_id, seed).items()}

    async def _open(self) -> Client:
        return await self.stack.enter_async_context(Client(self.server))
