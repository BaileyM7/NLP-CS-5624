"""MCP server exposing `search` and `get_article`, routed per episode."""
from __future__ import annotations

from typing import Any

from mcp import MCPError
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, TextContent
from starlette.requests import Request
from starlette.responses import JSONResponse

from .registry import EpisodeError, EpisodeRegistry

# Keys the client sends as HTTP headers (or as request _meta in in-memory tests)
EPISODE_KEY = "x-episode-id"
SCENARIO_KEY = "x-scenario-id"
SEED_KEY = "x-fault-seed"

INVALID_PARAMS = -32602  # JSON-RPC error code for bad request parameters

SEARCH_DESCRIPTION = """Search the article collection.

Returns JSON: {"query": str, "hits": [{"title": str, "page": int, "snippet": str}]}.
Hits are ranked best first. Use get_article with a hit's title and page to read it."""

GET_ARTICLE_DESCRIPTION = """Read one page of an article by its title.

Returns JSON: {"title": str, "page": int, "total_pages": int, "sentences": [str],
"last_updated": "YYYY-MM-DD", "source_id": str}.
If the title or page does not exist, returns {"error": "not_found", "title": str, "page": int | null, "message": str}."""


def read_key(ctx: Context, key: str) -> str | None:
    """Read one episode key from the HTTP headers, falling back to the request's _meta."""
    headers = ctx.headers
    if headers is not None and key in headers:
        return headers[key]
    meta = ctx.request_context.meta or {}
    value = meta.get(key)
    return None if value is None else str(value)


def episode_info(ctx: Context) -> tuple[str | None, str | None, int]:
    """Return (episode ID, scenario ID, fault seed) for the current request."""
    seed_text = read_key(ctx, SEED_KEY) or "0"
    try:
        seed = int(seed_text)
    except ValueError:
        raise EpisodeError(f"fault seed must be an integer, got {seed_text!r}") from None
    return read_key(ctx, EPISODE_KEY), read_key(ctx, SCENARIO_KEY), seed


def run_tool(registry: EpisodeRegistry, ctx: Context, tool: str, args: dict[str, Any]) -> CallToolResult:
    """Send one tool call to its episode's backend and wrap the JSON text for MCP."""
    try:
        backend = registry.get_or_create(*episode_info(ctx))
    except EpisodeError as error:
        # Harness misconfiguration: fail the request instead of showing the agent an error
        raise MCPError(INVALID_PARAMS, str(error)) from None

    response = backend.call(tool, args)
    return CallToolResult(content=[TextContent(text=response.text)], is_error=response.is_error)


def build_server(registry: EpisodeRegistry) -> MCPServer:
    """Create the MCP server with the two tools and a /health route."""
    server = MCPServer(
        "toolharness",
        instructions="Article search tools. Use search to find articles, then get_article to read them.",
    )

    @server.tool(description=SEARCH_DESCRIPTION, structured_output=False)
    async def search(query: str, ctx: Context, k: int = 5) -> CallToolResult:
        return run_tool(registry, ctx, "search", {"query": query, "k": k})

    @server.tool(description=GET_ARTICLE_DESCRIPTION, structured_output=False)
    async def get_article(title: str, ctx: Context, page: int = 1) -> CallToolResult:
        return run_tool(registry, ctx, "get_article", {"title": title, "page": page})

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "scenarios": len(registry.scenarios), "episodes": len(registry.episodes)}
        )

    return server
