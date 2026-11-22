"""Tests for the MCP server: in-memory client tests plus one real HTTP smoke test."""
import asyncio
import json
from pathlib import Path

import httpx2
import pytest
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client

from toolharness.data.scenario import FaultType, read_jsonl
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


@pytest.fixture
def scenarios():
    return [s for s in read_jsonl(SAMPLE_FILE) if s.fault_type == FaultType.NONE]


@pytest.fixture
def registry(scenarios):
    return EpisodeRegistry(scenarios)


@pytest.fixture
def server(registry):
    return build_server(registry)


def episode_meta(episode_id, scenario_id, seed=None):
    """Request _meta carrying the episode keys (in-memory clients have no HTTP headers)."""
    meta = {"x-episode-id": episode_id, "x-scenario-id": scenario_id}
    if seed is not None:
        meta["x-fault-seed"] = seed
    return meta


# In-memory client

@pytest.mark.anyio
async def test_lists_both_tools_as_plain_text_tools(server):
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    assert {tool.name for tool in tools} == {"search", "get_article"}
    for tool in tools:
        assert tool.output_schema is None


@pytest.mark.anyio
async def test_search_and_get_article_round_trip(server, scenarios):
    scenario = scenarios[0]
    meta = episode_meta("e1", scenario.scenario_id)
    async with Client(server) as client:
        found = await client.call_tool("search", {"query": scenario.question, "k": 3}, meta=meta)
        first_hit = json.loads(found.content[0].text)["hits"][0]
        page = await client.call_tool("get_article", {"title": first_hit["title"], "page": first_hit["page"]}, meta=meta)

    assert not found.is_error
    article = json.loads(page.content[0].text)
    assert article["title"] == first_hit["title"]
    assert article["total_pages"] == 1


@pytest.mark.anyio
async def test_unknown_title_is_a_normal_result(server, scenarios):
    meta = episode_meta("e1", scenarios[0].scenario_id)
    async with Client(server) as client:
        result = await client.call_tool("get_article", {"title": "No Such Article"}, meta=meta)
    assert not result.is_error
    assert json.loads(result.content[0].text)["error"] == "not_found"


@pytest.mark.anyio
async def test_episode_is_reused_across_calls(server, registry, scenarios):
    meta = episode_meta("e1", scenarios[0].scenario_id)
    async with Client(server) as client:
        await client.call_tool("search", {"query": "film"}, meta=meta)
        await client.call_tool("search", {"query": "music"}, meta=meta)
    assert list(registry.episodes) == ["e1"]


@pytest.mark.anyio
async def test_missing_episode_id_fails_the_request(server, scenarios):
    async with Client(server) as client:
        with pytest.raises(MCPError, match="missing episode ID"):
            await client.call_tool("search", {"query": "film"}, meta={"x-scenario-id": scenarios[0].scenario_id})


@pytest.mark.anyio
async def test_unknown_scenario_fails_the_request(server):
    async with Client(server) as client:
        with pytest.raises(MCPError, match="unknown scenario"):
            await client.call_tool("search", {"query": "film"}, meta=episode_meta("e1", "no-such-scenario"))


@pytest.mark.anyio
async def test_episode_cannot_switch_scenario(server, scenarios):
    async with Client(server) as client:
        await client.call_tool("search", {"query": "film"}, meta=episode_meta("e1", scenarios[0].scenario_id))
        with pytest.raises(MCPError, match="was started with scenario"):
            await client.call_tool("search", {"query": "film"}, meta=episode_meta("e1", scenarios[1].scenario_id))


@pytest.mark.anyio
async def test_non_integer_seed_fails_the_request(server, scenarios):
    async with Client(server) as client:
        with pytest.raises(MCPError, match="fault seed must be an integer"):
            await client.call_tool("search", {"query": "film"}, meta=episode_meta("e1", scenarios[0].scenario_id, "abc"))


def test_duplicate_scenario_ids_are_rejected(scenarios):
    with pytest.raises(ValueError, match="duplicate"):
        EpisodeRegistry([scenarios[0], scenarios[0]])


# Real HTTP server (the http_server fixture is in conftest.py)

async def call_over_http(base_url, episode_id, scenario_id, tool, args):
    """One MCP tool call over HTTP, with the episode keys sent as headers."""
    headers = {"X-Episode-Id": episode_id, "X-Scenario-Id": scenario_id}
    async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=60)) as http:
        async with Client(streamable_http_client(base_url + "/mcp", http_client=http)) as client:
            return await client.call_tool(tool, args)


@pytest.mark.anyio
async def test_http_serves_two_episodes_concurrently(http_server, scenarios, tmp_path):
    first, second = scenarios[0], scenarios[1]
    results = await asyncio.gather(
        call_over_http(http_server, "ep-1", first.scenario_id, "search", {"query": first.question}),
        call_over_http(http_server, "ep-2", second.scenario_id, "search", {"query": second.question}),
    )

    # Each episode only sees its own scenario's articles
    for result, scenario in zip(results, [first, second]):
        assert not result.is_error
        own_titles = {doc.title for doc in scenario.context}
        hit_titles = {hit["title"] for hit in json.loads(result.content[0].text)["hits"]}
        assert hit_titles and hit_titles <= own_titles

    health = httpx2.get(http_server + "/health", timeout=5).json()
    assert health["episodes"] == 2

    # The server logged one ground-truth row per episode
    log_lines = (tmp_path / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["episode_id"] for line in log_lines) == ["ep-1", "ep-2"]
