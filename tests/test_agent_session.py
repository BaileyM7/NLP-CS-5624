"""Tests for the agent's tool sessions (in-memory MCP server; the HTTP session is covered in test_agent_http.py)."""
import json

import pytest

from toolharness.agent.session import InMemoryToolSession, ToolServerError, episode_headers
from toolharness.data.scenario import FaultType
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server


@pytest.fixture
def registry(sample_scenarios):
    return EpisodeRegistry(sample_scenarios)


@pytest.fixture
def scenario(sample_scenarios):
    return next(s for s in sample_scenarios if s.fault_type == FaultType.NONE)


@pytest.mark.anyio
async def test_in_memory_session_lists_tools_and_calls_them(registry, scenario):
    async with InMemoryToolSession(build_server(registry), "e1", scenario.scenario_id, 0) as session:
        tools = await session.list_tools()
        text, is_error = await session.call("search", {"query": scenario.question})

    assert {tool["function"]["name"] for tool in tools} == {"search", "get_article"}
    assert not is_error
    assert json.loads(text)["hits"]
    assert "e1" in registry.episodes


@pytest.mark.anyio
async def test_bad_arguments_come_back_as_a_tool_error(registry, scenario):
    async with InMemoryToolSession(build_server(registry), "e1", scenario.scenario_id, 0) as session:
        text, is_error = await session.call("get_article", {"title": "X", "page": "two"})
        assert is_error and "validation error" in text
        text, is_error = await session.call("get_article", {})
        assert is_error


@pytest.mark.anyio
async def test_unknown_scenario_raises_tool_server_error(registry):
    async with InMemoryToolSession(build_server(registry), "e1", "no-such-scenario", 0) as session:
        with pytest.raises(ToolServerError, match="unknown scenario"):
            await session.call("search", {"query": "x"})


def test_episode_headers():
    assert episode_headers("e1", "s1", 3) == {"X-Episode-Id": "e1", "X-Scenario-Id": "s1", "X-Fault-Seed": "3"}
