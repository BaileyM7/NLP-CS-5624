"""Tests for the harness layer (#17): the _harness line, and how checks feed it."""
import json

import pytest

from toolharness.agent.session import ToolResult
from toolharness.harness.checks import CheckResult
from toolharness.harness.session import LINE_PREFIX, HarnessSession, harness_line

CLEAN = '{"query": "q", "hits": []}'


class FakeSession:
    """An inner tool session that returns queued ToolResults and records whether it was entered and exited."""

    def __init__(self, *results):
        self.results = list(results)
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        self.exited = True

    async def list_tools(self):
        return [{"type": "function", "function": {"name": "search"}}]

    async def call(self, name, args):
        return self.results.pop(0)


class FakeCheck:
    """A check with a fixed result that records the history it was given."""

    def __init__(self, key, value, flags=()):
        self.key = key
        self.result = CheckResult(value=value, flags=list(flags))
        self.histories = []

    def run(self, tool, result, history):
        self.histories.append(list(history))
        return self.result


def test_harness_line_lists_check_values_then_flags():
    assert harness_line({}, []) == '_harness: {"flags": []}'
    assert harness_line({"fresh": None}, []) == '_harness: {"fresh": null, "flags": []}'
    assert LINE_PREFIX == "_harness: "


@pytest.mark.anyio
async def test_line_is_appended_and_the_text_is_otherwise_unchanged():
    async with HarnessSession(FakeSession(ToolResult(text=CLEAN, is_error=False)), []) as session:
        result = await session.call("search", {"query": "q"})

    assert result.text == CLEAN + '\n\n_harness: {"flags": []}'
    assert result.is_error is False
    assert result.harness_flags == []


@pytest.mark.anyio
async def test_each_check_adds_its_value_and_flags_in_order():
    checks = [FakeCheck("a", True), FakeCheck("b", False, ["b: bad"])]
    async with HarnessSession(FakeSession(ToolResult(text=CLEAN, is_error=False)), checks) as session:
        result = await session.call("search", {"query": "q"})

    assert result.text == CLEAN + '\n\n_harness: {"a": true, "b": false, "flags": ["b: bad"]}'
    assert result.harness_flags == ["b: bad"]


@pytest.mark.anyio
async def test_checks_see_the_episodes_earlier_raw_responses():
    first = ToolResult(text=CLEAN, is_error=False)
    second = ToolResult(text='{"title": "T"}', is_error=False)
    check = FakeCheck("a", True)
    async with HarnessSession(FakeSession(first, second), [check]) as session:
        await session.call("search", {"query": "q"})
        await session.call("get_article", {"title": "T"})

    assert check.histories == [[], [("search", first)]]


@pytest.mark.anyio
async def test_error_results_keep_is_error_and_get_the_line():
    error = ToolResult(text="503 Service Unavailable", is_error=True)
    async with HarnessSession(FakeSession(error), []) as session:
        result = await session.call("get_article", {"title": "T"})

    assert result.is_error is True
    assert result.text == '503 Service Unavailable\n\n_harness: {"flags": []}'


@pytest.mark.anyio
async def test_list_tools_passes_through_and_latency_is_recorded():
    inner = FakeSession(ToolResult(text=CLEAN, is_error=False))
    async with HarnessSession(inner, [FakeCheck("a", True)]) as session:
        assert inner.entered
        assert await session.list_tools() == await inner.list_tools()
        result = await session.call("search", {"query": "q"})

    assert inner.exited
    assert result.proxy_latency_ms is not None and result.proxy_latency_ms >= 0
