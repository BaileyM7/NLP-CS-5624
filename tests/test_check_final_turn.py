"""Offline tests for scripts/check_final_turn.py: its checks must fail unless the forced final-answer turn really ran."""
import importlib.util
from pathlib import Path

import pytest
from fakes import FakeOpenRouter, answer_reply, make_client, tool_reply

from toolharness.data.scenario import FaultType
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server


def load_check_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "check_final_turn.py"
    spec = importlib.util.spec_from_file_location("check_final_turn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(sample_scenarios):
    scenario = next(s for s in sample_scenarios if s.fault_type == FaultType.NONE)
    return build_server(EpisodeRegistry(sample_scenarios)), scenario


def search():
    return (200, tool_reply(("search", {"thought": "t", "query": "Shin Hyun-joon"})))


@pytest.mark.anyio
async def test_both_checks_pass_when_each_forced_turn_runs(setup):
    check = load_check_script()
    server, scenario = setup
    fake = FakeOpenRouter(
        search(), (200, answer_reply("Final Answer: 2005")),  # turn-limit check: tool call, then the forced turn 2
        search(), (200, answer_reply("Final Answer: 2005")),  # token-limit check: 50 tokens used, then the forced turn
    )
    async with make_client(fake) as llm:
        problems, _ = await check.check_model(llm, server, scenario, "haiku-5.5")

    assert problems == []
    assert [request["tool_choice"] for request in fake.requests] == ["auto", "none", "auto", "none"]
    assert fake.requests[3]["max_tokens"] == 1000


@pytest.mark.anyio
async def test_checks_fail_when_no_forced_turn_was_sent(setup):
    check = load_check_script()
    server, scenario = setup
    fake = FakeOpenRouter((200, answer_reply("Final Answer: 2005")), (200, answer_reply("Final Answer: 2005", output_tokens=30)))
    async with make_client(fake) as llm:
        problems, _ = await check.check_model(llm, server, scenario, "haiku-5.5")

    assert len(problems) == 2
    assert "wasn't tested" in problems[0]
    assert "token-limit" in problems[1]
