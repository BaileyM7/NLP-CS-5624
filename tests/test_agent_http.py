"""End-to-end over HTTP: the runner against the real tool server subprocess, with a fake model."""
import json

import pytest
from fakes import FakeOpenRouter, answer_reply, free_port, make_client, tool_reply

from toolharness.agent.batch import check_server
from toolharness.agent.runner import run_episode
from toolharness.agent.session import HttpToolSession, ToolServerError
from toolharness.data.scenario import FaultType


@pytest.mark.anyio
async def test_episode_runs_over_http_and_the_fault_reaches_the_agent(http_server, sample_scenarios, tmp_path):
    scenario = next(s for s in sample_scenarios if s.fault_type == FaultType.HARD_FAILURE)
    fake = FakeOpenRouter(
        (200, tool_reply(("get_article", {"thought": "Read it.", "title": "Marrying the Mafia II"}))),
        (200, answer_reply("Final Answer: 2005")),
    )
    async with HttpToolSession(http_server + "/mcp", "http-ep-1", scenario.scenario_id, 0) as session:
        async with make_client(fake) as llm:
            episode = await run_episode(llm, session, scenario, "haiku-5.5", "baseline", "http-ep-1")

    assert episode.turns[0].tool_calls[0].is_error  # the injected hard failure
    assert episode.final_answer == "2005"
    rows = [json.loads(line) for line in (tmp_path / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["episode_id"] == "http-ep-1" and row["corrupted"] for row in rows)


@pytest.mark.anyio
async def test_http_session_to_a_down_server_raises_tool_server_error():
    url = f"http://127.0.0.1:{free_port()}/mcp"
    with pytest.raises(ToolServerError):
        async with HttpToolSession(url, "e1", "s1", 0) as session:
            await session.call("search", {"query": "x"})


def test_check_server(http_server):
    assert check_server(http_server + "/mcp")
    assert not check_server(f"http://127.0.0.1:{free_port()}/mcp")
