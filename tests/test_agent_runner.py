"""Tests for the agent loop on normal turns. A fake OpenRouter plays the model; the tools are the real in-memory server."""
import json

import pytest
from fakes import FakeOpenRouter, answer_reply, make_client, tool_reply

from toolharness.agent.prompts import BASE_PROMPT
from toolharness.agent.runner import INVALID_ARGUMENTS, Budget, extract_final_answer, run_episode
from toolharness.agent.session import InMemoryToolSession
from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.eval.trace import Episode
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

ANSWER_PAGE = "Marrying the Mafia II"


@pytest.fixture
def registry(sample_scenarios):
    return EpisodeRegistry(sample_scenarios)


@pytest.fixture
def scenario(sample_scenarios):
    """The Shin Hyun-joon question (gold answer 2005), no fault."""
    return next(s for s in sample_scenarios if s.fault_type == FaultType.NONE)


async def run(fake, registry, scenario, budget=Budget()):
    """Run one episode against the in-memory server with the fake model."""
    async with InMemoryToolSession(build_server(registry), "ep-1", scenario.scenario_id, 0) as session:
        async with make_client(fake) as llm:
            return await run_episode(llm, session, scenario, "haiku-5.5", "baseline", "ep-1", budget)


@pytest.mark.anyio
async def test_answer_after_one_search_writes_a_valid_episode(registry, scenario):
    fake = FakeOpenRouter(
        (200, tool_reply(("search", {"thought": "Find the sequel.", "query": "Shin Hyun-joon comedy sequel"}))),
        (200, answer_reply("It came out in 2005.\nFinal Answer: 2005")),
    )
    episode = await run(fake, registry, scenario)

    assert (episode.final_answer, episode.stop_reason, episode.gold_answer) == ("2005", "final_answer", "2005")
    assert (episode.episode_id, episode.scenario_id, episode.model, episode.condition) == ("ep-1", scenario.scenario_id, "haiku-5.5", "baseline")
    assert [turn.index for turn in episode.turns] == [1, 2]
    call = episode.turns[0].tool_calls[0]
    assert call.tool_name == "search" and call.tool_args == {"query": "Shin Hyun-joon comedy sequel"}
    assert "Thought: Find the sequel." in episode.turns[0].assistant_text
    assert episode.turns[1].assistant_text == "It came out in 2005.\nFinal Answer: 2005"
    assert episode.totals.agent_tokens.output_tokens == 100
    assert episode.totals.agent_cost_usd == pytest.approx(0.002)
    assert Episode.model_validate_json(episode.model_dump_json()) == episode


@pytest.mark.anyio
async def test_first_request_has_prompt_question_thought_tools_and_budget(registry, sample_scenarios):
    scenario = next(s for s in sample_scenarios if s.fault_type == FaultType.HARD_FAILURE)
    fake = FakeOpenRouter((200, answer_reply("Final Answer: 2005")))
    await run(fake, registry, scenario)

    request = fake.requests[0]
    assert request["messages"] == [{"role": "system", "content": BASE_PROMPT}, {"role": "user", "content": scenario.question}]
    assert {tool["function"]["name"] for tool in request["tools"]} == {"search", "get_article"}
    for tool in request["tools"]:
        assert tool["function"]["parameters"]["required"][0] == "thought"
    assert request["tool_choice"] == "auto"
    assert request["max_tokens"] == 7000
    assert "reasoning" not in request
    sent = json.dumps(request["messages"]).lower()
    for hidden in ("hard_failure", "fault", scenario.gold_answer):
        assert hidden not in sent


@pytest.mark.anyio
async def test_tool_results_go_back_in_order_and_thought_never_reaches_the_server(registry, scenario):
    query = "Shin Hyun-joon comedy sequel"
    reply = tool_reply(("search", {"thought": "Find the sequel.", "query": query}))
    call_id = reply["choices"][0]["message"]["tool_calls"][0]["id"]
    fake = FakeOpenRouter((200, reply), (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    clean = MockApp(scenario).call("search", {"query": query}).text
    messages = fake.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["id"] == call_id
    assert messages[3] == {"role": "tool", "tool_call_id": call_id, "content": clean}
    assert episode.turns[0].tool_calls[0].raw_tool_response == clean
    assert registry.episodes["ep-1"].backend.history[0].call.args == {"query": query, "k": 5}


@pytest.mark.anyio
async def test_every_call_in_a_turn_gets_a_tool_message(registry, scenario):
    reply = tool_reply(("search", {"thought": "t", "query": "Shin Hyun-joon"}), ("get_article", "not json"))
    fake = FakeOpenRouter((200, reply), (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    tool_messages = [m for m in fake.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2
    assert tool_messages[1]["content"] == INVALID_ARGUMENTS
    bad = episode.turns[0].tool_calls[1]
    assert bad.tool_args == {"_unparsed": "not json"}
    assert bad.raw_tool_response == INVALID_ARGUMENTS
    assert bad.is_error is False
    assert len(registry.episodes["ep-1"].backend.history) == 1  # only the valid call reached the server


@pytest.mark.anyio
async def test_repeated_call_is_marked(registry, scenario):
    fake = FakeOpenRouter(
        (200, tool_reply(("get_article", {"thought": "Read it.", "title": ANSWER_PAGE}))),
        (200, tool_reply(("get_article", {"thought": "Read it again.", "title": ANSWER_PAGE, "page": 1}))),
        (200, answer_reply("Final Answer: 2005")),
    )
    episode = await run(fake, registry, scenario)
    assert episode.turns[0].tool_calls[0].repeats_earlier_call is False
    assert episode.turns[1].tool_calls[0].repeats_earlier_call is True


@pytest.mark.anyio
async def test_answer_without_any_tool_call_ends_on_turn_1(registry, scenario):
    fake = FakeOpenRouter((200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)
    assert len(episode.turns) == 1
    assert episode.turns[0].tool_calls == []
    assert (episode.final_answer, episode.stop_reason) == ("2005", "final_answer")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Final Answer: 2005", "2005"),
        ("Let me think.\nfinal answer: **Paris**", "Paris"),
        ("Final Answer: A\nFinal Answer: B", "B"),
        ("Final Answer: 2005\nThe film was a hit.", "2005"),
        ("Just 2005", "Just 2005"),
        ("Final Answer:   ", None),
        ("", None),
        (None, None),
        ("**Final Answer**: 2005", "2005"),
        ("**Final Answer:** Paris", "Paris"),
        ("Final Answer:\n2005", "2005"),
        ("Final Answer:\n\n**2005**", "2005"),
    ],
)
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected
