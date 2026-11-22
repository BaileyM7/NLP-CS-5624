"""Tests for the episode limits: 12 turns, 3 retries, 8,000 output tokens, and the forced final-answer turn."""
import pytest
from fakes import FakeOpenRouter, answer_reply, completion, make_client, tool_reply

from toolharness.agent.prompts import FINAL_TURN_MESSAGE
from toolharness.agent.runner import NOT_RUN, Budget, run_episode
from toolharness.agent.session import InMemoryToolSession
from toolharness.data.scenario import FaultType
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


def search(query, **kwargs):
    return (200, tool_reply(("search", {"thought": "t", "query": query}), **kwargs))


def read_answer_page():
    return (200, tool_reply(("get_article", {"thought": "again", "title": ANSWER_PAGE})))


# Turn limit

@pytest.mark.anyio
async def test_turn_12_is_a_forced_final_answer_turn(registry, scenario):
    fake = FakeOpenRouter(*[search(f"q{i}") for i in range(11)], (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    assert len(episode.turns) == 12 and len(fake.requests) == 12
    last = fake.requests[11]
    assert last["tool_choice"] == "none" and "tools" in last
    assert last["messages"][-1] == {"role": "user", "content": FINAL_TURN_MESSAGE}
    assert all(request["tool_choice"] == "auto" for request in fake.requests[:11])
    assert (episode.stop_reason, episode.final_answer) == ("max_turns", "2005")


@pytest.mark.anyio
async def test_tool_calls_on_the_final_turn_are_not_run(registry, scenario):
    final = tool_reply(("search", {"thought": "one more", "query": "x"}), content="Final Answer: 2005")
    fake = FakeOpenRouter(search("q"), (200, final))
    episode = await run(fake, registry, scenario, Budget(max_turns=2))

    assert episode.turns[1].tool_calls == []
    assert "Thought: one more" in episode.turns[1].assistant_text
    assert (episode.final_answer, episode.stop_reason) == ("2005", "max_turns")
    assert len(registry.episodes["ep-1"].backend.history) == 1


# Retry limit

@pytest.mark.anyio
async def test_fourth_retry_is_not_run(registry, scenario):
    fake = FakeOpenRouter(*[read_answer_page() for _ in range(5)], (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    # Calls 2-4 are retries 1-3 and run; call 5 is the 4th retry
    assert [turn.tool_calls[0].raw_tool_response == NOT_RUN for turn in episode.turns[:5]] == [False, False, False, False, True]
    assert episode.turns[4].tool_calls[0].repeats_earlier_call
    assert len(registry.episodes["ep-1"].backend.history) == 4
    assert fake.requests[5]["tool_choice"] == "none"
    assert episode.stop_reason == "retry_budget" and len(episode.turns) == 6


@pytest.mark.anyio
async def test_retries_are_shared_across_tools(registry, scenario):
    fake = FakeOpenRouter(
        search("q"), search("q"), read_answer_page(), read_answer_page(), search("q"), read_answer_page(),
        (200, answer_reply("Final Answer: 2005")),
    )
    episode = await run(fake, registry, scenario)

    assert episode.turns[4].tool_calls[0].raw_tool_response != NOT_RUN  # retry 3
    assert episode.turns[5].tool_calls[0].raw_tool_response == NOT_RUN  # retry 4
    assert episode.stop_reason == "retry_budget"


@pytest.mark.anyio
async def test_calls_after_the_fourth_retry_in_the_same_turn_are_not_run(registry, scenario):
    both = tool_reply(("get_article", {"thought": "again", "title": ANSWER_PAGE}), ("search", {"thought": "new", "query": "brand new"}))
    fake = FakeOpenRouter(*[read_answer_page() for _ in range(4)], (200, both), (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    assert [call.raw_tool_response for call in episode.turns[4].tool_calls] == [NOT_RUN, NOT_RUN]
    tool_messages = [m for m in fake.requests[5]["messages"] if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages[-2:]] == [NOT_RUN, NOT_RUN]
    assert len(registry.episodes["ep-1"].backend.history) == 4


# Token limit

@pytest.mark.anyio
async def test_token_budget_forces_the_final_turn(registry, scenario):
    fake = FakeOpenRouter(
        search("q1", output_tokens=3500), search("q2", output_tokens=3500),
        (200, answer_reply("Final Answer: 2005", output_tokens=400)),
    )
    episode = await run(fake, registry, scenario)

    assert [request["max_tokens"] for request in fake.requests] == [7000, 3500, 1000]
    assert fake.requests[2]["tool_choice"] == "none"
    assert episode.stop_reason == "token_budget"
    assert episode.totals.agent_tokens.output_tokens == 7400


@pytest.mark.anyio
async def test_length_finish_forces_the_final_turn(registry, scenario):
    fake = FakeOpenRouter((200, answer_reply("I was thinking about", finish_reason="length")), (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    assert len(episode.turns) == 2
    assert fake.requests[1]["tool_choice"] == "none"
    assert (episode.stop_reason, episode.final_answer) == ("token_budget", "2005")


@pytest.mark.anyio
async def test_cut_off_reply_with_no_text_is_not_sent_back(registry, scenario):
    empty_cut_off = completion({"role": "assistant", "content": None}, finish_reason="length")
    fake = FakeOpenRouter((200, empty_cut_off), (200, answer_reply("Final Answer: 2005")))
    episode = await run(fake, registry, scenario)

    assert [m["role"] for m in fake.requests[1]["messages"]] == ["system", "user", "user"]
    assert (episode.stop_reason, episode.final_answer) == ("token_budget", "2005")


@pytest.mark.anyio
async def test_no_tokens_left_ends_without_a_final_turn(registry, scenario):
    fake = FakeOpenRouter(search("q", output_tokens=8500))
    episode = await run(fake, registry, scenario)

    assert len(fake.requests) == 1
    assert (episode.stop_reason, episode.final_answer) == ("token_budget", None)
