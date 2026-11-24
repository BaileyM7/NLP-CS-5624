"""Verifier isolation, validation, accounting and end-to-end batch integration."""
import dataclasses
import json

import pytest
from fakes import SAMPLE_FILE, FakeOpenRouter, answer_reply, make_client, tool_reply

from toolharness.agent.batch import BatchConfig, ManifestMismatch, check_manifest, manifest_entry, run_batch
from toolharness.agent.prompts import BASE_PROMPT
from toolharness.agent.session import InMemoryToolSession, ToolResult
from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.eval.trace import read_jsonl, trace_filename
from toolharness.harness.verifier import LINE_PREFIX, RESPONSE_FORMAT, SYSTEM_PROMPT, VerifierSession
from toolharness.llm.client import LLMError
from toolharness.llm.models import JUDGE_MODEL, VERIFIER_MODEL, get_model
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server


def verdict(suspicious=False, fault_guess=None, reason="No evidence of a problem."):
    return (200, answer_reply(json.dumps({"suspicious": suspicious, "fault_guess": fault_guess, "reason": reason})))


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def list_tools(self):
        return []

    async def call(self, name, args):
        self.calls.append((name, args))
        return self.results.pop(0)


@pytest.mark.anyio
async def test_clean_and_faulted_responses_are_passively_verified_with_raw_history(tmp_path):
    inner = FakeSession(ToolResult('{"hits":[]}', False), ToolResult("503 Service Unavailable", True))
    fake = FakeOpenRouter(verdict(), verdict(True, "hard failure", "The tool returned an error."))
    async with make_client(fake, usage_log=tmp_path / "usage.jsonl") as llm:
        async with VerifierSession(inner, llm, "Question?") as session:
            clean = await session.call("search", {"query": "q"})
            faulted = await session.call("get_article", {"title": "X"})

    assert inner.calls == [("search", {"query": "q"}), ("get_article", {"title": "X"})]
    assert not clean.is_error and faulted.is_error
    assert clean.text.startswith('{"hits":[]}\n\n' + LINE_PREFIX)
    assert faulted.text.startswith("503 Service Unavailable\n\n" + LINE_PREFIX)
    assert clean.verifier_output.suspicious is False
    assert faulted.verifier_output.suspicious is True
    assert json.loads(faulted.text.split(LINE_PREFIX)[1]) == faulted.verifier_output.model_dump()
    assert faulted.harness_flags is None and faulted.proxy_latency_ms is None
    assert faulted.verifier_tokens.input_tokens == 100
    assert faulted.verifier_cost_usd == 0.001
    assert faulted.verifier_latency_ms is not None
    assert VERIFIER_MODEL != JUDGE_MODEL
    for request in fake.requests:
        assert request["model"] == get_model(VERIFIER_MODEL).openrouter_id
        assert request["messages"][0]["content"] == SYSTEM_PROMPT
        assert request["response_format"] == RESPONSE_FORMAT
        assert "tools" not in request
    first, second = [json.loads(request["messages"][1]["content"]) for request in fake.requests]
    assert first == {"question": "Question?", "earlier_responses": [], "new_response": {"tool": "search", "response": '{"hits":[]}', "is_error": False}}
    assert second["earlier_responses"] == [first["new_response"]]
    assert second["new_response"] == {"tool": "get_article", "response": "503 Service Unavailable", "is_error": True}
    assert "_verifier" not in json.dumps(second)
    assert [json.loads(line)["tag"] for line in (tmp_path / "usage.jsonl").read_text().splitlines()] == ["verifier", "verifier"]


@pytest.mark.anyio
@pytest.mark.parametrize("reply", [
    "not json",
    '{"suspicious":"false","fault_guess":null,"reason":"ok"}',
    '{"suspicious":false,"reason":"ok"}',
    '{"suspicious":false,"fault_guess":null,"reason":""}',
    '{"suspicious":false,"fault_guess":null,"reason":"ok","extra":1}',
])
async def test_invalid_verifier_output_fails_explicitly(reply):
    fake = FakeOpenRouter((200, answer_reply(reply)))
    async with make_client(fake) as llm:
        session = VerifierSession(FakeSession(ToolResult("x", False)), llm, "Q?")
        with pytest.raises(LLMError, match="invalid verifier response"):
            await session.call("search", {})
        assert session.history == []


@pytest.mark.anyio
async def test_missing_verifier_cost_fails_explicitly():
    response = answer_reply('{"suspicious":false,"fault_guess":null,"reason":"ok"}')
    del response["usage"]["cost"]
    async with make_client(FakeOpenRouter((200, response))) as llm:
        session = VerifierSession(FakeSession(ToolResult("x", False)), llm, "Q?")
        with pytest.raises(LLMError, match="token usage and billed cost"):
            await session.call("search", {})


@pytest.mark.anyio
async def test_verifier_batch_records_separate_usage_and_sends_note_to_agent(tmp_path, sample_scenarios):
    config = BatchConfig(
        run_id="verifier-test", model="haiku-5.5", condition="llm_verifier",
        scenario_files=[SAMPLE_FILE], questions=1, faults=[FaultType.NONE], concurrency=1, runs_dir=tmp_path,
    )
    server = build_server(EpisodeRegistry(sample_scenarios))
    fake = FakeOpenRouter(
        (200, tool_reply(("search", {"thought": "Search.", "query": "Shin Hyun-joon"}))),
        verdict(),
        (200, answer_reply("Final Answer: 2005")),
    )
    async with make_client(fake, usage_log=config.run_dir / "llm_usage.jsonl") as llm:
        summary = await run_batch(config, llm, lambda e, s, seed: InMemoryToolSession(server, e, s, seed))
    assert (summary.ran, summary.errored) == (1, 0)
    assert summary.verifier_cost_usd == 0.001
    [episode] = read_jsonl(config.run_dir / "traces" / trace_filename(config.model, config.condition))
    [call] = episode.turns[0].tool_calls
    assert call.verifier_output is not None and not call.verifier_output.suspicious
    assert call.verifier_tokens == episode.totals.verifier_tokens
    assert call.verifier_cost_usd == episode.totals.verifier_cost_usd == 0.001
    assert episode.totals.agent_tokens.output_tokens == 100
    assert episode.totals.verifier_tokens.output_tokens == 50
    assert episode.totals.agent_cost_usd == 0.002
    assert fake.requests[0]["messages"][0]["content"] == BASE_PROMPT
    assert fake.requests[2]["messages"][-1]["content"] == call.raw_tool_response
    assert "_harness:" not in call.raw_tool_response
    assert call.harness_flags is None
    assert [json.loads(line)["tag"] for line in (config.run_dir / "llm_usage.jsonl").read_text().splitlines()] == ["agent", "verifier", "agent"]


@pytest.mark.anyio
@pytest.mark.parametrize("fault", [FaultType.NONE, FaultType.HARD_FAILURE])
async def test_real_mcp_clean_and_faulted_results_reach_verifier(tmp_path, sample_scenarios, fault):
    scenario = next(s for s in sample_scenarios if s.fault_type == fault)
    title, page = sorted(MockApp(scenario).answer_pages())[0]
    config = BatchConfig(
        run_id="r", model="haiku-5.5", condition="llm_verifier",
        scenario_files=[SAMPLE_FILE], questions=1, faults=[fault], concurrency=1, runs_dir=tmp_path,
    )
    server = build_server(EpisodeRegistry(sample_scenarios, gt_log_path=tmp_path / "ground_truth.jsonl"))
    suspicious = fault == FaultType.HARD_FAILURE
    fake = FakeOpenRouter(
        (200, tool_reply(("get_article", {"thought": "Read the article.", "title": title, "page": page}))),
        verdict(suspicious, "hard failure" if suspicious else None, "A simulated assessment."),
        (200, answer_reply("Final Answer: 2005")),
    )
    async with make_client(fake) as llm:
        summary = await run_batch(config, llm, lambda e, s, seed: InMemoryToolSession(server, e, s, seed))
    assert (summary.ran, summary.errored) == (1, 0)
    [episode] = read_jsonl(config.run_dir / "traces" / trace_filename(config.model, config.condition))
    [call] = episode.turns[0].tool_calls
    verifier_input = json.loads(fake.requests[1]["messages"][1]["content"])
    assert verifier_input["new_response"]["is_error"] is suspicious
    assert call.is_error is suspicious and call.verifier_output.suspicious is suspicious
    assert verifier_input["new_response"]["response"] == call.raw_tool_response.split("\n\n" + LINE_PREFIX)[0]
    [ground_truth] = [json.loads(line) for line in (tmp_path / "ground_truth.jsonl").read_text().splitlines()]
    assert ground_truth["corrupted"] is suspicious


@pytest.mark.anyio
async def test_invalid_verifier_reply_fails_episode_and_resume_uses_fresh_attempt(tmp_path, sample_scenarios):
    config = BatchConfig(
        run_id="r", model="haiku-5.5", condition="llm_verifier",
        scenario_files=[SAMPLE_FILE], questions=1, faults=[FaultType.NONE], concurrency=1, runs_dir=tmp_path,
    )
    server = build_server(EpisodeRegistry(sample_scenarios))
    open_session = lambda e, s, seed: InMemoryToolSession(server, e, s, seed)
    search = (200, tool_reply(("search", {"thought": "Search.", "query": "q"})))
    async with make_client(FakeOpenRouter(search, (200, answer_reply("bad JSON")))) as llm:
        summary = await run_batch(config, llm, open_session)
    assert (summary.ran, summary.errored) == (0, 1)
    [error] = [json.loads(line) for line in (config.run_dir / "errors.jsonl").read_text().splitlines()]
    assert "invalid verifier response" in error["error"]
    async with make_client(FakeOpenRouter(search, verdict(), (200, answer_reply("Final Answer: 2005")))) as llm:
        summary = await run_batch(config, llm, open_session)
    assert (summary.ran, summary.errored) == (1, 0)
    [episode] = read_jsonl(config.run_dir / "traces" / trace_filename(config.model, config.condition))
    assert episode.episode_id.endswith("__a2")


def test_manifest_freezes_verifier_settings(tmp_path):
    config = BatchConfig(run_id="r", model="haiku-5.5", condition="llm_verifier", scenario_files=[SAMPLE_FILE])
    entry = manifest_entry(config)
    assert entry["verifier"]["model"] == VERIFIER_MODEL
    assert manifest_entry(dataclasses.replace(config, condition="baseline"))["verifier"] is None
    path = tmp_path / "manifest.json"
    check_manifest(path, entry)
    changed = {**entry, "verifier": {**entry["verifier"], "system_prompt": "changed"}}
    with pytest.raises(ManifestMismatch, match="verifier"):
        check_manifest(path, changed)
