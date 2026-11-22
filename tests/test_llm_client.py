"""Tests for the model registry and the OpenRouter client. A fake HTTP transport stands in for OpenRouter."""
import importlib.util
import json
from pathlib import Path

import httpx2
import pytest
from mcp import Client

from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.llm import client as client_module
from toolharness.llm.client import MAX_DELAY_S, LLMError, OpenRouterClient, ToolCallRequest, tools_from_mcp
from toolharness.llm.models import FACT_EDIT_MODEL, JUDGE_MODEL, MAIN_RUN_MODELS, MODELS, VERIFIER_MODEL, get_model
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

MESSAGES = [{"role": "user", "content": 'Who was nicknamed "Lady Day"?'}]
SEARCH_TOOL = {"type": "function", "function": {"name": "search", "description": "", "parameters": {"type": "object"}}}


def completion(message, usage=None, finish_reason="stop"):
    """An OpenAI-style chat completion body."""
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


TOOL_CALL_REPLY = completion(
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "Lady Day"}'}}
        ],
        "reasoning": "I should search first.",
    },
    usage={
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "total_tokens": 160,
        "cost": 0.0012,
        "completion_tokens_details": {"reasoning_tokens": 25},
        "prompt_tokens_details": {"cached_tokens": 100},
    },
    finish_reason="tool_calls",
)


class FakeOpenRouter:
    """Answers requests with the queued (status, body) pairs in order and records each request body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return httpx2.Response(status, json=body)


def make_client(fake, **kwargs):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake))
    return OpenRouterClient(api_key="test-key", http_client=http, base_delay=0, **kwargs)


# Registry

def test_registry_has_the_four_main_run_models_and_two_optional_ones():
    assert MAIN_RUN_MODELS == ["opus-5.5", "haiku-5.5", "gpt-6-astra", "gpt-6-luna"]
    assert sorted(MODELS) == sorted(MAIN_RUN_MODELS + ["sonnet-5.5", "gpt-6.1-sol"])


def test_roles_follow_the_experiment_spec():
    assert (VERIFIER_MODEL, JUDGE_MODEL, FACT_EDIT_MODEL) == ("gpt-6-astra", "opus-5.5", "sonnet-5.5")
    assert VERIFIER_MODEL != JUDGE_MODEL
    assert {VERIFIER_MODEL, JUDGE_MODEL, FACT_EDIT_MODEL} <= set(MODELS)


def test_models_are_found_by_short_name_or_openrouter_id():
    assert get_model("opus-5.5").openrouter_id == "anthropic/claude-opus-5.5"
    assert get_model("anthropic/claude-sonnet-5.5").name == "sonnet-5.5"
    with pytest.raises(ValueError, match="unknown model"):
        get_model("gpt-7")


# Requests and results

@pytest.mark.anyio
async def test_tool_call_reply_is_parsed_with_usage_and_cost():
    fake = FakeOpenRouter((200, TOOL_CALL_REPLY))
    async with make_client(fake) as llm:
        result = await llm.chat("haiku-5.5", MESSAGES, tools=[SEARCH_TOOL], tool_choice="required")

    assert result.tool_calls == [ToolCallRequest(id="call_1", name="search", arguments='{"query": "Lady Day"}')]
    assert (result.model, result.model_id, result.finish_reason) == ("haiku-5.5", "anthropic/claude-haiku-5.5", "tool_calls")
    usage = result.usage
    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens, usage.cached_tokens) == (120, 40, 25, 100)
    assert usage.cost_usd == 0.0012
    assert result.attempts == 1
    assert result.latency_ms >= 0

    request = fake.requests[0]
    assert request["model"] == "anthropic/claude-haiku-5.5"
    assert request["tools"] == [SEARCH_TOOL]
    assert request["tool_choice"] == "required"
    assert request["usage"] == {"include": True}


@pytest.mark.anyio
async def test_assistant_message_keeps_tool_calls_and_reasoning_for_the_next_turn():
    fake = FakeOpenRouter((200, TOOL_CALL_REPLY))
    async with make_client(fake) as llm:
        result = await llm.chat("haiku-5.5", MESSAGES, tools=[SEARCH_TOOL])

    assert result.message["role"] == "assistant"
    assert result.message["tool_calls"][0]["id"] == "call_1"
    assert result.message["reasoning"] == "I should search first."  # OpenRouter needs it sent back on the next turn


@pytest.mark.anyio
async def test_optional_settings_are_only_sent_when_given():
    fake = FakeOpenRouter((200, completion({"role": "assistant", "content": "Billie Holiday"})), (200, completion({"role": "assistant", "content": "x"})))
    async with make_client(fake) as llm:
        plain = await llm.chat("opus-5.5", MESSAGES)
        await llm.chat("opus-5.5", MESSAGES, max_tokens=500, reasoning={"effort": "low"})

    assert plain.content == "Billie Holiday"
    assert plain.tool_calls == []
    assert "tools" not in fake.requests[0] and "max_tokens" not in fake.requests[0] and "reasoning" not in fake.requests[0]
    assert fake.requests[1]["max_tokens"] == 500
    assert fake.requests[1]["reasoning"] == {"effort": "low"}


# Retries

@pytest.mark.anyio
async def test_rate_limits_and_server_errors_are_retried():
    fake = FakeOpenRouter((429, {"error": {"message": "rate limited"}}), (502, {"error": {"message": "bad gateway"}}), (200, TOOL_CALL_REPLY))
    async with make_client(fake) as llm:
        result = await llm.chat("gpt-6-luna", MESSAGES)
    assert result.attempts == 3
    assert len(fake.requests) == 3


@pytest.mark.anyio
async def test_reply_without_choices_is_retried():
    no_choices = {"id": "gen-1", "object": "chat.completion", "created": 0, "model": "test", "choices": [], "error": {"message": "provider error"}}
    fake = FakeOpenRouter((200, no_choices), (200, TOOL_CALL_REPLY))
    async with make_client(fake) as llm:
        result = await llm.chat("gpt-6-luna", MESSAGES)
    assert result.attempts == 2


@pytest.mark.anyio
async def test_client_gives_up_after_max_attempts():
    fake = FakeOpenRouter(*[(500, {"error": {"message": "down"}})] * 3)
    async with make_client(fake, max_attempts=3) as llm:
        with pytest.raises(LLMError, match="gave up after 3 attempts"):
            await llm.chat("gpt-6-luna", MESSAGES)
    assert len(fake.requests) == 3


@pytest.mark.anyio
async def test_bad_requests_are_not_retried():
    fake = FakeOpenRouter((400, {"error": {"message": "context too long"}}))
    async with make_client(fake) as llm:
        with pytest.raises(LLMError, match="HTTP 400"):
            await llm.chat("gpt-6-luna", MESSAGES)
    assert len(fake.requests) == 1


@pytest.mark.anyio
async def test_errors_include_the_providers_own_message():
    body = {
        "error": {
            "message": "Provider returned error",
            "code": 400,
            "metadata": {"provider_name": "Anthropic", "raw": '{"message": "tool_choice any is not supported"}'},
        }
    }
    fake = FakeOpenRouter((400, body))
    async with make_client(fake) as llm:
        with pytest.raises(LLMError, match="Anthropic: .*tool_choice any is not supported"):
            await llm.chat("opus-5.5", MESSAGES)


def test_backoff_grows_and_respects_retry_after():
    llm = OpenRouterClient(api_key="test-key", base_delay=1.0)
    assert 1.0 <= llm.retry_delay(1, None) <= 2.0
    assert 4.0 <= llm.retry_delay(3, None) <= 5.0
    assert llm.retry_delay(1, 10.0) == 10.0
    assert llm.retry_delay(1, 600.0) == MAX_DELAY_S


# Usage log, API key and MCP tools

@pytest.mark.anyio
async def test_every_call_is_written_to_the_usage_log(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    fake = FakeOpenRouter((200, TOOL_CALL_REPLY), (200, TOOL_CALL_REPLY))
    async with make_client(fake, usage_log=log_path) as llm:
        await llm.chat("haiku-5.5", MESSAGES, tag="agent")
        await llm.chat("haiku-5.5", MESSAGES, tag="verifier")

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["tag"] for row in rows] == ["agent", "verifier"]
    assert rows[0]["cost_usd"] == 0.0012
    assert rows[0]["model"] == "haiku-5.5"


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(client_module, "load_dotenv", lambda: None)  # don't read the real .env
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is not set"):
        OpenRouterClient()


def load_smoke_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_models.py"
    spec = importlib.util.spec_from_file_location("smoke_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.anyio
async def test_smoke_round_trip_sends_the_tool_result_back(sample_scenarios):
    smoke = load_smoke_script()
    scenario = next(s for s in sample_scenarios if s.fault_type == FaultType.NONE)
    app = MockApp(scenario)
    answer = completion({"role": "assistant", "content": "Billie Holiday"})
    fake = FakeOpenRouter((200, TOOL_CALL_REPLY), (200, answer))

    async with make_client(fake) as llm:
        first, second = await smoke.round_trip(llm, "haiku-5.5", scenario.question, app, [SEARCH_TOOL])

    assert second.content == "Billie Holiday"
    sent = fake.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool"]
    assert sent[2]["tool_calls"][0]["id"] == "call_1"
    assert sent[3]["tool_call_id"] == "call_1"
    assert sent[3]["content"] == app.call("search", {"query": "Lady Day"}).text


@pytest.mark.anyio
async def test_mcp_tools_convert_to_openai_function_tools(sample_scenarios):
    async with Client(build_server(EpisodeRegistry(sample_scenarios))) as mcp_client:
        mcp_tools = (await mcp_client.list_tools()).tools

    tools = {tool["function"]["name"]: tool for tool in tools_from_mcp(mcp_tools)}
    assert set(tools) == {"search", "get_article"}
    assert tools["search"]["type"] == "function"
    assert tools["search"]["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert "ctx" not in tools["get_article"]["function"]["parameters"]["properties"]
