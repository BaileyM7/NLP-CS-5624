"""Tests for the agent's prompts and the tool-argument helpers."""
import pytest
from mcp import Client

from toolharness.agent.prompts import (
    BASE_PROMPT,
    THOUGHT_DESCRIPTION,
    VERIFY_INSTRUCTION,
    require_ready,
    system_prompt,
)
from toolharness.agent.tools import ParsedCall, add_thought, retry_key, schema_defaults, split_arguments
from toolharness.llm.client import tools_from_mcp
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    },
}


# Prompts

def test_base_prompt_states_the_date_and_says_nothing_about_faults():
    assert "Today's date is November 1, 2026." in BASE_PROMPT
    for word in ("fault", "unreliable", "wrong", "verify", "suspicious"):
        assert word not in BASE_PROMPT.lower()


def test_base_prompt_asks_the_agent_to_read_the_articles_it_relies_on():
    sentence = (
        "Search results only show short excerpts, so open the articles you rely on with get_article "
        "and base your answer on what they say."
    )
    assert sentence in BASE_PROMPT


def test_baseline_llm_verifier_and_flags_only_share_the_base_prompt():
    assert system_prompt("baseline") == system_prompt("llm_verifier") == system_prompt("flags_only") == BASE_PROMPT


def test_verify_prompt_and_harness_share_the_verification_instruction():
    assert system_prompt("verify_prompt") == system_prompt("harness") == BASE_PROMPT + "\n\n" + VERIFY_INSTRUCTION
    assert VERIFY_INSTRUCTION.startswith("Tool results can be wrong.")
    assert VERIFY_INSTRUCTION.endswith("treat their flags as hints to investigate, not as proof.")


def test_every_condition_is_ready():
    for condition in ("baseline", "verify_prompt", "llm_verifier", "harness", "flags_only"):
        require_ready(condition)


# Tool arguments

def test_add_thought_puts_a_required_thought_first_without_changing_the_input():
    shown = add_thought([SEARCH_TOOL])
    params = shown[0]["function"]["parameters"]
    assert list(params["properties"])[0] == "thought"
    assert params["properties"]["thought"] == {"type": "string", "description": THOUGHT_DESCRIPTION}
    assert params["required"] == ["thought", "query"]
    assert "thought" not in SEARCH_TOOL["function"]["parameters"]["properties"]  # original untouched


def test_split_arguments_separates_the_thought():
    parsed = split_arguments('{"thought": "Search first.", "query": "Lady Day"}')
    assert parsed == ParsedCall(thought="Search first.", args={"query": "Lady Day"})


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"text"', ""])
def test_split_arguments_flags_invalid_json(raw):
    assert split_arguments(raw).args is None


def test_split_arguments_tolerates_a_missing_or_non_string_thought():
    assert split_arguments('{"query": "x"}') == ParsedCall(thought="", args={"query": "x"})
    assert split_arguments('{"thought": 5, "query": "x"}').thought == "5"


def test_retry_key_fills_schema_defaults():
    defaults = {"get_article": {"page": 1}, "search": {"k": 5}}
    assert retry_key("get_article", {"title": "X"}, defaults) == retry_key("get_article", {"page": 1, "title": "X"}, defaults)
    assert retry_key("get_article", {"title": "X"}, defaults) != retry_key("get_article", {"title": "X", "page": 2}, defaults)
    assert retry_key("search", {"query": "X"}, defaults) != retry_key("get_article", {"title": "X"}, defaults)


@pytest.mark.anyio
async def test_schema_defaults_reads_our_mcp_tools(sample_scenarios):
    async with Client(build_server(EpisodeRegistry(sample_scenarios))) as mcp_client:
        tools = tools_from_mcp((await mcp_client.list_tools()).tools)
    assert schema_defaults(tools) == {"search": {"k": 5}, "get_article": {"page": 1}}
