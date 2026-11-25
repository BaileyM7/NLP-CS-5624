"""Smoke test for the OpenRouter client: one tool-call round trip per model, printing tokens and cost.

Each model gets a sample HotpotQA question and our real search/get_article tools. It must call a
tool; the mock app answers the call, and the model must then reply again without an error.

Usage:
    python scripts/smoke_models.py                           # the main-run models (Opus, Haiku, Luna)
    python scripts/smoke_models.py --all                     # plus the two optional models
    python scripts/smoke_models.py --models opus-5.5 --tool-choice required
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from mcp import Client

from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType, read_jsonl
from toolharness.llm.client import ChatResult, LLMError, OpenRouterClient, tools_from_mcp
from toolharness.llm.models import MAIN_RUN_MODELS, MODELS
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"
SYSTEM_PROMPT = "You answer questions using the search and get_article tools. Look facts up before answering."
MAX_TOKENS = 4000


async def load_tools(scenarios) -> list[dict]:
    """Our MCP server's tool definitions, converted for the chat API."""
    async with Client(build_server(EpisodeRegistry(scenarios))) as mcp_client:
        return tools_from_mcp((await mcp_client.list_tools()).tools)


def run_tool(app: MockApp, name: str, arguments: str) -> str:
    """Answer one tool call with the mock app; bad arguments become an error message for the model."""
    try:
        return app.call(name, json.loads(arguments)).text
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return json.dumps({"error": f"bad arguments: {error!r}"})


async def round_trip(
    llm: OpenRouterClient,
    model: str,
    question: str,
    app: MockApp,
    tools: list[dict],
    tool_choice: str = "auto",
):
    """Ask the question, answer every tool call the model makes, then let the model reply."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    first = await llm.chat(model, messages, tools=tools, tool_choice=tool_choice, max_tokens=MAX_TOKENS, tag="smoke")
    if not first.tool_calls:
        raise LLMError(f"answered without calling a tool (tool_choice={tool_choice}, finish_reason={first.finish_reason})")

    # Every tool call needs a result message before the model can continue
    messages.append(first.message)
    for call in first.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": run_tool(app, call.name, call.arguments)})
    second = await llm.chat(model, messages, tools=tools, max_tokens=MAX_TOKENS, tag="smoke")
    return first, second


def describe(first: ChatResult, second: ChatResult) -> str:
    """One summary line: the first tool call, tokens and cost for both calls, and how the second ended."""
    call = first.tool_calls[0]
    tokens_in = first.usage.input_tokens + second.usage.input_tokens
    tokens_out = first.usage.output_tokens + second.usage.output_tokens
    reasoning = first.usage.reasoning_tokens + second.usage.reasoning_tokens
    costs = [first.usage.cost_usd, second.usage.cost_usd]
    cost = "not reported" if None in costs else f"${sum(costs):.5f}"
    return (
        f"{call.name}({call.arguments[:60]}) | tokens in {tokens_in} out {tokens_out} (reasoning {reasoning}) | "
        f"cost {cost} | {first.latency_ms + second.latency_ms:.0f} ms | then: {second.finish_reason}"
    )


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="One tool-call round trip per model through OpenRouter.")
    parser.add_argument("--all", action="store_true", help="also test the optional models")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), help="test only these models")
    parser.add_argument("--tool-choice", default="auto", choices=["auto", "required"], help="tool_choice for the first call")
    parser.add_argument("--usage-log", help="append one row per API call to this JSONL file")
    args = parser.parse_args(argv)

    scenarios = read_jsonl(SAMPLE_FILE)
    scenario = next(s for s in scenarios if s.fault_type == FaultType.NONE)
    app = MockApp(scenario)
    tools = await load_tools(scenarios)
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # building the MCP server turned on INFO logging

    names = MAIN_RUN_MODELS
    if args.all:
        names = list(MODELS)
    if args.models:
        names = args.models
    print(f"Question: {scenario.question}\n")

    failed_main_run = []
    total_cost = 0.0
    async with OpenRouterClient(usage_log=args.usage_log) as llm:
        for name in names:
            try:
                first, second = await round_trip(llm, name, scenario.question, app, tools, args.tool_choice)
            except LLMError as error:
                print(f"FAILED {name:12} {error}")
                if MODELS[name].main_run:
                    failed_main_run.append(name)
                continue
            total_cost += (first.usage.cost_usd or 0) + (second.usage.cost_usd or 0)
            print(f"ok     {name:12} {describe(first, second)}")

    print(f"\nTotal cost: ${total_cost:.5f}")
    if failed_main_run:
        print(f"Main-run models that failed: {', '.join(failed_main_run)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
