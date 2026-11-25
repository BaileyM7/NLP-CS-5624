"""Live check for the agent runner's forced final-answer turn (paid; a few cents).

Each model runs two short episodes on a sample question:
- turn-limit check: a 2-turn limit, so turn 2 is the forced final-answer turn after a tool round trip
  (tool_choice="none", tool history in the conversation)
- token-limit check: an 1,050-token budget, so the first turn gets only 50 tokens (likely cut off) and the
  forced final-answer turn gets about 1,000

Both must reach their forced turn and get an answer from it.

Usage:
    python scripts/check_final_turn.py                    # the main-run models (Opus, Haiku, Luna)
    python scripts/check_final_turn.py --models opus-5.5
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from toolharness.agent.runner import Budget, run_episode
from toolharness.agent.session import InMemoryToolSession
from toolharness.data.scenario import FaultType, Scenario, read_jsonl
from toolharness.eval.trace import Episode
from toolharness.llm.client import LLMError, OpenRouterClient
from toolharness.llm.models import MAIN_RUN_MODELS, MODELS
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"
TURN_LIMIT = Budget(max_turns=2, max_output_tokens=2000)
TOKEN_LIMIT = Budget(max_output_tokens=1050)


async def run_check(llm, server, scenario: Scenario, model: str, name: str, budget: Budget) -> Episode:
    episode_id = f"check-{name}-{model}"
    async with InMemoryToolSession(server, episode_id, scenario.scenario_id, 0) as session:
        return await run_episode(llm, session, scenario, model, "baseline", episode_id, budget)


async def check_model(llm, server, scenario: Scenario, model: str) -> tuple[list[str], float]:
    """Run both checks for one model; returns the problems found (empty = ok) and the cost."""
    problems = []

    turns = await run_check(llm, server, scenario, model, "turns", TURN_LIMIT)
    if not turns.turns[0].tool_calls:
        problems.append("turn-limit check: no tool call on turn 1, so the forced turn after tool use wasn't tested")
    elif turns.stop_reason != "max_turns":
        problems.append(f"turn-limit check: stopped with {turns.stop_reason} instead of max_turns")
    elif turns.final_answer is None:
        problems.append("turn-limit check: no answer on the forced turn")

    tokens = await run_check(llm, server, scenario, model, "tokens", TOKEN_LIMIT)
    if tokens.stop_reason != "token_budget":
        problems.append(f"token-limit check: stopped with {tokens.stop_reason} instead of token_budget")
    elif tokens.final_answer is None:
        problems.append("token-limit check: no answer on the forced turn")

    return problems, turns.totals.agent_cost_usd + tokens.totals.agent_cost_usd


async def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check the forced final-answer turn on each model.")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=MAIN_RUN_MODELS)
    args = parser.parse_args(argv)

    scenarios = read_jsonl(SAMPLE_FILE)
    scenario = next(s for s in scenarios if s.fault_type == FaultType.NONE)
    server = build_server(EpisodeRegistry(scenarios))
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # building the MCP server turned on INFO logging

    failed = []
    total_cost = 0.0
    async with OpenRouterClient() as llm:
        for name in args.models:
            try:
                problems, cost = await check_model(llm, server, scenario, name)
            except LLMError as error:
                problems, cost = [f"API error: {error}"], 0.0
            total_cost += cost
            if problems:
                failed.append(name)
                print(f"FAILED {name:12} " + " | ".join(problems))
            else:
                print(f"ok     {name:12} both forced turns answered | cost ${cost:.5f}")

    print(f"\nTotal cost: ${total_cost:.5f}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
