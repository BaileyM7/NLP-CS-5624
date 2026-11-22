"""Run a batch of agent episodes against the running tool server.

Usage:
    python -m toolharness.agent --run-id pilot-1 --model haiku-5.5 --condition baseline \
        --scenarios data/scenarios/hotpotqa.jsonl --questions 5 --faults none hard_failure malformed
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import get_args

from ..data.scenario import FaultType
from ..eval.trace import Condition
from ..llm.client import LLMError, OpenRouterClient
from ..llm.models import MODELS
from .batch import DEFAULT_SERVER_URL, BatchConfig, BatchSummary, ManifestMismatch, check_server, health_url, run_batch
from .prompts import ConditionNotReady, require_ready


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agent episodes for one model and condition.")
    parser.add_argument("--run-id", required=True, help="names the folder runs/<run-id>/")
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--condition", required=True, choices=list(get_args(Condition)))
    parser.add_argument("--scenarios", required=True, nargs="+", type=Path, help="scenario JSONL files")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--questions", type=int, help="only the first N questions of each scenario file")
    parser.add_argument("--faults", nargs="+", choices=[fault.value for fault in FaultType], help="only these fault types")
    parser.add_argument("--seed", type=int, default=0, help="fault-injector seed")
    parser.add_argument("--concurrency", type=int, default=4, help="episodes run at the same time")
    return parser.parse_args(argv)


async def run(config: BatchConfig) -> BatchSummary:
    async with OpenRouterClient(usage_log=config.run_dir / "llm_usage.jsonl") as llm:
        return await run_batch(config, llm)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        require_ready(args.condition)
    except ConditionNotReady as error:
        print(error)
        return 2
    if not check_server(args.server_url):
        print(f"Tool server not reachable at {health_url(args.server_url)}. Start it first (see docs/tool-server.md).")
        return 2

    config = BatchConfig(
        run_id=args.run_id,
        model=args.model,
        condition=args.condition,
        scenario_files=args.scenarios,
        server_url=args.server_url,
        questions=args.questions,
        faults=[FaultType(value) for value in args.faults] if args.faults else None,
        seed=args.seed,
        concurrency=args.concurrency,
    )
    try:
        summary = asyncio.run(run(config))
    except (ManifestMismatch, LLMError) as error:
        print(error)
        return 2

    print(f"ran {summary.ran}, skipped {summary.skipped} (already done), errored {summary.errored}")
    print("stop reasons: " + ", ".join(f"{reason} {count}" for reason, count in summary.stop_reasons.most_common()))
    print(f"agent cost ${summary.cost_usd:.5f}")
    if summary.errored:
        print(f"Errors are in {config.run_dir / 'errors.jsonl'}. Rerun the same command to retry failed episodes.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
