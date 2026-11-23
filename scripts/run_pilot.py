"""Run one stage of the #15 pilot (paid). Start the tool server first; see docs/pilot.md.

Stages, in order:
    smoke  run pilot-smoke: Haiku and Luna, 1 question per scenario file, all 4 conditions (about a dime)
    cheap  run pilot-1: Haiku and Luna, 5 questions per scenario file, all 4 conditions
    big    run pilot-1: Opus and Astra, 2 HotpotQA questions, baseline and harness

Usage:
    python scripts/run_pilot.py smoke --dry-run   # print the commands only
    python scripts/run_pilot.py smoke
    python scripts/run_pilot.py smoke --run-id pilot-smoke-2 --server-url http://127.0.0.1:8766/mcp
"""
from __future__ import annotations

import argparse
import subprocess
import sys

HOTPOTQA = "data/scenarios/hotpotqa.jsonl"
LONGSEAL = "data/scenarios/longseal.jsonl"
FOUR_CONDITIONS = ["baseline", "verify_prompt", "harness", "flags_only"]
CHEAP_MODELS = ["haiku-5.5", "gpt-6-luna"]

# stage -> (run ID, models, conditions, scenario files, questions per file)
STAGES = {
    "smoke": ("pilot-smoke", CHEAP_MODELS, FOUR_CONDITIONS, [HOTPOTQA, LONGSEAL], 1),
    "cheap": ("pilot-1", CHEAP_MODELS, FOUR_CONDITIONS, [HOTPOTQA, LONGSEAL], 5),
    "big": ("pilot-1", ["opus-5.5", "gpt-6-astra"], ["baseline", "harness"], [HOTPOTQA], 2),
}


def stage_commands(
    stage: str,
    python: str,
    run_id: str | None = None,
    server_url: str | None = None,
    concurrency: int | None = None,
) -> list[list[str]]:
    """One batch command per model and condition in the stage. The keyword arguments override the defaults."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    default_run_id, models, conditions, scenario_files, questions = STAGES[stage]
    run_id = run_id or default_run_id

    commands = []
    for model in models:
        for condition in conditions:
            command = [python, "-m", "toolharness.agent", "--run-id", run_id, "--model", model, "--condition", condition]
            command += ["--scenarios", *scenario_files, "--questions", str(questions)]
            if server_url:
                command += ["--server-url", server_url]
            if concurrency:
                command += ["--concurrency", str(concurrency)]
            commands.append(command)
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage of the pilot.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--run-id", help="use another run ID, e.g. after a frozen prompt or harness change")
    parser.add_argument("--server-url", help="tool server URL, if not the default http://127.0.0.1:8765/mcp")
    parser.add_argument("--concurrency", type=int, help="episodes at once (default 4); use 1 if GPT models hit HTTP 429")
    args = parser.parse_args(argv)

    commands = stage_commands(args.stage, sys.executable, args.run_id, args.server_url, args.concurrency)
    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    # Stop at the first command that fails, so problems are seen before more money is spent
    for command in commands:
        print("\n$ " + " ".join(command), flush=True)
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Stopped: that command exited with code {result.returncode}.")
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
