"""Run one stage of the #15 pilot (paid). Start the tool server first; see docs/pilot.md.

Stages, in order:
    smoke     run pilot-smoke: Haiku and Luna, 1 question per scenario file, the 4 non-verifier conditions (about a dime)
    cheap     run pilot-2: Haiku and Luna, 5 questions per scenario file, the 4 non-verifier conditions
    verifier  run pilot-2: Haiku, the same 5 questions per scenario file, llm_verifier (Luna is projected from it)
    big       run pilot-2: Opus and Astra, 2 HotpotQA questions, baseline and harness

LongSeal leaves out contradiction (it almost never fires there), and GPT models, plus every llm_verifier
episode, run one at a time, because OpenRouter rate-limits GPT models.

Usage:
    python scripts/run_pilot.py smoke --dry-run   # print the commands only
    python scripts/run_pilot.py smoke
    python scripts/run_pilot.py cheap --run-id pilot-3 --server-url http://127.0.0.1:8766/mcp
"""
from __future__ import annotations

import argparse
import subprocess
import sys

HOTPOTQA = "data/scenarios/hotpotqa.jsonl"
LONGSEAL = "data/scenarios/longseal.jsonl"
FOUR_CONDITIONS = ["baseline", "verify_prompt", "harness", "flags_only"]
CHEAP_MODELS = ["haiku-5.5", "gpt-6-luna"]
LONGSEAL_SKIP = ["longseal:contradiction"]

# stage -> (run ID, models, conditions, scenario files, questions per file, --skip values)
STAGES = {
    "smoke": ("pilot-smoke", CHEAP_MODELS, FOUR_CONDITIONS, [HOTPOTQA, LONGSEAL], 1, LONGSEAL_SKIP),
    "cheap": ("pilot-2", CHEAP_MODELS, FOUR_CONDITIONS, [HOTPOTQA, LONGSEAL], 5, LONGSEAL_SKIP),
    "verifier": ("pilot-2", ["haiku-5.5"], ["llm_verifier"], [HOTPOTQA, LONGSEAL], 5, LONGSEAL_SKIP),
    "big": ("pilot-2", ["opus-5.5", "gpt-6-astra"], ["baseline", "harness"], [HOTPOTQA], 2, []),
}


def default_concurrency(model: str, condition: str) -> int | None:
    """1 for anything that calls a GPT model (the agent, or the llm_verifier's verifier); None keeps the runner's default."""
    if model.startswith("gpt") or condition == "llm_verifier":
        return 1
    return None


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
    default_run_id, models, conditions, scenario_files, questions, skips = STAGES[stage]
    run_id = run_id or default_run_id

    commands = []
    for model in models:
        for condition in conditions:
            command = [python, "-m", "toolharness.agent", "--run-id", run_id, "--model", model, "--condition", condition]
            command += ["--scenarios", *scenario_files, "--questions", str(questions)]
            for skip in skips:
                command += ["--skip", skip]
            if server_url:
                command += ["--server-url", server_url]
            episodes_at_once = concurrency or default_concurrency(model, condition)
            if episodes_at_once:
                command += ["--concurrency", str(episodes_at_once)]
            commands.append(command)
    return commands


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stage of the pilot.")
    parser.add_argument("stage", choices=list(STAGES))
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    parser.add_argument("--run-id", help="use another run ID, e.g. after a frozen prompt or harness change")
    parser.add_argument("--server-url", help="tool server URL, if not the default http://127.0.0.1:8765/mcp")
    parser.add_argument("--concurrency", type=int, help="episodes at once for every command (default: 1 for GPT models and llm_verifier, else 4)")
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
