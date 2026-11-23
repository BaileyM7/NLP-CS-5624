"""Batch runs: many scenarios for one model and condition, with resume and a frozen run manifest.

Files under runs/<run-id>/:
    traces/<model>__<condition>.jsonl   finished episodes
    started.jsonl                       one row per episode attempt, written before it runs
    errors.jsonl                        one row per failed attempt
    llm_usage.jsonl                     one row per model call (written by the client)
    manifest.json                       what each model/condition pair was run with
"""
from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx2

from ..data.scenario import FaultType, Scenario
from ..data.scenario import read_jsonl as read_scenarios
from ..eval.trace import Condition, append_jsonl, trace_filename
from ..eval.trace import read_jsonl as read_episodes
from ..harness.checks import HARNESS_CHECKS, HARNESS_CONDITIONS, harness_settings
from ..harness.session import HarnessSession
from ..llm.client import OpenRouterClient
from ..llm.models import get_model
from .prompts import FINAL_TURN_MESSAGE, THOUGHT_DESCRIPTION, system_prompt
from .runner import REASONING, Budget, run_episode
from .session import HttpToolSession, ToolSession

DEFAULT_SERVER_URL = "http://127.0.0.1:8765/mcp"

# Manifest fields that define the experiment; a resumed run must match them exactly
EXPERIMENT_FIELDS = (
    "openrouter_id",
    "condition",
    "system_prompt",
    "thought_description",
    "final_turn_message",
    "reasoning",
    "budget",
    "seed",
    "scenario_hashes",
    "harness",
)


@dataclass
class BatchConfig:
    run_id: str
    model: str
    condition: Condition
    scenario_files: list[Path]
    server_url: str = DEFAULT_SERVER_URL
    questions: int | None = None
    faults: list[FaultType] | None = None
    seed: int = 0
    concurrency: int = 4
    runs_dir: Path = Path("runs")

    @property
    def run_dir(self) -> Path:
        return Path(self.runs_dir) / self.run_id


@dataclass
class BatchSummary:
    ran: int = 0
    skipped: int = 0
    errored: int = 0
    stop_reasons: Counter = field(default_factory=Counter)
    cost_usd: float = 0.0


class ManifestMismatch(Exception):
    """A resumed run would use different experiment settings than the ones it started with."""


def select_scenarios(scenarios: list[Scenario], questions: int | None, faults: list[FaultType] | None) -> list[Scenario]:
    """The first `questions` distinct questions in file order, limited to the chosen fault types."""
    question_ids: list[str] = []
    for scenario in scenarios:
        if scenario.question_id not in question_ids:
            question_ids.append(scenario.question_id)
    if questions is not None:
        question_ids = question_ids[:questions]

    chosen = []
    for scenario in scenarios:
        if scenario.question_id in question_ids and (faults is None or scenario.fault_type in faults):
            chosen.append(scenario)
    return chosen


def episode_id_prefix(run_id: str, model: str, condition: str) -> str:
    """The start shared by every episode ID of one model/condition pair in a run."""
    return f"{run_id}__{model}__{condition}__"


def make_episode_id(run_id: str, model: str, condition: str, scenario_id: str, seed: int, attempt: int) -> str:
    return f"{episode_id_prefix(run_id, model, condition)}{scenario_id}__s{seed}__a{attempt}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> tuple[str, bool | None]:
    """The current commit and whether there are uncommitted changes, or ("unknown", None) outside git."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, bool(status.strip())


def manifest_entry(config: BatchConfig) -> dict:
    """Everything this model/condition pair runs with; only EXPERIMENT_FIELDS are checked on resume."""
    commit, dirty = git_state()
    hashes = {Path(path).as_posix(): file_sha256(path) for path in config.scenario_files}
    return {
        "model": config.model,
        "openrouter_id": get_model(config.model).openrouter_id,
        "condition": config.condition,
        "system_prompt": system_prompt(config.condition),
        "thought_description": THOUGHT_DESCRIPTION,
        "final_turn_message": FINAL_TURN_MESSAGE,
        "reasoning": REASONING,
        "budget": dataclasses.asdict(Budget()),
        "seed": config.seed,
        "harness": harness_settings(config.condition),
        "scenario_hashes": sorted(hashes.values()),  # compared on resume, whatever the paths were called
        "scenario_files": hashes,  # recorded only
        "git_commit": commit,
        "git_dirty": dirty,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def check_manifest(path: Path, entry: dict) -> None:
    """Record the entry on first use; on later runs, refuse if any experiment field changed."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"entries": {}}
    key = f"{entry['model']}__{entry['condition']}"
    existing = data["entries"].get(key)

    if existing is None:
        data["entries"][key] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return

    differing = [name for name in EXPERIMENT_FIELDS if existing.get(name) != entry.get(name)]
    if differing:
        raise ManifestMismatch(
            f"{path}: {key} was started with different {', '.join(differing)}. Use a new --run-id for a changed experiment."
        )


def append_row(path: Path, row: dict) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row) + "\n")


def read_progress(trace_path: Path, started_path: Path, id_prefix: str) -> tuple[set[str], Counter]:
    """Scenario IDs that already have a finished episode, and how many attempts this model/condition started for each."""
    done = set()
    if trace_path.exists():
        done = {episode.scenario_id for episode in read_episodes(trace_path)}

    # started.jsonl is shared by every model/condition in the run; count only this pair's rows
    started: Counter = Counter()
    if started_path.exists():
        for line in started_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["episode_id"].startswith(id_prefix):
                started[row["scenario_id"]] += 1
    return done, started


def health_url(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.endswith("/mcp"):
        base = base[: -len("/mcp")]
    return base + "/health"


def check_server(server_url: str) -> bool:
    """True if the tool server's /health route answers ok."""
    try:
        response = httpx2.get(health_url(server_url), timeout=5)
        return response.status_code == 200 and response.json().get("status") == "ok"
    except (httpx2.HTTPError, ValueError):
        return False


async def run_batch(
    config: BatchConfig,
    llm: OpenRouterClient,
    open_session: Callable[[str, str, int], ToolSession] | None = None,
) -> BatchSummary:
    """Run every selected scenario that doesn't have a finished episode yet."""
    if open_session is None:
        def open_session(episode_id: str, scenario_id: str, seed: int) -> ToolSession:
            return HttpToolSession(config.server_url, episode_id, scenario_id, seed)

    # --questions applies to each file separately, so HotpotQA and LongSeal both get N questions
    selected: list[Scenario] = []
    for path in config.scenario_files:
        selected.extend(select_scenarios(read_scenarios(path), config.questions, config.faults))

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    check_manifest(run_dir / "manifest.json", manifest_entry(config))
    trace_path = run_dir / "traces" / trace_filename(config.model, config.condition)
    started_path = run_dir / "started.jsonl"
    errors_path = run_dir / "errors.jsonl"
    done, started = read_progress(trace_path, started_path, episode_id_prefix(config.run_id, config.model, config.condition))

    to_run = [scenario for scenario in selected if scenario.scenario_id not in done]
    summary = BatchSummary(skipped=len(selected) - len(to_run))
    semaphore = asyncio.Semaphore(config.concurrency)

    # File writes below are synchronous, so concurrent episodes can't interleave them
    async def run_one(scenario: Scenario) -> None:
        async with semaphore:
            # A new attempt number for every start, so an interrupted episode never reuses its ID
            started[scenario.scenario_id] += 1
            attempt = started[scenario.scenario_id]
            episode_id = make_episode_id(config.run_id, config.model, config.condition, scenario.scenario_id, config.seed, attempt)
            append_row(started_path, {"episode_id": episode_id, "scenario_id": scenario.scenario_id, "attempt": attempt})

            # Any failure ends only this episode: it is logged and retried on the next run
            try:
                session = open_session(episode_id, scenario.scenario_id, config.seed)
                # Harness conditions check every tool response before the model sees it
                if config.condition in HARNESS_CONDITIONS:
                    session = HarnessSession(session, HARNESS_CHECKS)
                async with session:
                    episode = await run_episode(llm, session, scenario, config.model, config.condition, episode_id)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                row = {"episode_id": episode_id, "scenario_id": scenario.scenario_id, "attempt": attempt, "error": message}
                append_row(errors_path, row)
                summary.errored += 1
                return

            append_jsonl(episode, trace_path)
            summary.ran += 1
            summary.stop_reasons[episode.stop_reason] += 1
            summary.cost_usd += episode.totals.agent_cost_usd

    await asyncio.gather(*(run_one(scenario) for scenario in to_run))
    return summary
