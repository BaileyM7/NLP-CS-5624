"""Pydantic models and JSONL helpers for agent episode traces.

Ground-truth injection details belong in the injector's separate log. These
models intentionally have no fields for whether/where a fault was injected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from toolharness.data.scenario import Dataset, FaultType

Condition = Literal["baseline", "verify_prompt", "llm_verifier", "harness", "flags_only"]


class TraceModel(BaseModel):
    """Base configuration shared by the persisted trace models."""

    model_config = ConfigDict(extra="forbid")


class TokenUsage(TraceModel):
    """Token counts for a single model role."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class TurnTotals(TraceModel):
    """Agent usage during one model response cycle."""

    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)


class EpisodeTotals(TraceModel):
    """Episode totals; verifier usage is kept separate from agent usage."""

    agent_tokens: TokenUsage = Field(default_factory=TokenUsage)
    agent_cost_usd: float = Field(default=0.0, ge=0)
    verifier_tokens: TokenUsage = Field(default_factory=TokenUsage)
    verifier_cost_usd: float = Field(default=0.0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)


class VerifierOutput(TraceModel):
    suspicious: bool
    fault_guess: str | None = None
    reason: str


class ToolCallTrace(TraceModel):
    """One tool call and the exact response exposed to the agent."""

    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    raw_tool_response: str
    is_error: bool = False
    # Null when this condition has no proxy flags (including baseline).
    harness_flags: list[str] | None = None
    verifier_output: VerifierOutput | None = None
    verifier_tokens: TokenUsage | None = None
    verifier_cost_usd: float | None = Field(default=None, ge=0)
    verifier_latency_ms: float | None = Field(default=None, ge=0)
    proxy_latency_ms: float | None = Field(default=None, ge=0)
    repeats_earlier_call: bool = False


class Turn(TraceModel):
    """One-based model response cycle, which may contain multiple tool calls."""

    index: int = Field(ge=1)
    # Only text emitted to the conversation is stored; private reasoning is not.
    assistant_text: str = ""
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    totals: TurnTotals = Field(default_factory=TurnTotals)


class Episode(TraceModel):
    """One model answering one scenario in one experimental condition."""

    episode_id: str
    scenario_id: str
    dataset: Dataset
    model: str
    condition: Condition
    fault_type: FaultType
    question: str
    final_answer: str | None = None
    gold_answer: str
    stop_reason: Literal["final_answer", "max_turns", "retry_budget", "token_budget", "error"] | None = None
    totals: EpisodeTotals = Field(default_factory=EpisodeTotals)
    turns: list[Turn] = Field(default_factory=list)


def trace_filename(model: str, condition: Condition) -> str:
    """Return a filesystem-safe JSONL name for one model/condition pair."""
    if not model:
        raise ValueError("model must not be empty")
    slug = quote(model, safe="._-")
    return f"{slug}__{condition}.jsonl"


def write_jsonl(episodes: Iterable[Episode], path: str | Path) -> int:
    """Write one model/condition partition to JSONL and return its size."""
    episodes = list(episodes)
    _validate_single_partition(episodes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for episode in episodes:
            stream.write(episode.model_dump_json() + "\n")
            count += 1
    return count


def append_jsonl(episode: Episode, path: str | Path) -> None:
    """Append to a JSONL file only if it contains the same model/condition."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        existing = read_jsonl(path)
        if existing and _partition(existing[0]) != _partition(episode):
            raise ValueError(
                "cannot append an episode from a different model/condition "
                f"to {path}: existing={_partition(existing[0])}, "
                f"new={_partition(episode)}"
            )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(episode.model_dump_json() + "\n")


def _partition(episode: Episode) -> tuple[str, str]:
    return episode.model, episode.condition


def _validate_single_partition(episodes: list[Episode]) -> None:
    if not episodes:
        return
    expected = _partition(episodes[0])
    for episode in episodes[1:]:
        if _partition(episode) != expected:
            raise ValueError(
                "a JSONL trace file must contain one model/condition pair; "
                f"expected={expected}, got={_partition(episode)}"
            )


def read_jsonl(path: str | Path) -> list[Episode]:
    """Read and validate an episode JSONL file."""
    episodes: list[Episode] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                episodes.append(Episode.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid episode JSON on line {line_number} of {path}: {error}") from error
    return episodes
