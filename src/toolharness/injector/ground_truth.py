"""Ground-truth log: what the injector did on every agent call. Never shown to the agent or proxy."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path


class GroundTruthLog:
    """Appends one JSON row per agent tool call. Safe to share across episodes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


@dataclass
class EpisodeSummary:
    episode_id: str
    scenario_id: str
    fault_type: str
    calls: int
    fault_injected: bool
    injected_at_call: int | None  # index of the corrupted call, if any
    retries_of_corrupted: int


def summarize_gt(path: str | Path) -> dict[str, EpisodeSummary]:
    """Per-episode summary of a ground-truth log, keyed by episode ID."""
    summaries: dict[str, EpisodeSummary] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            summary = summaries.get(row["episode_id"])
            if summary is None:
                summary = EpisodeSummary(
                    episode_id=row["episode_id"],
                    scenario_id=row["scenario_id"],
                    fault_type=row["fault_type"],
                    calls=0,
                    fault_injected=False,
                    injected_at_call=None,
                    retries_of_corrupted=0,
                )
                summaries[row["episode_id"]] = summary

            summary.calls += 1
            if row["corrupted"]:
                summary.fault_injected = True
                summary.injected_at_call = row["call_index"]
            if row["retry_of_corrupted"]:
                summary.retries_of_corrupted += 1
    return summaries
