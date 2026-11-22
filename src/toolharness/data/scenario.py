"""Scenario model: one row = one question paired with one fault condition."""
from __future__ import annotations

import datetime as dt
import json
from enum import Enum
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

Dataset = Literal["hotpotqa", "longseal"]
DEFAULT_TOOLS = ["search", "get_article"]


class FaultType(str, Enum):
    NONE = "none"  # clean control episode
    HARD_FAILURE = "hard_failure"
    MALFORMED = "malformed"
    SILENTLY_WRONG = "silently_wrong"
    STALE = "stale"
    CONTRADICTION = "contradiction"


class Document(BaseModel):
    """One article the mock app can serve."""

    model_config = ConfigDict(frozen=True)

    title: str
    sentences: list[str]
    last_updated: dt.date | None = None  # None = the mock app assigns a date


class SupportingFact(BaseModel):
    """A sentence the answer depends on: (article title, sentence index)."""

    model_config = ConfigDict(frozen=True)

    title: str
    sent_id: int


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    scenario_id: str  # unique per (dataset, question, fault)
    question_id: str  # shared by all fault variants of one question
    dataset: Dataset
    question: str
    gold_answer: str
    tools: list[str] = Field(default_factory=lambda: list(DEFAULT_TOOLS))
    fault_type: FaultType
    fault_target_titles: list[str]  # articles the injector may corrupt
    supporting_facts: list[SupportingFact]
    context: list[Document]
    metadata: dict[str, str] = Field(default_factory=dict)  # dataset labels for analysis


def make_scenario_id(dataset: str, question_id: str, fault: FaultType) -> str:
    return f"{dataset}-{question_id}-{fault.value}"


def fault_variant_id(scenario_id: str, fault: FaultType) -> str:
    """The ID of the same question's scenario under another fault type."""
    question_part = scenario_id.rsplit("-", 1)[0]  # fault values contain no "-"
    return f"{question_part}-{fault.value}"


def write_jsonl(scenarios: Iterable[Scenario], path: str | Path) -> int:
    """Write one scenario per line and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for scenario in scenarios:
            f.write(scenario.model_dump_json() + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[Scenario]:
    """Read scenarios written by write_jsonl."""
    scenarios = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(Scenario.model_validate(json.loads(line)))
    return scenarios
