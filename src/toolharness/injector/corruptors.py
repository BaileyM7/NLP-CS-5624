"""Corruptor base class: one subclass per fault type decides when to fire and how to corrupt."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from ..app.backend import ToolResponse


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class CallRecord:
    """One agent tool call as the injector saw it."""

    index: int
    call: ToolCall
    clean: ToolResponse  # what the mock app returned
    returned: ToolResponse  # what the agent received
    corrupted: bool
    retry_of_corrupted: bool


@dataclass(frozen=True)
class AnswerTarget:
    """The answer page being corrupted and where its supporting sentences are."""

    scenario_id: str
    title: str
    page: int
    sentence_positions: list[int]  # indexes into the page's "sentences" list
    gold_answer: str


def to_json(data: dict) -> str:
    """Serialize a response the same way the mock app does: compact, non-ASCII kept as is."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def article_page_key(response: ToolResponse) -> tuple[str, int] | None:
    """(title, page) if the response is an article page, else None."""
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "sentences" in data and "title" in data and "page" in data:
        return data["title"], data["page"]
    return None


class Corruptor:
    """Base class for a fault type. Subclasses implement corrupt() and may override should_trigger()."""

    def should_trigger(
        self,
        call: ToolCall,
        clean: ToolResponse,
        history: list[CallRecord],
        target: AnswerTarget | None,
    ) -> bool:
        """Fire on a get_article call that returns an answer page (target is set only for those)."""
        return call.tool == "get_article" and target is not None

    def corrupt(
        self,
        call: ToolCall,
        clean: ToolResponse,
        history: list[CallRecord],
        rng: random.Random,
        target: AnswerTarget | None,
    ) -> ToolResponse:
        """Return the corrupted response. Use rng for every random choice.

        target is set when the clean response is an answer page, else None.
        """
        raise NotImplementedError


class NoFault(Corruptor):
    """Clean control episodes: never fires."""

    def should_trigger(self, call, clean, history, target) -> bool:
        return False

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        return clean
