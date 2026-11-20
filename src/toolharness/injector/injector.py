"""Fault injector: wraps a backend and corrupts at most one tool response per episode."""
from __future__ import annotations

import random
from typing import Any

from ..app.backend import ToolResponse
from ..app.mock_app import MockApp
from ..data.scenario import Scenario
from .corruptors import AnswerTarget, CallRecord, Corruptor, ToolCall, article_page_key
from .ground_truth import GroundTruthLog


class FaultInjector:
    """One episode's tool stack. Implements ToolBackend.

    The fault is transient: it fires on the first call the corruptor triggers on,
    and every later call (including an identical retry) gets clean data.
    """

    def __init__(
        self,
        inner: MockApp,
        scenario: Scenario,
        corruptor: Corruptor,
        seed: int,
        episode_id: str,
        gt_log: GroundTruthLog | None = None,
    ):
        self.inner = inner
        self.scenario = scenario
        self.corruptor = corruptor
        self.episode_id = episode_id
        self.gt_log = gt_log
        # Seeded by scenario, not episode: every model and condition gets the same corruption
        self.rng = random.Random(f"{seed}:{scenario.scenario_id}")
        self.answer_pages = inner.answer_pages()
        self.history: list[CallRecord] = []
        self.corrupted_record: CallRecord | None = None

    def call(self, tool: str, args: dict[str, Any]) -> ToolResponse:
        call = ToolCall(tool=tool, args=dict(args))
        clean = self.inner.call(tool, args)

        # A retry is the same tool returning the same clean data as the corrupted call
        retry = (
            self.corrupted_record is not None
            and self.corrupted_record.call.tool == tool
            and self.corrupted_record.clean.text == clean.text
        )

        corrupted = False
        returned = clean
        target = self._answer_target(clean)
        if self.corrupted_record is None and self.corruptor.should_trigger(call, clean, self.history, target):
            returned = self.corruptor.corrupt(call, clean, self.history, self.rng, target)
            corrupted = True

        record = CallRecord(
            index=len(self.history),
            call=call,
            clean=clean,
            returned=returned,
            corrupted=corrupted,
            retry_of_corrupted=retry,
        )
        self.history.append(record)
        if corrupted:
            self.corrupted_record = record
        self._log(record)
        return returned

    def _answer_target(self, clean: ToolResponse) -> AnswerTarget | None:
        """Describe the answer page in this response, or None if it isn't one."""
        key = article_page_key(clean)
        if key is None or key not in self.answer_pages:
            return None
        title, page = key
        return AnswerTarget(
            scenario_id=self.scenario.scenario_id,
            title=title,
            page=page,
            sentence_positions=self.inner.supporting_positions(title, page),
            gold_answer=self.scenario.gold_answer,
        )

    def _log(self, record: CallRecord) -> None:
        """Write this call to the ground-truth log; response text is kept only for the corrupted call."""
        if self.gt_log is None:
            return
        self.gt_log.write(
            {
                "episode_id": self.episode_id,
                "scenario_id": self.scenario.scenario_id,
                "fault_type": self.scenario.fault_type.value,
                "call_index": record.index,
                "tool": record.call.tool,
                "args": record.call.args,
                "corrupted": record.corrupted,
                "retry_of_corrupted": record.retry_of_corrupted,
                "original": record.clean.text if record.corrupted else None,
                "corrupted_text": record.returned.text if record.corrupted else None,
                "corrupted_is_error": record.returned.is_error if record.corrupted else None,
            }
        )
