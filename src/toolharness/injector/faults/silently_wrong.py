"""Silently wrong: the answer page looks normal, but one fact is swapped for a plausible wrong one."""
from __future__ import annotations

import json

from ...app.backend import ToolResponse
from ...data.fact_edits import FactEdit, apply_edit, edit_key
from ...data.scenario import FaultType, fault_variant_id
from ..corruptors import AnswerTarget, Corruptor, to_json


class SilentlyWrong(Corruptor):
    """Apply this answer page's pre-generated fact edit (see data/fact_edits.py)."""

    def __init__(self, edits: list[FactEdit]):
        self.edits = {edit_key(e): e for e in edits}

    def edit_for(self, target: AnswerTarget) -> FactEdit:
        """This answer page's edit; stale and contradiction scenarios use their question's silently_wrong edit."""
        scenario_id = fault_variant_id(target.scenario_id, FaultType.SILENTLY_WRONG)
        return self.edits[(scenario_id, target.title, target.page)]

    def edited_page(self, clean: ToolResponse, target: AnswerTarget) -> dict:
        """The clean page with this answer page's fact edit applied."""
        page = json.loads(clean.text)
        page["sentences"] = apply_edit(page["sentences"], self.edit_for(target))
        return page

    def should_trigger(self, call, clean, history, target) -> bool:
        """Fire on the first answer page with a real edit; skipped pages are served clean."""
        if call.tool != "get_article" or target is None:
            return False
        return not self.edit_for(target).skip_reason

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        return ToolResponse(text=to_json(self.edited_page(clean, target)))
