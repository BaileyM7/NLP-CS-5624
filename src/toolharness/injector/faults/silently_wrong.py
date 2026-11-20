"""Silently wrong: the answer page looks normal, but one fact is swapped for a plausible wrong one."""
from __future__ import annotations

import json

from ...app.backend import ToolResponse
from ...data.fact_edits import FactEdit, apply_edit, edit_key
from ..corruptors import Corruptor, to_json


class SilentlyWrong(Corruptor):
    """Apply this answer page's pre-generated fact edit (see data/fact_edits.py)."""

    def __init__(self, edits: list[FactEdit]):
        self.edits = {edit_key(e): e for e in edits}

    def should_trigger(self, call, clean, history, target) -> bool:
        """Fire on the first answer page with a real edit; skipped pages are served clean."""
        if call.tool != "get_article" or target is None:
            return False
        return not self.edits[edit_key(target)].skip_reason

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        edit = self.edits[edit_key(target)]
        page = json.loads(clean.text)
        page["sentences"] = apply_edit(page["sentences"], edit)
        return ToolResponse(text=to_json(page))
