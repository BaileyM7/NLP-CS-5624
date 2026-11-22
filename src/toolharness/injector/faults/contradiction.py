"""Contradiction: an answer page that conflicts with what the agent was already shown of that same page."""
from __future__ import annotations

import json

from ...data.fact_edits import span_pattern
from ..corruptors import CallRecord, article_page_key
from .silently_wrong import SilentlyWrong


def shown_text(record: CallRecord, page: tuple[str, int]) -> list[str]:
    """What an earlier call showed the agent of this (title, page): its search snippet, or its sentences."""
    try:
        data = json.loads(record.returned.text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    # A search result: the snippet of the hit for this page
    if "hits" in data:
        snippets = []
        for hit in data["hits"]:
            if (hit["title"], hit["page"]) == page:
                snippets.append(hit["snippet"])
        return snippets

    # An earlier visit to this page: its sentences
    if article_page_key(record.returned) == page:
        return data["sentences"]
    return []


class Contradiction(SilentlyWrong):
    """Apply the question's silently_wrong fact edit, but only once the agent has seen this page's clean fact."""

    def should_trigger(self, call, clean, history, target) -> bool:
        """Fire on an answer page with a real edit whose clean fact an earlier snippet or visit of it showed."""
        if not super().should_trigger(call, clean, history, target):
            return False

        pattern = span_pattern(self.edit_for(target).original_value)
        for record in history:
            for text in shown_text(record, (target.title, target.page)):
                if pattern.search(text):
                    return True
        return False
