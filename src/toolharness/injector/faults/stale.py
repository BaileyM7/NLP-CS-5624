"""Stale: an old copy of the answer page, with an outdated fact and an old last_updated date."""
from __future__ import annotations

import datetime as dt

from ...app.backend import ToolResponse
from ..corruptors import to_json
from .silently_wrong import SilentlyWrong

STALE_MIN_YEARS = 3
STALE_MAX_YEARS = 6


class Stale(SilentlyWrong):
    """Apply the question's silently_wrong fact edit and move last_updated 3 to 6 years back."""

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        page = self.edited_page(clean, target)

        # Older than the clean page's own date, so it's old even when that date already is
        days_back = rng.randint(STALE_MIN_YEARS * 365, STALE_MAX_YEARS * 365)
        clean_date = dt.date.fromisoformat(page["last_updated"])
        page["last_updated"] = (clean_date - dt.timedelta(days=days_back)).isoformat()
        return ToolResponse(text=to_json(page))
