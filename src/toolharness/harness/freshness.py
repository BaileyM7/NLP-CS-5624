"""Freshness check (#19): is an article page's last_updated date too old?"""
from __future__ import annotations

from pydantic import ValidationError

from ..agent.session import ToolResult
from ..app.backend import AS_OF, ArticlePage
from .base import CheckResult

FRESHNESS_DAYS = 365  # a page last updated more than this many days before AS_OF is flagged as stale


class FreshnessCheck:
    """Flags article pages whose last_updated date is more than a year before today (AS_OF)."""

    key = "fresh"

    def run(self, tool: str, result: ToolResult, history: list[tuple[str, ToolResult]]) -> CheckResult:
        # Only a valid article page has a date to judge
        if tool != "get_article" or result.is_error:
            return CheckResult(value=None, flags=[])
        try:
            page = ArticlePage.model_validate_json(result.text)
        except ValidationError:
            return CheckResult(value=None, flags=[])

        if (AS_OF - page.last_updated).days > FRESHNESS_DAYS:
            flag = f"stale: last updated {page.last_updated.isoformat()}, more than 1 year before today ({AS_OF.isoformat()})"
            return CheckResult(value=False, flags=[flag])
        return CheckResult(value=True, flags=[])
