"""Harness checks: what each check returns, which checks run, and which conditions use them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..agent.session import ToolResult

# Conditions whose tool responses go through the harness layer
HARNESS_CONDITIONS = ("harness", "flags_only")

LINE_PREFIX = "_harness: "


@dataclass(frozen=True)
class CheckResult:
    value: bool | None  # True = passed, False = failed, None = the check doesn't apply to this response
    flags: list[str]  # short reasons shown to the model; empty when passed


class Check(Protocol):
    key: str  # the check's key in the _harness line, e.g. "schema_valid"

    def run(self, tool: str, result: ToolResult, history: list[tuple[str, ToolResult]]) -> CheckResult:
        """Check one raw tool response; history holds this episode's earlier (tool, raw response) pairs."""
        ...


# The checks the harness runs, in the order they appear in the _harness line
HARNESS_CHECKS: list[Check] = []


def harness_settings(condition: str) -> dict | None:
    """The harness settings recorded in the run manifest; None for conditions without the harness."""
    if condition not in HARNESS_CONDITIONS:
        return None
    return {"checks": [check.key for check in HARNESS_CHECKS], "line_prefix": LINE_PREFIX}
