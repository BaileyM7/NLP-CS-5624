"""What every harness check implements and returns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..agent.session import ToolResult


@dataclass(frozen=True)
class CheckResult:
    value: bool | None  # True = passed, False = failed, None = the check doesn't apply to this response
    flags: list[str]  # short reasons shown to the model; empty when passed


class Check(Protocol):
    key: str  # the check's key in the _harness line, e.g. "schema_valid"

    def run(self, tool: str, result: ToolResult, history: list[tuple[str, ToolResult]]) -> CheckResult:
        """Check one raw tool response; history holds this episode's earlier (tool, raw response) pairs."""
        ...
