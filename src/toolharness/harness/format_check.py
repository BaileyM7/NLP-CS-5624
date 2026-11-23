"""Format check (#18): does a tool response match the tool's own output schema?"""
from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from ..agent.session import ToolResult
from ..app.backend import ArticlePage, NotFound, SearchResult
from .base import CheckResult


def failed(reason: str) -> CheckResult:
    return CheckResult(value=False, flags=[f"format: {reason}"])


def output_model(tool: str, data: dict) -> type[BaseModel]:
    """The output model a response should match; get_article returns a page or a not-found result."""
    if tool == "search":
        return SearchResult
    if data.get("error") == "not_found":
        return NotFound
    return ArticlePage


class FormatCheck:
    """Flags error results and responses that don't validate as the tool's output model."""

    key = "schema_valid"

    def run(self, tool: str, result: ToolResult, history: list[tuple[str, ToolResult]]) -> CheckResult:
        if tool not in ("search", "get_article"):
            return CheckResult(value=None, flags=[])
        if result.is_error:
            return failed("the tool returned an error")

        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            return failed("not valid JSON")
        if not isinstance(data, dict):
            return failed("expected a JSON object")

        # The first validation error decides the reason: a missing field, or a field of the wrong type
        try:
            output_model(tool, data).model_validate_json(result.text)
        except ValidationError as error:
            first = error.errors()[0]
            field = ".".join(str(part) for part in first["loc"])
            if first["type"] == "missing":
                return failed(f"missing field '{field}'")
            return failed(f"field '{field}' has the wrong type")
        return CheckResult(value=True, flags=[])
