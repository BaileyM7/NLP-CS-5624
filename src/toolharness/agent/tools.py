"""Tool helpers for the agent: the required "thought" argument, argument parsing and retry detection."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .prompts import THOUGHT_DESCRIPTION

THOUGHT = "thought"


@dataclass(frozen=True)
class ParsedCall:
    thought: str
    args: dict[str, Any] | None  # None = the arguments weren't a JSON object


def add_thought(tools: list[dict]) -> list[dict]:
    """Copies of the tools with a required "thought" string as the first argument."""
    shown = copy.deepcopy(tools)
    for tool in shown:
        params = tool["function"]["parameters"]
        properties = {THOUGHT: {"type": "string", "description": THOUGHT_DESCRIPTION}}
        properties.update(params.get("properties", {}))
        params["properties"] = properties
        params["required"] = [THOUGHT] + [name for name in params.get("required", []) if name != THOUGHT]
    return shown


def split_arguments(raw: str) -> ParsedCall:
    """Parse the model's argument string and take the thought out of it."""
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedCall(thought="", args=None)
    if not isinstance(args, dict):
        return ParsedCall(thought="", args=None)

    thought = args.pop(THOUGHT, "")
    return ParsedCall(thought=str(thought), args=args)


def schema_defaults(tools: list[dict]) -> dict[str, dict[str, Any]]:
    """Each tool's default argument values, e.g. {"get_article": {"page": 1}}."""
    defaults = {}
    for tool in tools:
        function = tool["function"]
        properties = function["parameters"].get("properties", {})
        defaults[function["name"]] = {name: spec["default"] for name, spec in properties.items() if "default" in spec}
    return defaults


def retry_key(name: str, args: dict[str, Any], defaults: dict[str, dict[str, Any]]) -> str:
    """A key that is equal for two calls of the same tool with the same arguments, defaults filled in."""
    filled = {**defaults.get(name, {}), **args}
    return json.dumps([name, filled], sort_keys=True)
