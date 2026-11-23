"""HarnessSession: wraps an episode's tool session and adds a _harness line to every tool response."""
from __future__ import annotations

import json
import time
from typing import Any

from ..agent.session import ToolResult, ToolSession
from .checks import LINE_PREFIX, Check


def harness_line(values: dict[str, bool | None], flags: list[str]) -> str:
    """The line the model sees: each check's value in order, then all flags."""
    block = dict(values)
    block["flags"] = flags
    return LINE_PREFIX + json.dumps(block)


class HarnessSession:
    """Runs the checks on each tool response before the model sees it. Never makes tool calls of its own."""

    def __init__(self, inner: ToolSession, checks: list[Check]):
        self.inner = inner
        self.checks = checks
        self.history: list[tuple[str, ToolResult]] = []  # this episode's earlier (tool, raw response) pairs

    async def __aenter__(self):
        await self.inner.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.inner.__aexit__(*exc)

    async def list_tools(self) -> list[dict]:
        return await self.inner.list_tools()

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        raw = await self.inner.call(name, args)

        # Run every check on the raw response and time them together
        start = time.perf_counter()
        values: dict[str, bool | None] = {}
        flags: list[str] = []
        for check in self.checks:
            result = check.run(name, raw, list(self.history))
            values[check.key] = result.value
            flags.extend(result.flags)
        latency_ms = (time.perf_counter() - start) * 1000

        self.history.append((name, raw))
        text = raw.text + "\n\n" + harness_line(values, flags)
        return ToolResult(text=text, is_error=raw.is_error, harness_flags=flags, proxy_latency_ms=latency_ms)
