"""Shared types for tool backends (the mock app and the fault injector)."""
from __future__ import annotations

import datetime as dt
from typing import Any, Protocol

from pydantic import BaseModel

AS_OF = dt.date(2026, 11, 1)  # fixed "today" for every date in the experiment
CLEAN_DATE_WINDOW_DAYS = 90  # clean undated articles get a date within this many days before AS_OF
PAGE_CHARS = 3000  # max characters per article page
MAX_SEARCH_RESULTS = 10


class ToolResponse(BaseModel):
    """Result of one tool call the JSON text the agent sees."""

    text: str
    is_error: bool = False


class ToolBackend(Protocol):
    """Anything that can answer tool calls by name."""

    def call(self, tool: str, args: dict[str, Any]) -> ToolResponse: ...


class SearchHit(BaseModel):
    title: str
    page: int
    snippet: str


class SearchResult(BaseModel):
    query: str
    hits: list[SearchHit]


class ArticlePage(BaseModel):
    title: str
    page: int
    total_pages: int
    sentences: list[str]
    last_updated: dt.date
    source_id: str


class NotFound(BaseModel):
    """Normal (non-error) result for an unknown title or page."""

    error: str = "not_found"
    title: str
    page: int | None = None
    message: str
