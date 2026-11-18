"""Hard failure: the answer page fetch fails with a realistic API error."""
from __future__ import annotations

from ...app.backend import ToolResponse
from ..corruptors import Corruptor

# Realistic upstream errors; the injector's rng picks one per scenario
ERROR_MESSAGES = [
    "Error 500 Internal Server Error: the article service encountered an unexpected error.",
    "Error 502 Bad Gateway: the article service received an invalid response from upstream.",
    "Error 503 Service Unavailable: the article service is temporarily overloaded.",
    "Error 504 Gateway Timeout: the article service did not respond within 30 seconds.",
    "Error 429 Too Many Requests: rate limit exceeded. Retry after 2 seconds.",
]


class HardFailure(Corruptor):
    """Replace the answer page with an error result (MCP isError: true)."""

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        return ToolResponse(text=rng.choice(ERROR_MESSAGES), is_error=True)
