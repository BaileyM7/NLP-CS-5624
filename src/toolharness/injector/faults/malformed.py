"""Malformed output: the answer page comes back broken, with its supporting sentences lost."""
from __future__ import annotations

import json
import random

from ...app.backend import ToolResponse
from ..corruptors import AnswerTarget, Corruptor

VARIANTS = ["invalid_json", "missing_field", "wrong_type", "truncated"]

# Served as a normal result, like a proxy that returns its error page with status 200
HTML_ERROR_PAGE = (
    "<!DOCTYPE html><html><head><title>503 Service Unavailable</title></head><body>"
    "<h1>Service Unavailable</h1>"
    "<p>The server is temporarily unable to service your request due to maintenance downtime "
    "or capacity problems. Please try again later.</p>"
    "</body></html>"
)

CUT_MARK = "￿"  # marks the truncation point; never appears in article text


def to_json(data: dict) -> str:
    """Serialize the same way the mock app does: compact, non-ASCII kept as is."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def truncate_in_target(data: dict, target: AnswerTarget, rng: random.Random) -> str:
    """Cut the response partway into the first supporting sentence, before the answer appears."""
    position = target.sentence_positions[0]
    sentence = data["sentences"][position]

    # Keep at most half the sentence, and stop before the answer if the sentence contains it
    limit = len(sentence) // 2
    answer_at = sentence.lower().find(target.gold_answer.lower())
    if answer_at != -1:
        limit = min(limit, answer_at)
    kept = sentence[: rng.randint(0, limit)]

    # Serialize with a mark where the cut goes, then drop everything from the mark on
    data["sentences"] = data["sentences"][:position] + [kept + CUT_MARK]
    text = to_json(data)
    return text[: text.index(CUT_MARK)]


def break_page(variant: str, clean_text: str, target: AnswerTarget, rng: random.Random) -> str:
    """Return the answer page broken in the given way. No variant keeps a full supporting sentence."""
    data = json.loads(clean_text)
    if variant == "invalid_json":
        return HTML_ERROR_PAGE
    if variant == "missing_field":
        del data["sentences"]
        return to_json(data)
    if variant == "wrong_type":
        data["sentences"] = len(data["sentences"])  # a count where the list should be
        return to_json(data)
    if variant == "truncated":
        return truncate_in_target(data, target, rng)
    raise ValueError(f"unknown malformed variant {variant!r}")


class Malformed(Corruptor):
    """Replace the answer page with a broken payload (variant picked by rng), returned as a normal result."""

    def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
        variant = rng.choice(VARIANTS)
        return ToolResponse(text=break_page(variant, clean.text, target, rng))
