"""The agent's fixed prompt text, and which conditions can run yet.

This text is part of the experiment: changing it changes the results, and the run manifest records it.
"""
from __future__ import annotations

from ..app.backend import AS_OF
from ..eval.trace import Condition

DATE_LINE = f"Today's date is {AS_OF:%B} {AS_OF.day}, {AS_OF.year}."

# Used by baseline, llm_verifier and flags_only; never mentions faults or unreliable tools
BASE_PROMPT = f"""You answer questions by looking things up with the search and get_article tools.
{DATE_LINE}

Use the tools to look up the facts you need before answering. Search results only show short excerpts, so open the articles you rely on with get_article and base your answer on what they say. Every tool call has a "thought" argument: in one or two sentences, say what you concluded from the previous result and why you are making this call.

When you are ready to answer, reply without calling a tool and end your reply with:
Final Answer: <short answer>
Keep the short answer as short as possible: a name, a date, a number, or yes/no, with no explanation."""

# Added to BASE_PROMPT for verify_prompt and harness: check each tool result before relying on it (#21)
VERIFY_INSTRUCTION = (
    "Tool results can be wrong. Before you rely on a result, check it: is it an error, garbled or incomplete, "
    "out of date, or in conflict with something you saw earlier? Start each thought with that check in one short "
    "sentence, then say what you will do next. If a result looks unreliable, don't build on it: fetch it again or "
    "confirm it from another source first. Some results end with automatic _harness checks; treat their flags as "
    "hints to investigate, not as proof."
)

THOUGHT_DESCRIPTION = "One or two sentences: what you concluded from the previous result, and why you are making this call."

FINAL_TURN_MESSAGE = "You have reached the limit for this question. Give your final answer now, ending with 'Final Answer: <short answer>'."

# Conditions that can run today, and what each of the others still needs
READY = {"baseline", "verify_prompt", "llm_verifier", "harness", "flags_only"}
WAITING_ON: dict[str, str] = {}


class ConditionNotReady(Exception):
    """The condition depends on work that hasn't landed yet."""


def system_prompt(condition: Condition) -> str:
    """The system prompt for a condition: the base prompt, plus the verification instruction where the spec adds it."""
    if condition in ("verify_prompt", "harness"):
        return BASE_PROMPT + "\n\n" + VERIFY_INSTRUCTION
    return BASE_PROMPT


def require_ready(condition: Condition) -> None:
    """Raise ConditionNotReady unless the condition can run today."""
    if condition not in READY:
        raise ConditionNotReady(f"condition {condition!r} needs {WAITING_ON[condition]}")
