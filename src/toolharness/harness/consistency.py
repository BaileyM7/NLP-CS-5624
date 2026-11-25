"""Consistency check (#20): does a page rewrite a sentence the agent was shown earlier in the episode?"""
from __future__ import annotations

import difflib
import json
import re

from ..agent.session import ToolResult
from ..app.mock_app import SNIPPET_CHARS
from .base import CheckResult

CONSISTENCY_OVERLAP = 0.7  # word-set overlap (Jaccard) at which two different sentences count as one sentence rewritten
MIN_WORDS = 6  # shorter sentences are too generic to compare
MAX_FLAGS = 2  # flags per response
QUOTE_CHARS = 60  # characters quoted from each side of a conflict
WORD = re.compile(r"\w+")


def shown_sentences(tool: str, text: str) -> list[str]:
    """The sentences a response showed the agent: snippets for a search, sentences for a page; [] if unusable."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    if tool == "search":
        hits = data.get("hits")
        if not isinstance(hits, list):
            return []
        return [hit["snippet"] for hit in hits if isinstance(hit, dict) and isinstance(hit.get("snippet"), str)]

    sentences = data.get("sentences")
    if not isinstance(sentences, list):
        return []
    return [sentence for sentence in sentences if isinstance(sentence, str)]


def comparable(new: str, old: str) -> str:
    """The part of `new` to compare with `old`: a snippet cut at SNIPPET_CHARS only covers a sentence's start."""
    if len(old) >= SNIPPET_CHARS and len(new) > len(old):
        return new[: len(old)]
    return new


def near_duplicate(new: str, old: str) -> bool:
    """Two different sentences that share most of their words."""
    if new == old or new.startswith(old) or old.startswith(new):
        return False
    new_words = WORD.findall(comparable(new, old).lower())
    old_words = WORD.findall(old.lower())
    if len(new_words) < MIN_WORDS or len(old_words) < MIN_WORDS:
        return False
    # The same words with different punctuation or spacing are the same sentence
    if new_words == old_words:
        return False
    new_set, old_set = set(new_words), set(old_words)
    return len(new_set & old_set) / len(new_set | old_set) >= CONSISTENCY_OVERLAP


def difference(new: str, old: str) -> tuple[str, str]:
    """The first span of words that differs: (what the new sentence says, what the old one said)."""
    new_words = WORD.findall(new)
    old_words = WORD.findall(old)
    matcher = difflib.SequenceMatcher(a=old_words, b=new_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            return " ".join(new_words[j1:j2]) or "(nothing)", " ".join(old_words[i1:i2]) or "(nothing)"
    return "(nothing)", "(nothing)"


class ConsistencyCheck:
    """Flags a page sentence that nearly matches, but differs from, a sentence shown earlier in the episode."""

    key = "consistent"

    def run(self, tool: str, result: ToolResult, history: list[tuple[str, ToolResult]]) -> CheckResult:
        # Only a page has sentences to compare
        if tool != "get_article" or result.is_error:
            return CheckResult(value=None, flags=[])
        try:
            data = json.loads(result.text)
        except json.JSONDecodeError:
            return CheckResult(value=None, flags=[])
        if not isinstance(data, dict) or not isinstance(data.get("sentences"), list):
            return CheckResult(value=None, flags=[])

        # Every sentence the agent was shown earlier: search snippets and page sentences
        seen = []
        for earlier_tool, earlier in history:
            if not earlier.is_error:
                seen.extend(shown_sentences(earlier_tool, earlier.text))

        # One flag per conflicting sentence, naming the words that differ
        flags = []
        for new in shown_sentences(tool, result.text):
            for old in seen:
                if near_duplicate(new, old):
                    says, said = difference(comparable(new, old), old)
                    flags.append(f'consistency: says "{says[:QUOTE_CHARS]}" where an earlier result said "{said[:QUOTE_CHARS]}"')
                    break
            if len(flags) == MAX_FLAGS:
                break
        return CheckResult(value=not flags, flags=flags)
