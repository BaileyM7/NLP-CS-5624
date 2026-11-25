"""Fact edits for the silently_wrong, stale and contradiction faults.

For every answer page of every silently_wrong scenario, an LLM picks one fact the question depends on
(an exact span of text) and a plausible wrong replacement. Edits are generated once and saved, so every
model and condition sees the same edit. The same question's stale and contradiction scenarios reuse them.

Usage:
    python -m toolharness.data.fact_edits --scenarios data/scenarios/hotpotqa.jsonl --out data/edits/hotpotqa.jsonl
        [--review data/edits/hotpotqa_review.jsonl]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Callable, Iterable, Iterator, Literal, Sequence

from pydantic import BaseModel, ConfigDict

from ..app.mock_app import MockApp
from ..llm.client import LLMError, OpenRouterClient, openrouter_api_key
from ..llm.models import FACT_EDIT_MODEL, MODELS
from .scenario import FaultType, Scenario, fault_variant_id, read_jsonl

DEFAULT_MODEL = MODELS[FACT_EDIT_MODEL].openrouter_id  # not one of the models under test
MAX_ATTEMPTS = 3
MAX_VALUE_CHARS = 80
EDIT_FAULTS = (FaultType.SILENTLY_WRONG, FaultType.STALE, FaultType.CONTRADICTION)  # faults that apply a fact edit

Complete = Callable[[str], str]  # sends one prompt to an LLM and returns the reply text

PROMPT = """We are testing whether AI agents notice when a tool returns a wrong fact. Your job is to plant one realistic wrong fact in an article page.

Question: {question}
Correct answer: {answer}

Page {page} of the article "{title}". Sentences marked * are the evidence for the answer.
{sentences}

Choose ONE fact on this page that answering the question depends on, and a wrong replacement for it, so that a reader who trusts this page would give a different answer to THIS question. For a multi-step question, changing the link this page provides is enough. The marked sentences are a hint; if the fact the question needs is in another sentence on the page, use that one. If changing a fact would not change the answer to this question, pick a different fact.

Rules:
- original_value: copied exactly, character for character, from a sentence on this page. Keep it short (a name, number, date, place or short phrase) and include any neighbouring words that must change with it (e.g. "an American" -> "a British").
- Don't change the article's own subject (the name in the title "{title}"). Change a fact stated about it.
- new_value: replaces the fact rather than adding to it. It's the same kind of thing (a year for a year, a person for a person), realistic, and not the correct answer.
- Every exact mention of original_value on this page will be replaced, so don't pick a span that also appears in unrelated places.
- new_answer: the answer a reader who trusts the edited page would now give.

Reply with only a JSON object: {{"original_value": "...", "new_value": "...", "new_answer": "..."}}"""

# Added to the prompt when a reviewer sent a page back
REVIEW_PROMPT = """

A reviewer rejected the previous edit for this page{previous}: {note}
Choose a different fact. If no fact on this page can change the answer to this question, reply with only {{"skip": "<one sentence on why>"}} instead."""


class FactEdit(BaseModel):
    """Swap original_value for new_value on one answer page of one scenario, or skip the page."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    title: str
    page: int
    original_value: str = ""
    new_value: str = ""
    new_answer: str = ""  # what a reader who trusts the edited page would answer
    model: str  # who wrote the edit: an OpenRouter model ID, or "reviewer" for a skip
    review_note: str = ""  # reviewer feedback the edit was regenerated with
    skip_reason: str = ""  # set when nothing on the page affects the answer; the fault doesn't fire here


class EditReview(BaseModel):
    """A reviewer's verdict on one page: regenerate its edit using the note, or skip the page."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str
    title: str
    page: int
    action: Literal["redo", "skip"]
    note: str


class EditRejected(ValueError):
    """An LLM reply that can't be used as a fact edit."""


def edit_key(item) -> tuple[str, str, int]:
    """(scenario_id, title, page): identifies an answer page for edits, reviews and answer targets."""
    return item.scenario_id, item.title, item.page


def span_pattern(value: str) -> re.Pattern:
    """Match value on its own, not inside a longer word or number (same rule as LongSeal answers)."""
    return re.compile(r"(?<![\w.,])" + re.escape(value) + r"(?![\w%]|[.,]\d)")


def apply_edit(sentences: list[str], edit: FactEdit) -> list[str]:
    """Replace every mention of the original value in the page's sentences."""
    pattern = span_pattern(edit.original_value)
    # A function replacement so backslashes in new_value are kept literally
    return [pattern.sub(lambda _: edit.new_value, sentence) for sentence in sentences]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split()).strip(".")


def title_subject(title: str) -> str:
    """The article's subject: its title before any "(...)" qualifier or comma ("TD Bank, N.A." -> "TD Bank")."""
    return re.split(r"\s*[(,]", title)[0]


def build_prompt(scenario: Scenario, title: str, page: int, sentences: list[str], positions: list[int]) -> str:
    """Show the LLM the question, the answer and the page, with supporting sentences marked *."""
    lines = []
    for i, sentence in enumerate(sentences):
        marker = "*" if i in positions else " "
        lines.append(f"{marker} {sentence}")
    return PROMPT.format(
        question=scenario.question,
        answer=scenario.gold_answer,
        page=page,
        title=title,
        sentences="\n".join(lines),
    )


def review_prompt(review: EditReview, previous: FactEdit | None) -> str:
    """The reviewer's note, quoting the rejected edit when there was one."""
    quoted = ""
    if previous is not None and not previous.skip_reason:
        quoted = f" ({previous.original_value!r} -> {previous.new_value!r})"
    return REVIEW_PROMPT.format(previous=quoted, note=review.note)


def parse_edit(
    reply: str,
    scenario: Scenario,
    title: str,
    page: int,
    sentences: list[str],
    model: str,
    allow_skip: bool = False,
) -> FactEdit:
    """Turn an LLM reply into a FactEdit, or raise EditRejected saying what's wrong with it."""
    # Read the JSON object, ignoring any text or code fences around it
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end == -1:
        raise EditRejected("the reply has no JSON object")
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        raise EditRejected("the reply is not valid JSON") from None
    if not isinstance(data, dict):
        raise EditRejected("the reply is not a JSON object")

    # A skip reply is only allowed when a reviewer sent the page back
    if "skip" in data:
        reason = str(data["skip"] or "").strip()
        if not allow_skip:
            raise EditRejected("this page needs an edit; skipping is not allowed")
        if not reason:
            raise EditRejected("a skip needs a reason")
        return FactEdit(scenario_id=scenario.scenario_id, title=title, page=page, model=model, skip_reason=reason)

    original = str(data.get("original_value") or "").strip()
    new = str(data.get("new_value") or "").strip()
    new_answer = str(data.get("new_answer") or "").strip()

    if not original or not new or not new_answer:
        raise EditRejected("original_value, new_value and new_answer must all be non-empty")
    if len(original) > MAX_VALUE_CHARS or len(new) > MAX_VALUE_CHARS:
        raise EditRejected(f"original_value and new_value must be at most {MAX_VALUE_CHARS} characters")
    if normalize(original) == normalize(new):
        raise EditRejected("new_value must be different from original_value")
    if normalize(original) in normalize(new):
        raise EditRejected("new_value must replace the fact, not add to it")
    if normalize(new) == normalize(scenario.gold_answer) or normalize(new_answer) == normalize(scenario.gold_answer):
        raise EditRejected("new_value and new_answer must not be the correct answer")
    if normalize(title_subject(title)) in normalize(original):
        raise EditRejected("original_value must not contain the article's own subject (its title); change a fact about it")

    pattern = span_pattern(original)
    if not any(pattern.search(sentence) for sentence in sentences):
        raise EditRejected(f"{original!r} does not appear exactly on this page")

    return FactEdit(
        scenario_id=scenario.scenario_id,
        title=title,
        page=page,
        original_value=original,
        new_value=new,
        new_answer=new_answer,
        model=model,
    )


def write_edit(
    scenario: Scenario,
    app: MockApp,
    title: str,
    page: int,
    complete: Complete,
    model: str,
    review: EditReview | None = None,
    previous: FactEdit | None = None,
) -> FactEdit:
    """Ask the LLM for an edit to one answer page, retrying with the reason a reply was rejected."""
    sentences = app.get_article(title, page).sentences
    positions = app.supporting_positions(title, page)
    prompt = build_prompt(scenario, title, page, sentences, positions)
    if review is not None:
        prompt += review_prompt(review, previous)

    feedback = ""
    reason = ""
    for _ in range(MAX_ATTEMPTS):
        reply = complete(prompt + feedback)
        try:
            edit = parse_edit(reply, scenario, title, page, sentences, model, allow_skip=review is not None)
        except EditRejected as error:
            reason = str(error)
            feedback = f"\n\nYour previous reply was rejected: {reason}. Try again."
            continue
        if review is not None:
            edit = edit.model_copy(update={"review_note": review.note})
        return edit
    raise EditRejected(f"{scenario.scenario_id} {title!r} page {page}: {reason}")


def pages_to_edit(scenarios: Iterable[Scenario]) -> Iterator[tuple[Scenario, MockApp, str, int]]:
    """Every answer page of every silently_wrong scenario, as (scenario, app, title, page)."""
    for scenario in scenarios:
        if scenario.fault_type != FaultType.SILENTLY_WRONG:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            yield scenario, app, title, page


def build_edits(
    scenarios: Iterable[Scenario],
    complete: Complete,
    model: str,
    existing: Iterable[FactEdit] = (),
    reviews: Iterable[EditReview] = (),
) -> tuple[list[FactEdit], list[str]]:
    """Write an edit for every page that needs one. Returns (edits, failures).

    Existing edits are kept unless a review sends the page back. A "skip" review marks the
    page skipped without asking the LLM; a "redo" review regenerates the edit with the note.
    """
    done = {edit_key(e): e for e in existing}
    review_by_page = {edit_key(r): r for r in reviews}
    edits: list[FactEdit] = []
    failures: list[str] = []
    for scenario, app, title, page in pages_to_edit(scenarios):
        key = (scenario.scenario_id, title, page)
        review = review_by_page.get(key)
        previous = done.get(key)

        # Reviewer skip: no LLM call needed
        if review is not None and review.action == "skip":
            edits.append(
                FactEdit(
                    scenario_id=scenario.scenario_id,
                    title=title,
                    page=page,
                    model="reviewer",
                    review_note=review.note,
                    skip_reason=review.note,
                )
            )
            continue

        # Keep an existing edit unless a review asks for a redo it hasn't had yet
        if previous is not None and (review is None or previous.review_note == review.note):
            edits.append(previous)
            continue

        # A page that fails (bad replies or an API error after retries) is listed and the run continues
        try:
            edit = write_edit(scenario, app, title, page, complete, model, review, previous)
        except (EditRejected, LLMError) as error:
            failures.append(str(error))
            print(f"FAILED  {error}", flush=True)
            continue
        edits.append(edit)
        if edit.skip_reason:
            print(f"{scenario.scenario_id} {title!r}: SKIPPED ({edit.skip_reason})", flush=True)
        else:
            print(f"{scenario.scenario_id} {title!r}: {edit.original_value!r} -> {edit.new_value!r}", flush=True)
    return edits, failures


def missing_edits(scenarios: Iterable[Scenario], edits: Iterable[FactEdit]) -> list[str]:
    """Answer pages that need an edit and have none, each listed once under its silently_wrong scenario ID."""
    have = {(e.scenario_id, e.title, e.page) for e in edits}
    missing = []
    for scenario in scenarios:
        if scenario.fault_type not in EDIT_FAULTS:
            continue
        # Stale and contradiction scenarios use the edit of the same question's silently_wrong scenario
        edit_scenario_id = fault_variant_id(scenario.scenario_id, FaultType.SILENTLY_WRONG)
        for title, page in sorted(MockApp(scenario).answer_pages()):
            message = f"{edit_scenario_id} {title!r} page {page}"
            if (edit_scenario_id, title, page) not in have and message not in missing:
                missing.append(message)
    return missing


def write_edits(edits: Iterable[FactEdit], path: str | Path) -> int:
    """Write one edit per line and return how many were written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for edit in edits:
            f.write(edit.model_dump_json() + "\n")
            count += 1
    return count


def read_edits(paths: Iterable[str | Path]) -> list[FactEdit]:
    """Read edits from one or more files written by write_edits."""
    edits = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    edits.append(FactEdit.model_validate_json(line))
    return edits


def read_reviews(path: str | Path) -> list[EditReview]:
    """Read a hand-written review file: one EditReview JSON object per line."""
    reviews = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reviews.append(EditReview.model_validate_json(line))
    return reviews


def openrouter_complete(model: str) -> Complete:
    """A Complete function that sends each prompt to `model` through the shared OpenRouter client."""
    try:
        api_key = openrouter_api_key()
    except LLMError as error:
        raise SystemExit(str(error)) from None

    async def ask(prompt: str) -> str:
        async with OpenRouterClient(api_key=api_key) as llm:
            result = await llm.chat(model, [{"role": "user", "content": prompt}], tag="fact_edit")
        return result.content or ""

    def complete(prompt: str) -> str:
        # A fresh client per prompt: an async client can't be reused across asyncio.run calls
        return asyncio.run(ask(prompt))

    return complete


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate fact edits for silently_wrong scenarios.")
    parser.add_argument("--scenarios", nargs="+", required=True, help="scenario JSONL files")
    parser.add_argument("--out", required=True, help="edit JSONL file; existing edits in it are kept")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID that writes the edits")
    parser.add_argument("--review", help="review JSONL: pages to redo with a note, or to skip")
    args = parser.parse_args(argv)

    scenarios = []
    for path in args.scenarios:
        scenarios.extend(read_jsonl(path))

    # Catch typos in the review file: every review must name a real answer page
    reviews = read_reviews(args.review) if args.review else []
    page_keys = {(s.scenario_id, title, page) for s, _, title, page in pages_to_edit(scenarios)}
    unknown = [r for r in reviews if edit_key(r) not in page_keys]
    if unknown:
        raise SystemExit(f"{len(unknown)} reviews name no answer page, e.g. {edit_key(unknown[0])}")

    out = Path(args.out)
    existing = read_edits([out]) if out.exists() else []
    edits, failures = build_edits(scenarios, openrouter_complete(args.model), args.model, existing, reviews)
    write_edits(edits, out)
    print(f"Wrote {len(edits)} edits to {out}")

    if failures:
        print(f"{len(failures)} pages have no edit yet. Run the same command again to retry only those.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
