"""Convert LongSeal (the long-context subset of SealQA) into Scenario rows.

LongSeal is a benchmark with a training-contamination canary: never commit its data.

Usage:
    python -m toolharness.data.longseal --n 30 --seed 0 --out data/scenarios/longseal.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
from typing import Any, Sequence
from urllib.parse import unquote

from ..app.mock_app import tokenize
from .sampling import stratified_sample
from .scenario import Document, FaultType, Scenario, SupportingFact, make_scenario_id, write_jsonl

HF_DATASET = "vtllms/sealqa"
HF_FILE = "longseal.parquet"
DOC_FIELD = "12_docs"  # distractor documents served next to the gold documents
MAX_PIECE_CHARS = 1000  # longest "sentence" kept; longer text is cut
MAX_SUPPORTING_FACTS = 3
HEADING_MAX_CHARS = 60  # a highlighted line this short with no digits is treated as a heading
ROWS_LOOKAHEAD = 10  # how many lines below a heading to search for data rows
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

Row = dict[str, Any]  # one raw LongSeal row (plus the "id" added by load_rows)


def question_id(question: str) -> str:
    """Stable ID from the question text (LongSeal rows have no ID)."""
    return "ls" + hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


def load_rows() -> list[Row]:
    """Download (or read from the local cache) LongSeal and give each row an ID."""
    from datasets import load_dataset  # imported here so tests don't need it

    rows = list(load_dataset(HF_DATASET, data_files={"test": HF_FILE}, split="test"))
    for row in rows:
        row["id"] = question_id(row["question"])
    return rows


def parse_date(text: str | None) -> dt.date | None:
    """Parse 'Apr 03, 2022' or an ISO timestamp; None if missing or unrecognised."""
    if not text:
        return None
    text = text.strip()
    try:
        return dt.datetime.strptime(text, "%b %d, %Y").date()
    except ValueError:
        pass
    try:
        return dt.datetime.fromisoformat(text).date()
    except ValueError:
        return None


def chunk_words(text: str, max_chars: int) -> list[str]:
    """Cut text into pieces of at most max_chars characters, breaking between words."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = ""
    for word in text.split(" "):
        candidate = word if not current else current + " " + word
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split a document into lines; split long lines into sentences, then into chunks."""
    pieces = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [line] if len(line) <= MAX_PIECE_CHARS else SENTENCE_END.split(line)
        for part in parts:
            part = part.strip()
            if part:
                pieces.extend(chunk_words(part, MAX_PIECE_CHARS))
    return pieces


def unique_title(title: str | None, seen: set[str]) -> str:
    """Return a title not yet in seen (compared without case), adding ' (2)', ' (3)' if needed."""
    base = (title or "").strip() or "Untitled document"
    candidate = base
    number = 2
    while candidate.lower() in seen:
        candidate = f"{base} ({number})"
        number += 1
    seen.add(candidate.lower())
    return candidate


def contains_answer(sentence: str, answer: str) -> bool:
    """True if the answer appears in the sentence on its own (any case).

    The answer can't be part of a longer word or number: "7" doesn't match
    "53.7%", "1,7" or "7.5", and "381" doesn't match "1,381".
    """
    before = r"(?<![\w.,])"  # not preceded by a letter, digit, '.' or ','
    after = r"(?![\w%]|[.,]\d)"  # not followed by a letter, digit, '%', or '.5' / ',000'
    pattern = before + re.escape(answer) + after
    return re.search(pattern, sentence, re.IGNORECASE) is not None


def build_documents(row: Row) -> tuple[list[Document], list[str]]:
    """All documents for a question (gold first, then distractors) and the gold titles."""
    seen: set[str] = set()
    docs = []
    gold_titles = []
    for raw in row["golds"]:
        title = unique_title(raw["title"], seen)
        docs.append(Document(title=title, sentences=split_sentences(raw["text"] or ""), last_updated=parse_date(raw["date"])))
        gold_titles.append(title)
    for raw in row[DOC_FIELD]:
        title = unique_title(raw["title"], seen)
        docs.append(Document(title=title, sentences=split_sentences(raw["text"] or ""), last_updated=parse_date(raw["date"])))
    return docs, gold_titles


def is_bare_number(answer: str) -> bool:
    """True for answers like '7', '381' or '1,250' (no words or units)."""
    return re.fullmatch(r"[\d,.]+", answer.strip()) is not None


def parse_text_fragments(urls: list[str] | None) -> list[tuple[str, str | None]]:
    """(start, end) passages from '#:~:text=' links; end is None for a single passage.

    Fragment format: text=[prefix-,]start[,end][,-suffix], percent-encoded.
    """
    passages = []
    for url in urls or []:
        if not url or ":~:text=" not in url:
            continue
        for directive in url.split(":~:text=", 1)[1].split("&text="):
            items = directive.split(",")
            if items and items[0].endswith("-"):
                items = items[1:]  # drop the prefix
            if items and items[-1].startswith("-"):
                items = items[:-1]  # drop the suffix
            if not items:
                continue
            start = unquote(items[0]).strip()
            end = unquote(items[1]).strip() if len(items) > 1 else None
            if start:
                passages.append((start, end))
    return passages


def normalize(text: str) -> str:
    """Lowercase with all whitespace collapsed to single spaces."""
    return re.sub(r"\s+", " ", text).strip().lower()


def locate_passage(sentences: list[str], start: str, end: str | None) -> list[int]:
    """Indexes of the sentences covering the passage from start to end (empty if not found)."""
    # Join the sentences into one normalized string, remembering where each one sits
    joined = ""
    bounds = []
    for sentence in sentences:
        piece = normalize(sentence)
        if joined:
            joined += " "
        bounds.append((len(joined), len(joined) + len(piece)))
        joined += piece

    begin = joined.find(normalize(start))
    if begin == -1:
        return []
    finish = begin + len(normalize(start))
    if end:
        end_at = joined.find(normalize(end), finish)
        if end_at != -1:
            finish = end_at + len(normalize(end))

    covered = []
    for index, (first, last) in enumerate(bounds):
        if first < finish and last > begin:
            covered.append(index)
    return covered


def is_heading(sentence: str) -> bool:
    """Short line with no digits, e.g. 'List of highest-grossing films'."""
    return len(sentence) <= HEADING_MAX_CHARS and not any(ch.isdigit() for ch in sentence)


def rows_below(sentences: list[str], index: int, count: int) -> list[int]:
    """Up to `count` lines containing a digit just below `index` (the rows of a table or list)."""
    rows = []
    last = min(index + 1 + ROWS_LOOKAHEAD, len(sentences))
    for i in range(index + 1, last):
        if any(ch.isdigit() for ch in sentences[i]):
            rows.append(i)
            if len(rows) == count:
                break
    return rows


def find_fragment_facts(
    docs: list[Document],
    gold_titles: list[str],
    passages: list[tuple[str, str | None]],
    bare_number: bool,
) -> list[SupportingFact]:
    """Gold-document sentences covered by the dataset's highlighted evidence passages."""
    facts: list[SupportingFact] = []
    for start, end in passages:
        for doc in docs:
            if doc.title not in gold_titles:
                continue
            covered = locate_passage(doc.sentences, start, end)

            # For number answers, a passage that is only headings points at the rows below it
            if bare_number and covered and all(is_heading(doc.sentences[i]) for i in covered):
                covered = rows_below(doc.sentences, covered[-1], MAX_SUPPORTING_FACTS)

            for sent_id in covered:
                fact = SupportingFact(title=doc.title, sent_id=sent_id)
                if fact not in facts:
                    facts.append(fact)
            if covered:
                break  # use the first gold document containing this passage
    return facts[:MAX_SUPPORTING_FACTS]


def find_answer_facts(question: str, answer: str, docs: list[Document], gold_titles: list[str]) -> list[SupportingFact]:
    """Gold-document sentences containing the answer, keeping the ones most related to the question."""
    question_words = set(tokenize(question))
    candidates = []  # (question words shared, position found, fact)
    for doc in docs:
        if doc.title not in gold_titles:
            continue
        for sent_id, sentence in enumerate(doc.sentences):
            if contains_answer(sentence, answer):
                shared = len(question_words & set(tokenize(sentence)))
                candidates.append((shared, len(candidates), SupportingFact(title=doc.title, sent_id=sent_id)))

    # Most shared words first; earlier position breaks ties
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [fact for _, _, fact in candidates[:MAX_SUPPORTING_FACTS]]


def find_supporting_facts(row: Row, docs: list[Document], gold_titles: list[str]) -> tuple[list[SupportingFact], str]:
    """Supporting sentences and how they were found: "answer_match", "fragment" or "none".

    Text answers use the sentence containing the answer. Bare numbers (and text answers
    with no literal match) use the dataset's highlighted evidence passage instead.
    """
    bare_number = is_bare_number(row["answer"])
    if not bare_number:
        facts = find_answer_facts(row["question"], row["answer"], docs, gold_titles)
        if facts:
            return facts, "answer_match"
    facts = find_fragment_facts(docs, gold_titles, parse_text_fragments(row["urls"]), bare_number)
    if facts:
        return facts, "fragment"
    return [], "none"


def is_usable(row: Row) -> bool:
    """True if at least one supporting sentence can be found in a gold document."""
    docs, gold_titles = build_documents(row)
    facts, _ = find_supporting_facts(row, docs, gold_titles)
    return len(facts) > 0


def row_to_scenarios(row: Row) -> list[Scenario]:
    """Turn one question into one Scenario per fault condition."""
    docs, gold_titles = build_documents(row)
    facts, evidence = find_supporting_facts(row, docs, gold_titles)

    # Gold articles holding a supporting fact, without duplicates, in order
    targets: list[str] = []
    for fact in facts:
        if fact.title not in targets:
            targets.append(fact.title)

    metadata = {
        "search_results": str(row["search_results"]),
        "freshness": str(row["freshness"]),
        "question_types": "; ".join(row["question_types"]),
        "topic": str(row["topic"]),
        "effective_year": str(row["effective_year"]),
        "canary": str(row["canary"]),
        "evidence": evidence,  # how the supporting sentences were found
    }

    scenarios = []
    for fault in FaultType:
        scenarios.append(
            Scenario(
                scenario_id=make_scenario_id("longseal", row["id"], fault),
                question_id=row["id"],
                dataset="longseal",
                question=row["question"],
                gold_answer=row["answer"],
                fault_type=fault,
                fault_target_titles=targets,
                supporting_facts=facts,
                context=docs,
                metadata=metadata,
            )
        )
    return scenarios


def build_scenarios(rows: Sequence[Row], n: int, seed: int) -> list[Scenario]:
    """Filter unusable rows, sample n questions by search-result type, and expand each into scenarios."""
    usable = [row for row in rows if is_usable(row)]
    picked = stratified_sample(usable, n, seed, key="search_results")
    scenarios = []
    for row in picked:
        scenarios.extend(row_to_scenarios(row))
    return scenarios


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build LongSeal scenarios (local only, never commit).")
    parser.add_argument("--n", type=int, default=30, help="number of questions (each x6 fault conditions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/scenarios/longseal.jsonl")
    args = parser.parse_args(argv)

    rows = load_rows()
    scenarios = build_scenarios(rows, args.n, args.seed)
    count = write_jsonl(scenarios, args.out)
    usable = sum(1 for row in rows if is_usable(row))
    print(f"{len(rows)} rows, {usable} usable -> {args.n} questions -> {count} scenarios -> {args.out}")


if __name__ == "__main__":
    main()
