"""Convert HotpotQA (distractor setting, dev split) into Scenario rows.

Usage:
    python -m toolharness.data.hotpotqa --n 50 --seed 0 --out data/scenarios/hotpotqa.jsonl
    python -m toolharness.data.hotpotqa --n 5  --seed 0 --out data/sample/hotpotqa_sample.jsonl
"""
from __future__ import annotations

import argparse
from typing import Any, Sequence

from .sampling import stratified_sample
from .scenario import (
    Document,
    FaultType,
    Scenario,
    SupportingFact,
    make_scenario_id,
    write_jsonl,
)

HF_DATASET = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"

Row = dict[str, Any]  # one raw HotpotQA row from the `datasets` library


def load_rows(split: str = HF_SPLIT) -> list[Row]:
    """Download (or read from the local cache) one split of HotpotQA distractor."""
    from datasets import load_dataset  # imported here so tests don't need it

    # Point at the split's parquet file directly so other splits aren't downloaded
    data_files = {split: f"{HF_CONFIG}/{split}-*.parquet"}
    return list(load_dataset(HF_DATASET, data_files=data_files, split=split))


def is_usable(row: Row) -> bool:
    """True if every supporting fact points at a sentence that exists in the context."""
    sentences_by_title = dict(zip(row["context"]["title"], row["context"]["sentences"]))
    facts = zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"])
    for title, sent_id in facts:
        if title not in sentences_by_title:
            return False
        if sent_id < 0 or sent_id >= len(sentences_by_title[title]):
            return False
    return True


def sample_questions(rows: Sequence[Row], n: int, seed: int) -> list[Row]:
    """Pick n questions, keeping the dataset's bridge/comparison mix."""
    return stratified_sample(rows, n, seed, key="type")


def row_to_scenarios(row: Row) -> list[Scenario]:
    """Turn one question into one Scenario per fault condition."""
    docs = []
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        docs.append(Document(title=title, sentences=[s.strip() for s in sentences]))

    facts = []
    for title, sent_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        facts.append(SupportingFact(title=title, sent_id=sent_id))

    # Articles holding a supporting fact, without duplicates, in order
    targets: list[str] = []
    for fact in facts:
        if fact.title not in targets:
            targets.append(fact.title)

    scenarios = []
    for fault in FaultType:
        scenarios.append(
            Scenario(
                scenario_id=make_scenario_id("hotpotqa", row["id"], fault),
                question_id=row["id"],
                dataset="hotpotqa",
                question=row["question"],
                gold_answer=row["answer"],
                fault_type=fault,
                fault_target_titles=targets,
                supporting_facts=facts,
                context=docs,
                metadata={"type": row["type"], "level": row["level"]},
            )
        )
    return scenarios


def build_scenarios(rows: Sequence[Row], n: int, seed: int) -> list[Scenario]:
    """Filter unusable rows, sample n questions, and expand each into scenarios."""
    usable = [row for row in rows if is_usable(row)]
    picked = sample_questions(usable, n, seed)
    scenarios = []
    for row in picked:
        scenarios.extend(row_to_scenarios(row))
    return scenarios


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build HotpotQA scenarios.")
    parser.add_argument("--n", type=int, default=50, help="number of questions (each x6 fault conditions)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/scenarios/hotpotqa.jsonl")
    args = parser.parse_args(argv)

    rows = load_rows()
    scenarios = build_scenarios(rows, args.n, args.seed)
    count = write_jsonl(scenarios, args.out)
    usable = sum(1 for row in rows if is_usable(row))
    print(f"{len(rows)} rows, {usable} usable -> {args.n} questions -> {count} scenarios -> {args.out}")


if __name__ == "__main__":
    main()
