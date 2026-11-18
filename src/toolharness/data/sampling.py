"""Seeded, proportional stratified sampling of dataset rows."""
from __future__ import annotations

import random
from typing import Any, Sequence

Row = dict[str, Any]


def stratified_sample(
    rows: Sequence[Row], n: int, seed: int, key: str, id_key: str = "id"
) -> list[Row]:
    """Pick n rows so each group (rows sharing row[key]) keeps its share of the total.

    Same rows + n + seed always give the same rows in the same order,
    regardless of the order of the input.
    """
    if n > len(rows):
        raise ValueError(f"asked for {n} rows but only {len(rows)} are available")

    # Group rows by label
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    labels = sorted(groups)

    # Give each group its exact share, rounded down
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for label in labels:
        exact = n * len(groups[label]) / len(rows)
        quotas[label] = int(exact)
        remainders.append((exact - int(exact), label))

    # Hand the leftover slots to the groups with the largest remainders
    leftover = n - sum(quotas.values())
    remainders.sort(key=lambda r: (-r[0], r[1]))
    for _, label in remainders[:leftover]:
        quotas[label] += 1

    # Make sure every group appears at least once when n allows it
    if n >= len(labels):
        for label in labels:
            if quotas[label] == 0:
                largest = max(labels, key=lambda name: quotas[name])
                quotas[largest] -= 1
                quotas[label] = 1

    # Draw from each group, then shuffle so groups are interleaved
    rng = random.Random(seed)
    picked: list[Row] = []
    for label in labels:
        group = sorted(groups[label], key=lambda row: row[id_key])
        picked.extend(rng.sample(group, quotas[label]))
    rng.shuffle(picked)
    return picked
