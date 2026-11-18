"""Tests for stratified_sample."""
import random

import pytest

from toolharness.data.sampling import stratified_sample


def make_rows(counts: dict[str, int]) -> list[dict]:
    """Rows labelled by 'group', e.g. {"a": 80, "b": 20} -> 80 'a' rows and 20 'b' rows."""
    rows = []
    for label, count in counts.items():
        for i in range(count):
            rows.append({"id": f"{label}{i:03d}", "group": label})
    return rows


def count_groups(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return counts


def ids(rows: list[dict]) -> list[str]:
    return [row["id"] for row in rows]


def test_keeps_group_proportions():
    picked = stratified_sample(make_rows({"a": 80, "b": 20}), 20, seed=0, key="group")
    assert count_groups(picked) == {"a": 16, "b": 4}


def test_leftover_slots_go_to_largest_remainders():
    # Exact shares are 3.33 / 3.33 / 3.33 for n=10 -> one group gets the extra slot
    picked = stratified_sample(make_rows({"a": 30, "b": 30, "c": 30}), 10, seed=0, key="group")
    assert sum(count_groups(picked).values()) == 10
    assert sorted(count_groups(picked).values()) == [3, 3, 4]


def test_small_group_still_appears():
    picked = stratified_sample(make_rows({"a": 98, "b": 2}), 5, seed=0, key="group")
    assert count_groups(picked)["b"] == 1
    assert len(picked) == 5


def test_returns_distinct_rows():
    picked = stratified_sample(make_rows({"a": 80, "b": 20}), 50, seed=0, key="group")
    assert len(set(ids(picked))) == 50


def test_same_seed_same_result_different_seed_different_result():
    rows = make_rows({"a": 80, "b": 20})
    first = ids(stratified_sample(rows, 20, seed=0, key="group"))
    again = ids(stratified_sample(rows, 20, seed=0, key="group"))
    other = ids(stratified_sample(rows, 20, seed=1, key="group"))
    assert first == again
    assert first != other


def test_input_order_does_not_change_result():
    rows = make_rows({"a": 80, "b": 20})
    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    assert ids(stratified_sample(rows, 20, seed=0, key="group")) == ids(
        stratified_sample(shuffled, 20, seed=0, key="group")
    )


def test_asking_for_too_many_rows_fails():
    with pytest.raises(ValueError):
        stratified_sample(make_rows({"a": 3}), 5, seed=0, key="group")
