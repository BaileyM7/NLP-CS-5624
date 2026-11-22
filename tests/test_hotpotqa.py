"""Tests for HotpotQA -> Scenario conversion. Uses fake rows, no network."""
from pathlib import Path

import pytest

from toolharness.data.hotpotqa import build_scenarios, is_usable, row_to_scenarios
from toolharness.data.scenario import FaultType, fault_variant_id, make_scenario_id, read_jsonl, write_jsonl

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


def make_row(i: int, qtype: str = "bridge") -> dict:
    """A fake row in the HotpotQA format: 10 articles of 3 sentences each."""
    titles = [f"Article {i}-{k}" for k in range(10)]
    return {
        "id": f"q{i:04d}",
        "question": f"Question {i}?",
        "answer": f"Answer {i}",
        "type": qtype,
        "level": "hard",
        "supporting_facts": {"title": [titles[0], titles[3], titles[0]], "sent_id": [0, 1, 2]},
        "context": {
            "title": titles,
            "sentences": [[f" Sentence {s} of {t}." for s in range(3)] for t in titles],
        },
    }


def fake_rows(n: int = 100) -> list[dict]:
    """n fake rows, 80% bridge and 20% comparison."""
    return [make_row(i, "comparison" if i % 5 == 0 else "bridge") for i in range(n)]


def test_row_expands_to_every_fault_condition():
    scenarios = row_to_scenarios(make_row(1))
    assert [s.fault_type for s in scenarios] == list(FaultType)
    assert len({s.scenario_id for s in scenarios}) == len(FaultType)
    assert {s.question_id for s in scenarios} == {"q0001"}


def test_fault_variant_id_swaps_only_the_fault():
    sw = make_scenario_id("longseal", "q-12", FaultType.SILENTLY_WRONG)
    assert fault_variant_id(sw, FaultType.STALE) == make_scenario_id("longseal", "q-12", FaultType.STALE)
    assert fault_variant_id(sw, FaultType.SILENTLY_WRONG) == sw


def test_fault_targets_are_supporting_titles_in_order_without_duplicates():
    scenario = row_to_scenarios(make_row(1))[0]
    assert scenario.fault_target_titles == ["Article 1-0", "Article 1-3"]


def test_sentences_are_stripped_and_context_kept():
    scenario = row_to_scenarios(make_row(1))[0]
    assert len(scenario.context) == 10
    assert scenario.context[0].sentences[0] == "Sentence 0 of Article 1-0."
    assert scenario.context[0].last_updated is None


def test_rows_with_missing_supporting_sentences_are_unusable():
    bad_index = make_row(2)
    bad_index["supporting_facts"]["sent_id"][0] = 99
    bad_title = make_row(3)
    bad_title["supporting_facts"]["title"][0] = "Not In Context"
    assert is_usable(make_row(1))
    assert not is_usable(bad_index)
    assert not is_usable(bad_title)


def test_jsonl_round_trip(tmp_path):
    scenarios = row_to_scenarios(make_row(1))
    path = tmp_path / "scenarios.jsonl"
    assert write_jsonl(scenarios, path) == 6
    assert read_jsonl(path) == scenarios


def test_build_output_is_byte_identical_for_same_seed(tmp_path):
    rows = fake_rows()
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(build_scenarios(rows, 10, seed=0), first)
    write_jsonl(build_scenarios(rows, 10, seed=0), second)
    assert first.read_bytes() == second.read_bytes()


def test_build_keeps_both_question_types():
    scenarios = build_scenarios(fake_rows(), 10, seed=0)
    assert {s.metadata["type"] for s in scenarios} == {"bridge", "comparison"}


@pytest.mark.skipif(not SAMPLE_FILE.exists(), reason="sample not generated yet")
def test_committed_sample_is_valid():
    scenarios = read_jsonl(SAMPLE_FILE)
    variants_per_question: dict[str, int] = {}
    for scenario in scenarios:
        variants_per_question[scenario.question_id] = variants_per_question.get(scenario.question_id, 0) + 1
        assert scenario.fault_target_titles, scenario.scenario_id
    assert set(variants_per_question.values()) == {len(FaultType)}
