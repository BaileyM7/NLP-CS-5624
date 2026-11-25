"""Tests for the harness consistency check (#20)."""
import json
from pathlib import Path

from toolharness.agent.session import ToolResult
from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import read_edits
from toolharness.data.scenario import FaultType
from toolharness.harness.checks import HARNESS_CHECKS, harness_settings
from toolharness.harness.consistency import CONSISTENCY_OVERLAP, ConsistencyCheck
from toolharness.injector.faults import build_corruptors
from toolharness.injector.injector import FaultInjector

SAMPLE_EDITS = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample_edits.jsonl"
CHECK = ConsistencyCheck()
CLEAN = "Once a Day is a song recorded as the debut single by American country artist Connie Smith."
EDITED = "Once a Day is a song recorded as the debut single by American country artist Dottie West."


def page(title, sentences):
    data = {"title": title, "page": 1, "total_pages": 1, "sentences": sentences, "last_updated": "2026-09-01", "source_id": "x"}
    return ToolResult(text=json.dumps(data), is_error=False)


def search(snippets):
    hits = [{"title": f"T{i}", "page": 1, "snippet": snippet} for i, snippet in enumerate(snippets)]
    return ToolResult(text=json.dumps({"query": "q", "hits": hits}), is_error=False)


def run(result, history, tool="get_article"):
    return CHECK.run(tool, result, history)


def test_key_overlap_and_manifest_setting():
    assert CHECK.key == "consistent"
    assert CONSISTENCY_OVERLAP == 0.7
    assert harness_settings("flags_only")["consistency_overlap"] == 0.7
    assert [check.key for check in HARNESS_CHECKS] == ["schema_valid", "fresh", "consistent"]


def test_page_that_rewrites_a_sentence_seen_earlier_is_flagged():
    result = run(page("Once a Day", [EDITED]), [("search", search([CLEAN]))])
    assert result.value is False
    assert result.flags == ['consistency: says "Dottie West" where an earlier result said "Connie Smith"']


def test_a_conflict_with_another_articles_sentence_is_flagged():
    history = [("get_article", page("Connie Smith", ["Intro sentence here.", CLEAN]))]
    assert run(page("Once a Day", [EDITED]), history).value is False


def test_the_same_page_fetched_again_is_consistent():
    history = [("get_article", page("Once a Day", [CLEAN, "Another sentence that is long enough here."]))]
    result = run(page("Once a Day", [CLEAN, "Another sentence that is long enough here."]), history)
    assert (result.value, result.flags) == (True, [])


def test_a_cut_snippet_of_the_same_sentence_is_not_a_conflict():
    long_sentence = CLEAN[:-1] + ", " + "who later recorded many more singles for the label over a long career " * 3 + "."
    assert len(long_sentence) > 200
    result = run(page("Once a Day", [long_sentence]), [("search", search([long_sentence[:200]]))])
    assert (result.value, result.flags) == (True, [])


def test_a_punctuation_only_difference_is_not_a_conflict():
    history = [("search", search(["Connie Smith, born in 1941, was an American singer of country music songs."]))]
    result = run(page("Connie Smith", ["Connie Smith (born in 1941) was an American singer of country music songs."]), history)
    assert (result.value, result.flags) == (True, [])


def test_unrelated_sentences_are_consistent():
    history = [("search", search(["Symphony in Black is a 1935 musical short featuring Duke Ellington."]))]
    assert run(page("Once a Day", [EDITED]), history).value is True


def test_short_sentences_are_ignored():
    history = [("search", search(["Born in 1941."]))]
    assert run(page("X", ["Born in 1942."]), history).value is True


def test_unusable_earlier_responses_are_ignored():
    history = [
        ("get_article", ToolResult(text="503 Service Unavailable", is_error=True)),
        ("get_article", ToolResult(text="<html>broken</html>", is_error=False)),
        ("get_article", ToolResult(text="[1, 2]", is_error=False)),
        ("search", ToolResult(text='{"query": "q", "hits": "none"}', is_error=False)),
    ]
    assert run(page("Once a Day", [EDITED]), history).value is True


def test_non_page_responses_are_not_applicable():
    history = [("search", search([CLEAN]))]
    not_found = json.dumps({"error": "not_found", "title": "T", "page": None, "message": "No article."})
    cases = [
        ("search", search([EDITED])),
        ("get_article", ToolResult(text="503 Service Unavailable", is_error=True)),
        ("get_article", ToolResult(text=not_found, is_error=False)),
        ("get_article", ToolResult(text='{"title": "T", "sentences": ["cut', is_error=False)),
    ]
    for tool, result in cases:
        assert (run(result, history, tool).value, run(result, history, tool).flags) == (None, [])


def test_at_most_two_flags_per_response():
    seen = [f"Sentence number {n} about the song recorded by Connie Smith in the year 1964." for n in range(3)]
    edited = [sentence.replace("Connie Smith", "Dottie West") for sentence in seen]
    result = run(page("X", edited), [("search", search(seen))])
    assert result.value is False and len(result.flags) == 2


def test_every_contradiction_fault_on_the_sample_is_flagged(sample_scenarios):
    corruptor = build_corruptors(read_edits([SAMPLE_EDITS]))[FaultType.CONTRADICTION]
    checked = 0
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.CONTRADICTION:
            continue
        app = MockApp(scenario)
        injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
        history = []
        calls = [("search", {"query": scenario.question})]
        for title, number in sorted(app.answer_pages()):
            calls += [("get_article", {"title": title, "page": number})] * 2
        for tool, args in calls:
            response = injector.call(tool, args)
            result = ToolResult(text=response.text, is_error=response.is_error)
            if injector.history[-1].corrupted:
                assert CHECK.run(tool, result, list(history)).value is False, scenario.scenario_id
                checked += 1
            history.append((tool, result))
    assert checked > 0


def test_clean_sample_pages_are_never_flagged(sample_scenarios):
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.NONE:
            continue
        app = MockApp(scenario)
        history = [("search", ToolResult(text=app.call("search", {"query": scenario.question}).text, is_error=False))]
        for title, pages in app.pages.items():
            for number in range(1, len(pages) + 1):
                result = ToolResult(text=app.call("get_article", {"title": title, "page": number}).text, is_error=False)
                outcome = CHECK.run("get_article", result, list(history))
                assert (outcome.value, outcome.flags) == (True, []), (scenario.scenario_id, title)
                history.append(("get_article", result))
