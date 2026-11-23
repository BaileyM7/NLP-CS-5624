"""Tests for the harness format check (#18)."""
import random

import pytest

from toolharness.agent.session import ToolResult
from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.harness.format_check import FormatCheck
from toolharness.injector.corruptors import AnswerTarget
from toolharness.injector.faults.hard_failure import ERROR_MESSAGES
from toolharness.injector.faults.malformed import VARIANTS, break_page

CHECK = FormatCheck()


def run(tool, text, is_error=False):
    return CHECK.run(tool, ToolResult(text=text, is_error=is_error), [])


def clean_responses(scenario):
    """Every clean response an agent can get in this scenario: a search, every page, and an unknown title."""
    app = MockApp(scenario)
    responses = [("search", app.call("search", {"query": scenario.question}))]
    for title, pages in app.pages.items():
        for page in range(1, len(pages) + 1):
            responses.append(("get_article", app.call("get_article", {"title": title, "page": page})))
    responses.append(("get_article", app.call("get_article", {"title": "No Such Article"})))
    responses.append(("get_article", app.call("get_article", {"title": scenario.context[0].title, "page": 99})))
    return responses


def test_key_is_schema_valid():
    assert CHECK.key == "schema_valid"


def test_every_clean_sample_response_passes(sample_scenarios):
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.NONE:
            continue
        for tool, response in clean_responses(scenario):
            result = run(tool, response.text, response.is_error)
            assert (result.value, result.flags) == (True, []), (tool, response.text[:80])


@pytest.mark.parametrize("message", ERROR_MESSAGES)
def test_every_hard_failure_message_is_flagged(message):
    result = run("get_article", message, is_error=True)
    assert (result.value, result.flags) == (False, ["format: the tool returned an error"])


@pytest.mark.parametrize(
    "variant, flag",
    [
        ("invalid_json", "format: not valid JSON"),
        ("missing_field", "format: missing field 'sentences'"),
        ("wrong_type", "format: field 'sentences' has the wrong type"),
        ("truncated", "format: not valid JSON"),
    ],
)
def test_every_malformed_variant_is_flagged(sample_scenarios, variant, flag):
    assert set(VARIANTS) == {"invalid_json", "missing_field", "wrong_type", "truncated"}
    checked = 0
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.MALFORMED:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            target = AnswerTarget(
                scenario_id=scenario.scenario_id,
                title=title,
                page=page,
                sentence_positions=app.supporting_positions(title, page),
                gold_answer=scenario.gold_answer,
            )
            clean = app.call("get_article", {"title": title, "page": page}).text
            result = run("get_article", break_page(variant, clean, target, random.Random(0)))
            assert (result.value, result.flags) == (False, [flag])
            checked += 1
    assert checked > 0


def test_json_that_is_not_an_object_is_flagged():
    for text in ["[1, 2]", '"text"', "42"]:
        result = run("get_article", text)
        assert (result.value, result.flags) == (False, ["format: expected a JSON object"])


def test_a_search_result_with_a_broken_hit_is_flagged():
    result = run("search", '{"query": "q", "hits": [{"title": "T", "page": 1}]}')
    assert (result.value, result.flags) == (False, ["format: missing field 'hits.0.snippet'"])


def test_other_tools_are_not_applicable():
    result = run("calculator", "not json at all")
    assert (result.value, result.flags) == (None, [])
