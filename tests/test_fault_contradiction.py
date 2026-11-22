"""Tests for the contradiction fault (#13)."""
import json
from pathlib import Path

import pytest
from fakes import fault_variant
from mcp import Client

from toolharness.app.backend import ArticlePage, ToolResponse
from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import read_edits
from toolharness.data.scenario import Document, FaultType
from toolharness.injector.corruptors import CallRecord, ToolCall
from toolharness.injector.faults import build_corruptors
from toolharness.injector.faults.contradiction import Contradiction, shown_text
from toolharness.injector.injector import FaultInjector
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_EDITS = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample_edits.jsonl"
SYMPHONY = {"title": "Symphony in Black", "page": 1}
HOLIDAY = {"title": "Billie Holiday", "page": 1}
DEBUT_QUERY = {"query": "Billie Holiday screen debut"}
CLEAN_DEBUT = "The film, Billie Holiday's screen debut, was directed by Fred Waller."
EDITED_DEBUT = "The film, Ella Fitzgerald's screen debut, was directed by Fred Waller."


@pytest.fixture
def contradiction_scenario(lady_day_scenario):
    return fault_variant(lady_day_scenario, FaultType.CONTRADICTION)


def make_injector(scenario, edits):
    app = MockApp(scenario)
    return app, FaultInjector(app, scenario, Contradiction(edits), seed=0, episode_id="ep-1")


def record(text: str, is_error: bool = False) -> CallRecord:
    """An earlier call whose response was `text`."""
    response = ToolResponse(text=text, is_error=is_error)
    return CallRecord(
        index=0,
        call=ToolCall(tool="search", args={}),
        clean=response,
        returned=response,
        corrupted=False,
        retry_of_corrupted=False,
    )


# When the fault fires

def test_page_conflicts_with_the_snippet_the_agent_was_shown(contradiction_scenario, lady_day_edits):
    _, injector = make_injector(contradiction_scenario, lady_day_edits)
    result = json.loads(injector.call("search", DEBUT_QUERY).text)
    page_text = injector.call("get_article", SYMPHONY).text

    snippet = next(hit["snippet"] for hit in result["hits"] if hit["title"] == "Symphony in Black")
    assert snippet == CLEAN_DEBUT
    ArticlePage.model_validate_json(page_text)  # still a valid page
    assert json.loads(page_text)["sentences"][1] == EDITED_DEBUT
    assert [r.corrupted for r in injector.history] == [False, True]


def test_first_answer_page_is_clean_when_nothing_was_shown_yet(contradiction_scenario, lady_day_edits):
    app, injector = make_injector(contradiction_scenario, lady_day_edits)
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)
    assert injector.corrupted_record is None


def test_fact_mentioned_on_the_other_answer_page_doesnt_count(contradiction_scenario, lady_day_edits):
    # The Billie Holiday page mentions "Billie Holiday" but never says which film was her screen debut
    app, injector = make_injector(contradiction_scenario, lady_day_edits)
    injector.call("get_article", HOLIDAY)
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)
    assert injector.corrupted_record is None


def test_revisiting_a_page_fires_and_the_next_visit_is_clean(contradiction_scenario, lady_day_edits):
    app, injector = make_injector(contradiction_scenario, lady_day_edits)
    first = injector.call("get_article", SYMPHONY)
    second = json.loads(injector.call("get_article", SYMPHONY).text)
    third = injector.call("get_article", SYMPHONY)

    assert first == third == app.call("get_article", SYMPHONY)
    assert second["sentences"][1] == EDITED_DEBUT
    assert [(r.corrupted, r.retry_of_corrupted) for r in injector.history] == [(False, False), (True, False), (False, True)]


def test_lowercase_title_still_fires(contradiction_scenario, lady_day_edits):
    _, injector = make_injector(contradiction_scenario, lady_day_edits)
    injector.call("search", DEBUT_QUERY)
    page = json.loads(injector.call("get_article", {"title": "symphony in black", "page": 1}).text)
    assert page["sentences"][1] == EDITED_DEBUT


# When it doesn't

def test_fact_on_a_non_answer_page_doesnt_count(contradiction_scenario, lady_day_edits):
    lester = Document(title="Lester Young", sentences=["Lester Young often played with Billie Holiday."])
    scenario = contradiction_scenario.model_copy(update={"context": [*contradiction_scenario.context, lester]})
    app, injector = make_injector(scenario, lady_day_edits)

    injector.call("get_article", {"title": "Lester Young", "page": 1})
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)
    assert injector.corrupted_record is None


def test_not_found_and_empty_search_dont_count(contradiction_scenario, lady_day_edits):
    app, injector = make_injector(contradiction_scenario, lady_day_edits)
    injector.call("get_article", {"title": "No Such Article", "page": 1})
    injector.call("search", {"query": "zzzz"})
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)


def test_skipped_page_never_fires_even_after_its_fact_was_shown(contradiction_scenario, lady_day_edits):
    skipped = lady_day_edits[0].model_copy(update={"skip_reason": "nothing here affects the answer"})  # Symphony page
    app, injector = make_injector(contradiction_scenario, [skipped, lady_day_edits[1]])
    injector.call("search", DEBUT_QUERY)
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)


# What counts as shown

def test_shown_text_keeps_only_the_target_pages_snippet():
    search = {
        "query": "q",
        "hits": [
            {"title": "Symphony in Black", "page": 1, "snippet": CLEAN_DEBUT},
            {"title": "Billie Holiday", "page": 1, "snippet": "Billie Holiday was an American jazz singer."},
        ],
    }
    assert shown_text(record(json.dumps(search)), ("Symphony in Black", 1)) == [CLEAN_DEBUT]


def test_shown_text_keeps_only_the_target_pages_sentences():
    holiday = {"title": "Billie Holiday", "page": 1, "total_pages": 1, "sentences": ["Billie Holiday was a singer."]}
    assert shown_text(record(json.dumps(holiday)), ("Symphony in Black", 1)) == []
    assert shown_text(record(json.dumps(holiday)), ("Billie Holiday", 1)) == ["Billie Holiday was a singer."]


def test_shown_text_ignores_responses_that_are_not_pages_or_searches():
    page = ("Symphony in Black", 1)
    not_found = {"error": "not_found", "title": "Symphony in Black", "page": 1, "message": "Page 1 does not exist."}
    assert shown_text(record("<html>oops</html>", is_error=True), page) == []
    assert shown_text(record("[1, 2]"), page) == []
    assert shown_text(record(json.dumps(not_found)), page) == []


# Real data and the server

def test_committed_sample_contradiction_pages_fire_on_a_second_visit(sample_scenarios):
    corruptor = build_corruptors(read_edits([SAMPLE_EDITS]))[FaultType.CONTRADICTION]
    checked = 0
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.CONTRADICTION:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            args = {"title": title, "page": page}
            injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
            first = injector.call("get_article", args)
            second_text = injector.call("get_article", args).text

            assert first == app.call("get_article", args)
            ArticlePage.model_validate_json(second_text)  # still a valid page
            assert second_text != first.text
            checked += 1
    assert checked > 0


@pytest.mark.anyio
async def test_server_serves_contradiction_when_edits_are_loaded(contradiction_scenario, lady_day_edits):
    registry = EpisodeRegistry([contradiction_scenario], corruptors=build_corruptors(lady_day_edits))
    meta = {"x-episode-id": "ep-1", "x-scenario-id": contradiction_scenario.scenario_id}
    async with Client(build_server(registry)) as client:
        await client.call_tool("search", DEBUT_QUERY, meta=meta)
        result = await client.call_tool("get_article", SYMPHONY, meta=meta)

    assert not result.is_error
    assert json.loads(result.content[0].text)["sentences"][1] == EDITED_DEBUT
