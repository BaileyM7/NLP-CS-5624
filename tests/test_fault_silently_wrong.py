"""Tests for the silently-wrong fault (#11)."""
import json
from pathlib import Path

import pytest
from mcp import Client

from toolharness.app.backend import ArticlePage
from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import FactEdit, missing_edits, read_edits, write_edits
from toolharness.data.scenario import FaultType, write_jsonl
from toolharness.injector.faults import build_corruptors
from toolharness.injector.faults.silently_wrong import SilentlyWrong
from toolharness.injector.injector import FaultInjector
from toolharness.mcp_server.__main__ import main
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_EDITS = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample_edits.jsonl"
SYMPHONY = {"title": "Symphony in Black", "page": 1}
HOLIDAY = {"title": "Billie Holiday", "page": 1}


@pytest.fixture
def edits(lady_day_scenario):
    def edit(title, original, new):
        return FactEdit(
            scenario_id=lady_day_scenario.scenario_id,
            title=title,
            page=1,
            original_value=original,
            new_value=new,
            new_answer="another film",
            model="test/model",
        )

    return [edit("Symphony in Black", "Billie Holiday", "Ella Fitzgerald"), edit("Billie Holiday", "Lady Day", "Lady Swing")]


def make_injector(scenario, edits):
    app = MockApp(scenario)
    return app, FaultInjector(app, scenario, SilentlyWrong(edits), seed=0, episode_id="ep-1")


def test_page_stays_valid_and_every_mention_is_swapped(lady_day_scenario, edits):
    app, injector = make_injector(lady_day_scenario, edits)
    clean = json.loads(app.call("get_article", SYMPHONY).text)
    corrupted_text = injector.call("get_article", SYMPHONY).text

    ArticlePage.model_validate_json(corrupted_text)  # still passes the schema
    corrupted = json.loads(corrupted_text)
    assert corrupted["sentences"] == [
        "Symphony in Black is a 1935 musical short featuring Duke Ellington.",
        "The film, Ella Fitzgerald's screen debut, was directed by Fred Waller.",
        "Ella Fitzgerald sang in one scene.",
    ]
    for field in ["title", "page", "total_pages", "last_updated", "source_id"]:
        assert corrupted[field] == clean[field]


def test_fault_uses_the_edit_for_whichever_answer_page_comes_first(lady_day_scenario, edits):
    _, injector = make_injector(lady_day_scenario, edits)
    holiday = json.loads(injector.call("get_article", HOLIDAY).text)
    symphony = json.loads(injector.call("get_article", SYMPHONY).text)

    assert holiday["sentences"][1] == 'Nicknamed "Lady Swing" by Lester Young, Holiday shaped jazz singing.'
    assert "Billie Holiday's screen debut" in symphony["sentences"][1]  # fault already used: clean


def test_skipped_page_is_clean_and_the_fault_fires_on_the_next_answer_page(lady_day_scenario, edits):
    skipped = edits[1].model_copy(update={"skip_reason": "nothing here affects the answer"})  # Billie Holiday page
    app, injector = make_injector(lady_day_scenario, [edits[0], skipped])

    holiday = injector.call("get_article", HOLIDAY)
    symphony = json.loads(injector.call("get_article", SYMPHONY).text)

    assert holiday == app.call("get_article", HOLIDAY)
    assert "Ella Fitzgerald's screen debut" in symphony["sentences"][1]
    assert [r.corrupted for r in injector.history] == [False, True]


def test_committed_sample_edits_cover_every_answer_page(sample_scenarios):
    edits = read_edits([SAMPLE_EDITS])
    assert missing_edits(sample_scenarios, edits) == []

    corruptor = SilentlyWrong(edits)
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.SILENTLY_WRONG:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            args = {"title": title, "page": page}
            injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
            corrupted = injector.call("get_article", args).text
            ArticlePage.model_validate_json(corrupted)  # still a valid page
            assert corrupted != app.call("get_article", args).text


def test_retry_gets_the_clean_page(lady_day_scenario, edits):
    app, injector = make_injector(lady_day_scenario, edits)
    injector.call("get_article", SYMPHONY)
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)


@pytest.mark.anyio
async def test_server_serves_the_edit_when_edits_are_loaded(lady_day_scenario, edits):
    registry = EpisodeRegistry([lady_day_scenario], corruptors=build_corruptors(edits))
    meta = {"x-episode-id": "ep-1", "x-scenario-id": lady_day_scenario.scenario_id}
    async with Client(build_server(registry)) as client:
        result = await client.call_tool("get_article", SYMPHONY, meta=meta)

    assert not result.is_error
    assert "Ella Fitzgerald's screen debut" in result.content[0].text


def test_server_refuses_to_start_with_missing_edits(tmp_path, lady_day_scenario, edits):
    scenario_file = tmp_path / "scenarios.jsonl"
    edits_file = tmp_path / "edits.jsonl"
    write_jsonl([lady_day_scenario], scenario_file)
    write_edits(edits[:1], edits_file)  # the Billie Holiday page has no edit

    with pytest.raises(SystemExit, match="1 silently_wrong answer pages have no fact edit"):
        main(["--scenarios", str(scenario_file), "--edits", str(edits_file)])
