"""Tests for the stale fault (#12)."""
import datetime as dt
import json
from pathlib import Path

import pytest
from fakes import fault_variant
from mcp import Client

from toolharness.app.backend import AS_OF, CLEAN_DATE_WINDOW_DAYS, ArticlePage
from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import read_edits
from toolharness.data.scenario import FaultType
from toolharness.injector.faults import build_corruptors
from toolharness.injector.faults.stale import Stale
from toolharness.injector.injector import FaultInjector
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_EDITS = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample_edits.jsonl"
SYMPHONY = {"title": "Symphony in Black", "page": 1}
SYMPHONY_DATE = dt.date(2026, 9, 1)  # Symphony in Black's date in the lady_day fixture
FRESHNESS_LIMIT = AS_OF - dt.timedelta(days=CLEAN_DATE_WINDOW_DAYS)  # clean undated pages are newer than this


@pytest.fixture
def stale_scenario(lady_day_scenario):
    return fault_variant(lady_day_scenario, FaultType.STALE)


def make_injector(scenario, edits, seed=0):
    app = MockApp(scenario)
    return app, FaultInjector(app, scenario, Stale(edits), seed=seed, episode_id="ep-1")


def stale_date(injector) -> dt.date:
    """The last_updated date of the corrupted Symphony in Black page."""
    page = json.loads(injector.call("get_article", SYMPHONY).text)
    return dt.date.fromisoformat(page["last_updated"])


def years_before(date: dt.date, years: int) -> dt.date:
    return date - dt.timedelta(days=years * 365)


def test_stale_page_has_the_edit_and_an_old_date(stale_scenario, lady_day_edits):
    app, injector = make_injector(stale_scenario, lady_day_edits)
    corrupted_text = injector.call("get_article", SYMPHONY).text
    clean = json.loads(app.call("get_article", SYMPHONY).text)

    ArticlePage.model_validate_json(corrupted_text)  # still passes the schema
    corrupted = json.loads(corrupted_text)
    assert corrupted["sentences"][1] == "The film, Ella Fitzgerald's screen debut, was directed by Fred Waller."
    date = dt.date.fromisoformat(corrupted["last_updated"])
    assert years_before(SYMPHONY_DATE, 6) <= date <= years_before(SYMPHONY_DATE, 3)
    for field in ["title", "page", "total_pages", "source_id"]:
        assert corrupted[field] == clean[field]


def test_stale_date_is_set_by_the_seed(stale_scenario, lady_day_edits):
    dates = []
    for seed in range(20):
        _, injector = make_injector(stale_scenario, lady_day_edits, seed)
        dates.append(stale_date(injector))

    _, same_seed = make_injector(stale_scenario, lady_day_edits, seed=0)
    assert stale_date(same_seed) == dates[0]
    assert len(set(dates)) > 1
    for date in dates:
        assert date < FRESHNESS_LIMIT


def test_stale_date_is_older_than_an_already_old_clean_date(stale_scenario, lady_day_edits):
    old_symphony = stale_scenario.context[0].model_copy(update={"last_updated": dt.date(2015, 3, 1)})
    scenario = stale_scenario.model_copy(update={"context": [old_symphony, *stale_scenario.context[1:]]})
    _, injector = make_injector(scenario, lady_day_edits)

    date = stale_date(injector)
    assert years_before(dt.date(2015, 3, 1), 6) <= date <= years_before(dt.date(2015, 3, 1), 3)


def test_stale_retry_gets_the_clean_page(stale_scenario, lady_day_edits):
    app, injector = make_injector(stale_scenario, lady_day_edits)
    injector.call("get_article", SYMPHONY)
    assert injector.call("get_article", SYMPHONY) == app.call("get_article", SYMPHONY)


def test_committed_sample_stale_pages_are_valid_and_old(sample_scenarios):
    corruptor = build_corruptors(read_edits([SAMPLE_EDITS]))[FaultType.STALE]
    checked = 0
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.STALE:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            args = {"title": title, "page": page}
            injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
            corrupted_text = injector.call("get_article", args).text

            ArticlePage.model_validate_json(corrupted_text)  # still a valid page
            corrupted = json.loads(corrupted_text)
            assert corrupted["sentences"] != json.loads(app.call("get_article", args).text)["sentences"]
            assert dt.date.fromisoformat(corrupted["last_updated"]) < FRESHNESS_LIMIT
            checked += 1
    assert checked > 0


@pytest.mark.anyio
async def test_server_serves_stale_pages_when_edits_are_loaded(stale_scenario, lady_day_edits):
    # Only the stale scenario is loaded; its edit is found by the silently_wrong ID
    registry = EpisodeRegistry([stale_scenario], corruptors=build_corruptors(lady_day_edits))
    meta = {"x-episode-id": "ep-1", "x-scenario-id": stale_scenario.scenario_id}
    async with Client(build_server(registry)) as client:
        result = await client.call_tool("get_article", SYMPHONY, meta=meta)

    assert not result.is_error
    page = json.loads(result.content[0].text)
    assert dt.date.fromisoformat(page["last_updated"]) <= years_before(SYMPHONY_DATE, 3)
