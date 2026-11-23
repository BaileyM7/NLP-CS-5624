"""Tests for the harness freshness check (#19)."""
import datetime as dt
import json
from pathlib import Path

from toolharness.agent.session import ToolResult
from toolharness.app.backend import AS_OF
from toolharness.app.mock_app import MockApp
from toolharness.data.fact_edits import read_edits
from toolharness.data.scenario import FaultType
from toolharness.harness.checks import harness_settings
from toolharness.harness.freshness import FRESHNESS_DAYS, FreshnessCheck
from toolharness.injector.faults.stale import Stale
from toolharness.injector.injector import FaultInjector

SAMPLE_EDITS = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample_edits.jsonl"
CHECK = FreshnessCheck()


def run(tool, text, is_error=False):
    return CHECK.run(tool, ToolResult(text=text, is_error=is_error), [])


def page_dated(date: dt.date) -> str:
    """A valid article page JSON with the given last_updated date."""
    page = {"title": "T", "page": 1, "total_pages": 1, "sentences": ["S."], "last_updated": date.isoformat(), "source_id": "x"}
    return json.dumps(page)


def test_key_window_and_manifest_setting():
    assert CHECK.key == "fresh"
    assert FRESHNESS_DAYS == 365
    assert harness_settings("flags_only")["freshness_days"] == 365


def test_every_stale_fault_is_flagged(sample_scenarios):
    corruptor = Stale(read_edits([SAMPLE_EDITS]))
    checked = 0
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.STALE:
            continue
        app = MockApp(scenario)
        for title, page in sorted(app.answer_pages()):
            injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
            corrupted = injector.call("get_article", {"title": title, "page": page})
            result = run("get_article", corrupted.text)
            assert result.value is False
            assert len(result.flags) == 1 and result.flags[0].startswith("stale: last updated ")
            checked += 1
    assert checked > 0


def test_clean_hotpotqa_pages_are_fresh(sample_scenarios):
    for scenario in sample_scenarios:
        if scenario.fault_type != FaultType.NONE:
            continue
        app = MockApp(scenario)
        for title, pages in app.pages.items():
            for page in range(1, len(pages) + 1):
                result = run("get_article", app.call("get_article", {"title": title, "page": page}).text)
                assert (result.value, result.flags) == (True, [])


def test_freshness_boundaries():
    assert run("get_article", page_dated(AS_OF - dt.timedelta(days=365))).value is True
    stale = run("get_article", page_dated(AS_OF - dt.timedelta(days=366)))
    assert (stale.value, stale.flags) == (False, ["stale: last updated 2025-10-31, more than 1 year before today (2026-11-01)"])
    assert run("get_article", page_dated(AS_OF + dt.timedelta(days=10))).value is True


def test_non_page_responses_are_not_applicable():
    search = json.dumps({"query": "q", "hits": []})
    not_found = json.dumps({"error": "not_found", "title": "T", "page": None, "message": "No article."})
    cases = [
        ("search", search, False),
        ("get_article", not_found, False),
        ("get_article", page_dated(AS_OF - dt.timedelta(days=3000)), True),  # an error result is never judged on its date
        ("get_article", "[1, 2]", False),
        ("get_article", '{"title": "T", "page": 1, "sentences": ["cut', False),
    ]
    for tool, text, is_error in cases:
        result = run(tool, text, is_error)
        assert (result.value, result.flags) == (None, []), text[:40]
