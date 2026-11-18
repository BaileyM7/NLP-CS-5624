"""Tests for the hard-failure fault (#9)."""
import json

import pytest
from mcp import Client

from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType
from toolharness.injector.faults import CORRUPTORS
from toolharness.injector.faults.hard_failure import ERROR_MESSAGES
from toolharness.injector.ground_truth import GroundTruthLog
from toolharness.injector.injector import FaultInjector
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server


@pytest.fixture
def scenarios(sample_scenarios):
    return [s for s in sample_scenarios if s.fault_type == FaultType.HARD_FAILURE]


def answer_page_args(app):
    """get_article args for the scenario's first answer page."""
    title, page = sorted(app.answer_pages())[0]
    return {"title": title, "page": page}


def make_injector(scenario, seed=0, episode_id="ep-1", gt_log=None):
    app = MockApp(scenario)
    injector = FaultInjector(app, scenario, CORRUPTORS[FaultType.HARD_FAILURE], seed, episode_id, gt_log)
    return app, injector


def test_answer_page_becomes_a_realistic_error(scenarios):
    for scenario in scenarios:
        app, injector = make_injector(scenario)
        response = injector.call("get_article", answer_page_args(app))
        assert response.is_error
        assert response.text in ERROR_MESSAGES


def test_retry_gets_the_clean_page(scenarios):
    app, injector = make_injector(scenarios[0])
    args = answer_page_args(app)
    injector.call("get_article", args)
    assert injector.call("get_article", args) == app.call("get_article", args)


def test_message_depends_only_on_seed_and_scenario(scenarios):
    def message(scenario, seed, episode_id):
        app, injector = make_injector(scenario, seed, episode_id)
        return injector.call("get_article", answer_page_args(app)).text

    assert message(scenarios[0], 0, "opus-ep") == message(scenarios[0], 0, "haiku-ep")
    messages = {message(s, seed, "ep") for s in scenarios for seed in range(5)}
    assert len(messages) > 1


def test_ground_truth_log_records_the_error(tmp_path, scenarios):
    log_path = tmp_path / "gt.jsonl"
    app, injector = make_injector(scenarios[0], gt_log=GroundTruthLog(log_path))
    args = answer_page_args(app)
    injector.call("get_article", args)

    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["corrupted"] is True
    assert row["corrupted_is_error"] is True
    assert row["corrupted_text"] in ERROR_MESSAGES
    assert row["original"] == app.call("get_article", args).text


@pytest.mark.anyio
async def test_server_returns_an_mcp_error_result(scenarios):
    scenario = scenarios[0]
    args = answer_page_args(MockApp(scenario))
    meta = {"x-episode-id": "ep-1", "x-scenario-id": scenario.scenario_id}
    async with Client(build_server(EpisodeRegistry(scenarios))) as client:
        result = await client.call_tool("get_article", args, meta=meta)

    assert result.is_error
    assert result.content[0].text in ERROR_MESSAGES
