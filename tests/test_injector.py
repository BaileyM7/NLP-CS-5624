"""Tests for the fault injector, its ground-truth log, and its wiring into the server."""
import json
from pathlib import Path

import pytest
from mcp import Client

from toolharness.app.backend import ToolResponse
from toolharness.app.mock_app import MockApp
from toolharness.data.scenario import FaultType, read_jsonl
from toolharness.injector.corruptors import Corruptor, NoFault
from toolharness.injector.ground_truth import GroundTruthLog, summarize_gt
from toolharness.injector.injector import FaultInjector
from toolharness.mcp_server.registry import EpisodeError, EpisodeRegistry
from toolharness.mcp_server.server import build_server

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"


class MarkCorruptor(Corruptor):
    """Test corruptor: replaces the response with a fixed marker and remembers its target."""

    def __init__(self):
        self.targets = []

    def corrupt(self, call, clean, history, rng, target):
        self.targets.append(target)
        return ToolResponse(text="CORRUPTED")


class RandomCorruptor(Corruptor):
    """Test corruptor: output depends only on the injector's rng."""

    def corrupt(self, call, clean, history, rng, target):
        return ToolResponse(text=str(rng.random()))


@pytest.fixture
def scenario():
    return next(s for s in read_jsonl(SAMPLE_FILE) if s.fault_type == FaultType.STALE)


@pytest.fixture
def app(scenario):
    return MockApp(scenario)


@pytest.fixture
def answer_titles(app):
    return sorted({title for title, _ in app.answer_pages()})


@pytest.fixture
def other_title(scenario, answer_titles):
    return next(doc.title for doc in scenario.context if doc.title not in answer_titles)


def get_article(title):
    return {"title": title, "page": 1}


# Injector behaviour

def test_no_fault_output_is_identical_to_mock_app(app, scenario):
    injector = FaultInjector(app, scenario, NoFault(), seed=0, episode_id="ep-1")
    calls = [("search", {"query": scenario.question, "k": 5})]
    for doc in scenario.context:
        calls.append(("get_article", get_article(doc.title)))
    calls.append(("get_article", get_article("No Such Article")))
    calls.append(("get_article", {"title": scenario.context[0].title, "page": 9}))

    for tool, args in calls:
        assert injector.call(tool, args) == app.call(tool, args)


def test_fires_once_on_the_first_answer_page(app, scenario, answer_titles, other_title):
    injector = FaultInjector(app, scenario, MarkCorruptor(), seed=0, episode_id="ep-1")
    injector.call("get_article", get_article(other_title))
    injector.call("search", {"query": scenario.question, "k": 5})
    first_fetch = injector.call("get_article", get_article(answer_titles[0]))
    retry = injector.call("get_article", get_article(answer_titles[0]))
    other_answer = injector.call("get_article", get_article(answer_titles[-1]))

    assert first_fetch.text == "CORRUPTED"
    assert retry == app.call("get_article", get_article(answer_titles[0]))
    assert other_answer == app.call("get_article", get_article(answer_titles[-1]))
    assert [r.corrupted for r in injector.history] == [False, False, True, False, False]
    assert [r.retry_of_corrupted for r in injector.history] == [False, False, False, True, False]


def test_corruptor_gets_the_answer_target(app, scenario, answer_titles):
    corruptor = MarkCorruptor()
    injector = FaultInjector(app, scenario, corruptor, seed=0, episode_id="ep-1")
    injector.call("get_article", get_article(answer_titles[0]))

    target = corruptor.targets[0]
    assert (target.title, target.page) == (answer_titles[0], 1)
    assert target.gold_answer == scenario.gold_answer
    expected_positions = sorted({f.sent_id for f in scenario.supporting_facts if f.title == answer_titles[0]})
    assert target.sentence_positions == expected_positions  # one-page articles: position == sentence index


def test_title_case_does_not_dodge_the_fault(app, scenario, answer_titles):
    injector = FaultInjector(app, scenario, MarkCorruptor(), seed=0, episode_id="ep-1")
    assert injector.call("get_article", get_article(answer_titles[0].lower())).text == "CORRUPTED"


def test_search_never_triggers(app, scenario):
    injector = FaultInjector(app, scenario, MarkCorruptor(), seed=0, episode_id="ep-1")
    for query in [scenario.question, scenario.gold_answer, "film", "music"]:
        assert injector.call("search", {"query": query, "k": 5}).text != "CORRUPTED"


def test_same_seed_gives_same_corruption_across_episodes(app, scenario, answer_titles):
    def corrupted_text(seed, episode_id):
        injector = FaultInjector(app, scenario, RandomCorruptor(), seed=seed, episode_id=episode_id)
        return injector.call("get_article", get_article(answer_titles[0])).text

    assert corrupted_text(0, "opus-ep") == corrupted_text(0, "haiku-ep")
    assert corrupted_text(0, "opus-ep") != corrupted_text(1, "opus-ep")


# Ground-truth log

def test_log_has_one_row_per_call_and_text_only_for_the_corrupted_call(
    tmp_path, app, scenario, answer_titles, other_title
):
    log_path = tmp_path / "gt.jsonl"
    injector = FaultInjector(app, scenario, MarkCorruptor(), 0, "ep-1", GroundTruthLog(log_path))
    injector.call("get_article", get_article(other_title))
    injector.call("get_article", get_article(answer_titles[0]))
    injector.call("get_article", get_article(answer_titles[0]))

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["call_index"] for row in rows] == [0, 1, 2]
    assert rows[0]["original"] is None
    assert rows[1]["corrupted"] is True
    assert rows[1]["corrupted_text"] == "CORRUPTED"
    assert json.loads(rows[1]["original"])["title"] == answer_titles[0]
    assert rows[2]["retry_of_corrupted"] is True
    assert {row["fault_type"] for row in rows} == {"stale"}


def test_summary_reports_whether_the_fault_was_injected(tmp_path, app, scenario, answer_titles, other_title):
    log = GroundTruthLog(tmp_path / "gt.jsonl")
    hit = FaultInjector(app, scenario, MarkCorruptor(), 0, "ep-hit", log)
    hit.call("search", {"query": scenario.question, "k": 5})
    hit.call("get_article", get_article(answer_titles[0]))
    hit.call("get_article", get_article(answer_titles[0]))
    missed = FaultInjector(app, scenario, MarkCorruptor(), 0, "ep-missed", log)
    missed.call("get_article", get_article(other_title))

    summary = summarize_gt(tmp_path / "gt.jsonl")
    assert summary["ep-hit"].fault_injected is True
    assert summary["ep-hit"].injected_at_call == 1
    assert summary["ep-hit"].retries_of_corrupted == 1
    assert summary["ep-missed"].fault_injected is False
    assert summary["ep-missed"].calls == 1


# Registry and server wiring

def test_registry_rejects_fault_types_without_a_corruptor(scenario):
    registry = EpisodeRegistry([scenario])
    with pytest.raises(EpisodeError, match="no corruptor"):
        registry.get_or_create("ep-1", scenario.scenario_id, 0)


@pytest.mark.anyio
async def test_server_injects_the_fault_once_per_episode(tmp_path, scenario, answer_titles):
    registry = EpisodeRegistry(
        [scenario],
        gt_log_path=tmp_path / "gt.jsonl",
        corruptors={FaultType.STALE: MarkCorruptor()},
    )
    args = get_article(answer_titles[0])
    async with Client(build_server(registry)) as client:
        results = []
        for episode_id in ["ep-1", "ep-1", "ep-2"]:
            meta = {"x-episode-id": episode_id, "x-scenario-id": scenario.scenario_id}
            result = await client.call_tool("get_article", args, meta=meta)
            results.append(result.content[0].text)

    # ep-1: corrupted, then clean on retry; ep-2 has its own fault and gets corrupted too
    assert results[0] == "CORRUPTED"
    assert json.loads(results[1])["title"] == answer_titles[0]
    assert results[2] == "CORRUPTED"
    summary = summarize_gt(tmp_path / "gt.jsonl")
    assert summary["ep-1"].injected_at_call == 0
    assert summary["ep-2"].injected_at_call == 0
