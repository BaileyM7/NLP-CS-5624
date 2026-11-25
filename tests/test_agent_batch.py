"""Tests for batch runs: scenario selection, episode IDs, resume, the run manifest and the command line."""
import argparse
import dataclasses
import json

import pytest
from fakes import SAMPLE_FILE, FakeOpenRouter, answer_reply, free_port, make_client, tool_reply

from toolharness.agent.__main__ import main, parse_skip
from toolharness.agent.batch import (
    BatchConfig,
    ManifestMismatch,
    check_manifest,
    make_episode_id,
    manifest_entry,
    run_batch,
    select_scenarios,
)
from toolharness.agent.runner import INVALID_ARGUMENTS
from toolharness.agent.session import InMemoryToolSession
from toolharness.data.scenario import FaultType
from toolharness.eval.trace import read_jsonl, trace_filename
from toolharness.harness.checks import HARNESS_CHECKS, harness_settings
from toolharness.mcp_server.registry import EpisodeRegistry
from toolharness.mcp_server.server import build_server

ANSWER = (200, answer_reply("Final Answer: 2005"))


@pytest.fixture
def config(tmp_path):
    return BatchConfig(
        run_id="r1",
        model="haiku-5.5",
        condition="baseline",
        scenario_files=[SAMPLE_FILE],
        questions=1,
        faults=[FaultType.NONE, FaultType.HARD_FAILURE],
        concurrency=1,
        runs_dir=tmp_path,
    )


@pytest.fixture
def open_session(sample_scenarios):
    server = build_server(EpisodeRegistry(sample_scenarios))
    return lambda episode_id, scenario_id, seed: InMemoryToolSession(server, episode_id, scenario_id, seed)


async def batch(config, fake, open_session):
    async with make_client(fake) as llm:
        return await run_batch(config, llm, open_session)


def trace_path(config):
    return config.runs_dir / config.run_id / "traces" / trace_filename(config.model, config.condition)


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# Selection and IDs

def test_select_scenarios_takes_the_first_questions_and_chosen_faults(sample_scenarios):
    assert len(select_scenarios(sample_scenarios, 2, None)) == 12
    picked = select_scenarios(sample_scenarios, 2, [FaultType.NONE, FaultType.HARD_FAILURE])
    assert len(picked) == 4
    assert [s.fault_type for s in picked].count(FaultType.NONE) == 2
    assert select_scenarios(sample_scenarios, None, None) == sample_scenarios


def test_skip_drops_only_that_dataset_and_fault(sample_scenarios):
    longseal = [s.model_copy(update={"dataset": "longseal", "scenario_id": "longseal-" + s.scenario_id}) for s in sample_scenarios]
    picked = select_scenarios(sample_scenarios + longseal, None, None, skip=[("longseal", FaultType.CONTRADICTION)])

    assert len(picked) == 2 * len(sample_scenarios) - 5  # the 5 sample questions each have one contradiction scenario
    assert not any(s.dataset == "longseal" and s.fault_type == FaultType.CONTRADICTION for s in picked)
    assert any(s.dataset == "hotpotqa" and s.fault_type == FaultType.CONTRADICTION for s in picked)


def test_skip_parses_and_rejects_unknown_values():
    assert parse_skip("longseal:contradiction") == ("longseal", FaultType.CONTRADICTION)
    for bad in ["nope:stale", "longseal:nope", "longseal"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_skip(bad)

    argv = ["--run-id", "r", "--model", "haiku-5.5", "--condition", "baseline", "--scenarios", str(SAMPLE_FILE), "--skip", "nope:stale"]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2


@pytest.mark.anyio
async def test_batch_skips_the_chosen_dataset_and_fault(config, open_session):
    config = dataclasses.replace(config, faults=[FaultType.NONE, FaultType.CONTRADICTION], skip=[("hotpotqa", FaultType.CONTRADICTION)])
    summary = await batch(config, FakeOpenRouter(ANSWER), open_session)
    assert (summary.ran, summary.skipped) == (1, 0)
    assert [e.fault_type for e in read_jsonl(trace_path(config))] == [FaultType.NONE]


def test_episode_id_format():
    episode_id = make_episode_id("pilot", "haiku-5.5", "baseline", "hotpotqa-x-none", 0, 1)
    assert episode_id == "pilot__haiku-5.5__baseline__hotpotqa-x-none__s0__a1"


# Running and resuming

@pytest.mark.anyio
async def test_batch_writes_traces_and_skips_finished_episodes_on_resume(config, open_session):
    summary = await batch(config, FakeOpenRouter(ANSWER, ANSWER), open_session)
    assert (summary.ran, summary.skipped, summary.errored) == (2, 0, 0)
    assert summary.stop_reasons["final_answer"] == 2
    episodes = read_jsonl(trace_path(config))
    assert len(episodes) == 2 and all(e.episode_id.endswith("__a1") for e in episodes)
    manifest = json.loads((config.runs_dir / "r1" / "manifest.json").read_text(encoding="utf-8"))
    assert "haiku-5.5__baseline" in manifest["entries"]

    fake = FakeOpenRouter()
    summary = await batch(config, fake, open_session)
    assert (summary.ran, summary.skipped) == (0, 2)
    assert fake.requests == []


@pytest.mark.anyio
async def test_errored_episode_is_recorded_and_retried_with_a_new_episode_id(config, open_session):
    fake = FakeOpenRouter((400, {"error": {"message": "bad request"}}), ANSWER)
    summary = await batch(config, fake, open_session)
    assert (summary.ran, summary.errored) == (1, 1)
    errors = read_rows(config.runs_dir / "r1" / "errors.jsonl")
    assert len(errors) == 1 and errors[0]["attempt"] == 1 and "HTTP 400" in errors[0]["error"]
    failed_scenario = errors[0]["scenario_id"]
    assert failed_scenario not in {e.scenario_id for e in read_jsonl(trace_path(config))}

    summary = await batch(config, FakeOpenRouter(ANSWER), open_session)
    assert (summary.ran, summary.skipped) == (1, 1)
    retried = next(e for e in read_jsonl(trace_path(config)) if e.scenario_id == failed_scenario)
    assert retried.episode_id.endswith("__a2")


@pytest.mark.anyio
async def test_interrupted_attempt_gets_a_new_episode_id(config, open_session, sample_scenarios):
    scenario = next(s for s in sample_scenarios if s.fault_type == FaultType.NONE)
    run_dir = config.runs_dir / "r1"
    run_dir.mkdir(parents=True)
    killed = make_episode_id("r1", "haiku-5.5", "baseline", scenario.scenario_id, 0, 1)
    row = {"episode_id": killed, "scenario_id": scenario.scenario_id, "attempt": 1}
    (run_dir / "started.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    await batch(config, FakeOpenRouter(ANSWER, ANSWER), open_session)
    episode = next(e for e in read_jsonl(trace_path(config)) if e.scenario_id == scenario.scenario_id)
    assert episode.episode_id.endswith("__a2")


@pytest.mark.anyio
async def test_a_second_model_in_the_same_run_starts_at_attempt_1(config, open_session):
    await batch(config, FakeOpenRouter(ANSWER, ANSWER), open_session)
    config.model = "gpt-6-luna"
    await batch(config, FakeOpenRouter(ANSWER, ANSWER), open_session)

    episodes = read_jsonl(trace_path(config))
    assert len(episodes) == 2 and all(e.episode_id.endswith("__a1") for e in episodes)


@pytest.mark.anyio
async def test_an_unexpected_error_in_one_episode_does_not_stop_the_batch(config, sample_scenarios):
    server = build_server(EpisodeRegistry(sample_scenarios))
    broken = next(s for s in sample_scenarios if s.fault_type == FaultType.NONE).scenario_id

    def open_session(episode_id, scenario_id, seed):
        if scenario_id == broken:
            raise ValueError("odd provider reply")
        return InMemoryToolSession(server, episode_id, scenario_id, seed)

    summary = await batch(config, FakeOpenRouter(ANSWER), open_session)
    assert (summary.ran, summary.errored) == (1, 1)
    errors = read_rows(config.runs_dir / "r1" / "errors.jsonl")
    assert errors[0]["scenario_id"] == broken and errors[0]["error"] == "ValueError: odd provider reply"


@pytest.mark.anyio
async def test_questions_limit_applies_to_each_scenario_file(config, open_session, sample_scenarios, tmp_path):
    # A second file whose questions are copies of the sample's under new IDs
    second = tmp_path / "second.jsonl"
    with second.open("w", encoding="utf-8") as stream:
        for s in sample_scenarios:
            copy = s.model_copy(update={"scenario_id": "copy-" + s.scenario_id, "question_id": "copy-" + s.question_id})
            stream.write(copy.model_dump_json() + "\n")
    scenarios = sample_scenarios + [s.model_copy(update={"scenario_id": "copy-" + s.scenario_id, "question_id": "copy-" + s.question_id}) for s in sample_scenarios]
    server = build_server(EpisodeRegistry(scenarios))
    config.scenario_files = [SAMPLE_FILE, second]

    summary = await batch(config, FakeOpenRouter(*[ANSWER] * 4), lambda e, s, seed: InMemoryToolSession(server, e, s, seed))
    assert summary.ran == 4  # 1 question x 2 faults from each file
    assert sum(e.scenario_id.startswith("copy-") for e in read_jsonl(trace_path(config))) == 2


@pytest.mark.anyio
async def test_concurrent_batch_writes_every_episode_once(config, open_session):
    config.questions = 4
    config.faults = [FaultType.NONE, FaultType.HARD_FAILURE, FaultType.MALFORMED]
    config.concurrency = 4
    summary = await batch(config, FakeOpenRouter(*[ANSWER] * 12), open_session)

    assert summary.ran == 12
    rows = read_rows(trace_path(config))
    assert len(rows) == 12
    assert len({row["scenario_id"] for row in rows}) == 12


# Manifest

def test_manifest_blocks_a_changed_experiment_but_not_a_new_commit(config, tmp_path):
    path = tmp_path / "manifest.json"
    entry = manifest_entry(config)
    check_manifest(path, entry)
    check_manifest(path, {**entry, "git_commit": "abc", "git_dirty": True, "created_at": "later"})
    with pytest.raises(ManifestMismatch, match="system_prompt"):
        check_manifest(path, {**entry, "system_prompt": "changed"})


def test_manifest_accepts_the_same_scenario_file_under_another_path(config, tmp_path):
    path = tmp_path / "manifest.json"
    check_manifest(path, manifest_entry(config))
    copy = tmp_path / "elsewhere" / "sample.jsonl"
    copy.parent.mkdir()
    copy.write_bytes(SAMPLE_FILE.read_bytes())
    config.scenario_files = [copy]
    check_manifest(path, manifest_entry(config))  # same content, different spelling: fine

    copy.write_bytes(SAMPLE_FILE.read_bytes() + b"\n")
    with pytest.raises(ManifestMismatch, match="scenario_hashes"):
        check_manifest(path, manifest_entry(config))


# Command line

def test_cli_stops_when_the_server_is_down(capsys):
    url = f"http://127.0.0.1:{free_port()}/mcp"
    argv = ["--run-id", "r", "--model", "haiku-5.5", "--condition", "baseline", "--scenarios", str(SAMPLE_FILE), "--server-url", url]
    assert main(argv) == 2
    assert "not reachable" in capsys.readouterr().out


# The harness layer (#17)

SEARCH = (200, tool_reply(("search", {"thought": "Look it up.", "query": "Shin Hyun-joon"})))
LINE = "\n\n_harness: "  # how the harness line starts in a tool response


def only_call(config):
    """The single tool call of the single episode in this config's trace file."""
    [episode] = read_jsonl(trace_path(config))
    [call] = episode.turns[0].tool_calls
    return call


@pytest.mark.anyio
async def test_flags_only_episode_records_the_harness_line_and_fields(config, open_session):
    config = dataclasses.replace(config, condition="flags_only", faults=[FaultType.NONE])
    fake = FakeOpenRouter(SEARCH, ANSWER)
    await batch(config, fake, open_session)

    call = only_call(config)
    assert call.raw_tool_response.count(LINE) == 1
    assert json.loads(call.raw_tool_response.split(LINE)[1])["flags"] == []
    assert call.harness_flags == []
    assert call.proxy_latency_ms is not None
    assert fake.requests[1]["messages"][-1]["content"] == call.raw_tool_response  # the model saw the line


@pytest.mark.anyio
async def test_baseline_episode_has_no_harness_fields(config, open_session):
    config = dataclasses.replace(config, faults=[FaultType.NONE])
    await batch(config, FakeOpenRouter(SEARCH, ANSWER), open_session)

    call = only_call(config)
    assert "_harness" not in call.raw_tool_response
    assert call.harness_flags is None
    assert call.proxy_latency_ms is None


@pytest.mark.anyio
async def test_runner_made_results_get_no_line(config, open_session):
    config = dataclasses.replace(config, condition="flags_only", faults=[FaultType.NONE])
    await batch(config, FakeOpenRouter((200, tool_reply(("search", "not json"))), ANSWER), open_session)

    call = only_call(config)
    assert call.raw_tool_response == INVALID_ARGUMENTS
    assert call.harness_flags is None


def test_manifest_records_harness_settings(config):
    settings = harness_settings("flags_only")
    assert manifest_entry(dataclasses.replace(config, condition="flags_only"))["harness"] == settings
    assert settings["checks"] == [check.key for check in HARNESS_CHECKS]
    assert settings["line_prefix"] == "_harness: "
    assert manifest_entry(config)["harness"] is None


def test_changed_harness_settings_block_a_resume(tmp_path, config):
    entry = manifest_entry(dataclasses.replace(config, condition="flags_only"))
    check_manifest(tmp_path / "manifest.json", entry)
    changed = dict(entry, harness={**entry["harness"], "checks": ["something_else"]})
    with pytest.raises(ManifestMismatch, match="harness"):
        check_manifest(tmp_path / "manifest.json", changed)
