"""Tests for the pilot report (#15): costs, the full-run projection, fault injection, harness flags and reflection."""
import pytest
from fakes import gt_row, make_episode, write_run

from toolharness.agent.runner import INVALID_ARGUMENTS
from toolharness.eval.pilot_report import (
    Estimate,
    estimate_cost,
    exact_match,
    flag_counts,
    injection_counts,
    load_run,
    main,
    mean_costs,
    project_costs,
    projected_total,
    reflection_counts,
    verifier_cost,
)
from toolharness.llm.models import MODELS

CHEAP = ("haiku-5.5", "gpt-6-luna")

# Cheap models: harness costs 2x baseline, and LongSeal 3x HotpotQA. Opus measured only baseline on HotpotQA.
COSTS = {("opus-5.5", "baseline", "hotpotqa"): 10.0}
for cheap in CHEAP:
    COSTS[(cheap, "baseline", "hotpotqa")] = 1.0
    COSTS[(cheap, "harness", "hotpotqa")] = 2.0
    COSTS[(cheap, "baseline", "longseal")] = 3.0
    COSTS[(cheap, "harness", "longseal")] = 6.0


def test_model_prices_match_the_llm_client_doc():
    prices = {name: (spec.usd_per_m_input, spec.usd_per_m_output) for name, spec in MODELS.items()}
    assert prices == {
        "opus-5.5": (4.0, 20.0),
        "haiku-5.5": (0.10, 0.50),
        "gpt-6-astra": (10.0, 50.0),
        "gpt-6-luna": (0.10, 0.50),
        "sonnet-5.5": (2.0, 10.0),
        "gpt-6.1-sol": (2.0, 10.0),
    }


def test_exact_match_normalizes_like_hotpotqa():
    assert exact_match("The  Beatles.", "beatles")
    assert exact_match("an Apple", "apple")
    assert not exact_match("1999", "2000")
    assert not exact_match(None, "x")


def test_mean_costs_group_by_model_condition_and_dataset():
    episodes = [
        make_episode("haiku-5.5", "baseline", "hotpotqa", "none", cost=1.0, scenario="a"),
        make_episode("haiku-5.5", "baseline", "hotpotqa", "none", cost=3.0, scenario="b"),
        make_episode("haiku-5.5", "baseline", "longseal", "none", cost=5.0),
    ]
    assert mean_costs(episodes) == {("haiku-5.5", "baseline", "hotpotqa"): 2.0, ("haiku-5.5", "baseline", "longseal"): 5.0}


# Projection

def test_estimate_uses_measured_costs_and_scales_the_rest():
    assert estimate_cost(COSTS, "opus-5.5", "baseline", "hotpotqa") == Estimate(10.0, "measured")
    assert estimate_cost(COSTS, "opus-5.5", "harness", "hotpotqa").cost == pytest.approx(20.0)
    assert estimate_cost(COSTS, "opus-5.5", "baseline", "longseal").cost == pytest.approx(30.0)
    assert estimate_cost(COSTS, "opus-5.5", "harness", "longseal").cost == pytest.approx(60.0)


def test_estimate_names_missing_measurements():
    astra = estimate_cost(COSTS, "gpt-6-astra", "baseline", "hotpotqa")
    assert astra.cost is None and astra.note.startswith("missing:") and "gpt-6-astra" in astra.note
    verify = estimate_cost(COSTS, "opus-5.5", "verify_prompt", "hotpotqa")
    assert verify.cost is None and "verify_prompt" in verify.note


def test_projected_total_multiplies_by_n_and_six_faults():
    costs = {(model, "baseline", "hotpotqa"): 1.0 for model in ("opus-5.5", "haiku-5.5", "gpt-6-astra", "gpt-6-luna")}
    total, notes = projected_total(costs, "baseline", "hotpotqa", 25)
    assert total == pytest.approx(4 * 1.0 * 25 * 6)
    assert notes == []
    assert projected_total(COSTS, "baseline", "hotpotqa", 25)[0] is None  # gpt-6-astra is missing


def test_projection_names_missing_measurements(tmp_path):
    episodes = [make_episode(model, "baseline", "hotpotqa", "none", cost=0.001) for model in CHEAP]
    text = project_costs(load_run(write_run(tmp_path / "pilot-1", episodes)))
    assert "missing:" in text and "gpt-6-astra" in text


def test_llm_verifier_estimate_from_response_sizes():
    episode = make_episode(
        "haiku-5.5", "baseline", "hotpotqa", "none",
        question="Q" * 40, calls=[("search", "s" * 400, None), ("get_article", "a" * 800, None)],
    )
    # Inputs: (40 + 400) / 4 = 110 and (40 + 400 + 800) / 4 = 310 tokens; outputs: 150 tokens per call
    expected = (110 + 310) * 10.0 / 1e6 + 300 * 50.0 / 1e6
    assert verifier_cost([episode], "hotpotqa") == pytest.approx(expected)
    assert verifier_cost([episode], "longseal") is None


# Injection, flags and reflection

def test_injection_counts_only_finished_episodes(tmp_path):
    hit = make_episode("haiku-5.5", "baseline", "hotpotqa", "malformed", scenario="a")
    miss = make_episode("haiku-5.5", "baseline", "hotpotqa", "malformed", scenario="b")
    failed_attempt = make_episode("haiku-5.5", "baseline", "hotpotqa", "malformed", scenario="c")
    rows = [
        gt_row(hit, 0, "get_article", corrupted=True, original="{}", corrupted_text="<html>"),
        gt_row(miss, 0, "search"),
        gt_row(failed_attempt, 0, "get_article", corrupted=True, original="{}", corrupted_text="<html>"),
    ]
    data = load_run(write_run(tmp_path / "pilot-1", [hit, miss], rows))
    assert injection_counts(data) == {("malformed", "hotpotqa"): (1, 2)}


def test_flags_are_split_by_fault_and_by_corrupted_clean_and_tool_error_responses(tmp_path):
    episode = make_episode(
        "haiku-5.5", "harness", "hotpotqa", "malformed",
        calls=[
            ("search", 'clean\n\n_harness: {"flags": []}', []),
            ("get_article", '<html>broken\n\n_harness: {"flags": ["format: not valid JSON"]}', ["format: not valid JSON"]),
        ],
    )
    # A bad-argument error: flagged, but it is neither the corrupted response nor a false flag
    bad_args = make_episode(
        "haiku-5.5", "harness", "hotpotqa", "none", scenario="b",
        calls=[("get_article", "validation error\n\n_harness: {}", ["format: the tool returned an error"])],
    )
    bad_args.turns[0].tool_calls[0].is_error = True
    rows = [
        gt_row(episode, 0, "search"),
        gt_row(episode, 1, "get_article", corrupted=True, original="{}", corrupted_text="<html>broken"),
        gt_row(bad_args, 0, "get_article"),
    ]
    data = load_run(write_run(tmp_path / "pilot-1", [episode, bad_args], rows))
    assert flag_counts(data) == {
        ("harness", "malformed", "hotpotqa"): {"corrupted": (1, 1), "clean": (0, 1), "tool_errors": (0, 0)},
        ("harness", "none", "hotpotqa"): {"corrupted": (0, 0), "clean": (0, 0), "tool_errors": (1, 1)},
    }


def test_reflection_counts_tool_turns_with_a_thought():
    with_thought = make_episode("haiku-5.5", "verify_prompt", "hotpotqa", "none", scenario="a", calls=[("search", "r", None)])
    without = make_episode("haiku-5.5", "verify_prompt", "hotpotqa", "none", scenario="b", calls=[("search", "r", None)])
    without.turns[0].assistant_text = ""
    baseline = make_episode("haiku-5.5", "baseline", "hotpotqa", "none", calls=[("search", "r", None)])
    assert reflection_counts([with_thought, without, baseline]) == {"verify_prompt": (1, 2)}


def test_projection_discloses_what_it_leaves_out(tmp_path):
    episodes = [make_episode(model, "baseline", "hotpotqa", "none", cost=0.001) for model in CHEAP]
    text = project_costs(load_run(write_run(tmp_path / "pilot-1", episodes)))
    assert "judge" in text and "failed attempts" in text


# Missing inputs

def test_report_warns_when_the_ground_truth_log_is_missing(tmp_path, capsys):
    run_dir = write_run(tmp_path / "pilot-1", [make_episode("haiku-5.5", "baseline", "hotpotqa", "none")])
    (run_dir / "ground_truth.jsonl").unlink()

    assert main(["--run-id", "pilot-1", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "## Warnings" in out and "ground_truth.jsonl" in out


def test_report_counts_episodes_with_tool_calls_but_no_ground_truth(tmp_path, capsys):
    called = make_episode("haiku-5.5", "baseline", "hotpotqa", "malformed", calls=[("search", "r", None)])
    runner_only = make_episode("haiku-5.5", "baseline", "hotpotqa", "none", scenario="b", calls=[("search", INVALID_ARGUMENTS, None)])
    write_run(tmp_path / "pilot-1", [called, runner_only])  # an empty ground-truth log

    assert main(["--run-id", "pilot-1", "--runs-dir", str(tmp_path)]) == 0
    assert "1 finished episode(s) made tool calls but have no ground-truth rows" in capsys.readouterr().out


def test_unknown_run_id_is_an_error(tmp_path, capsys):
    assert main(["--run-id", "nope", "--runs-dir", str(tmp_path)]) == 2
    assert "nope" in capsys.readouterr().out


def test_report_prints_every_section(tmp_path, capsys):
    episode = make_episode("haiku-5.5", "harness", "hotpotqa", "none", cost=0.001, calls=[("search", 'r\n\n_harness: {"flags": []}', [])])
    error = {"episode_id": "pilot-1__haiku-5.5__harness__hotpotqa-q2-none__s0__a1", "scenario_id": "hotpotqa-q2-none", "attempt": 1, "error": "E"}
    write_run(tmp_path / "pilot-1", [episode], [gt_row(episode, 0, "search")], [error])

    assert main(["--run-id", "pilot-1", "--runs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for heading in ["## Episodes", "## Fault injection", "## Harness flags", "## Reflection", "## Projected full-run cost"]:
        assert heading in out
