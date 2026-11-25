"""Offline tests for scripts/run_pilot.py: which commands each pilot stage runs."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

HOTPOTQA = "data/scenarios/hotpotqa.jsonl"
LONGSEAL = "data/scenarios/longseal.jsonl"
FOUR_CONDITIONS = {"baseline", "verify_prompt", "harness", "flags_only"}


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_pilot.py"
    spec = importlib.util.spec_from_file_location("run_pilot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(command, name):
    """The values after a flag, up to the next flag."""
    values = []
    for part in command[command.index(name) + 1 :]:
        if part.startswith("--"):
            break
        values.append(part)
    return values


def check_stage(commands, run_id, questions, scenarios, models, conditions):
    assert len(commands) == len(models) * len(conditions)
    pairs = set()
    for command in commands:
        assert command[:3] == ["py", "-m", "toolharness.agent"]
        assert option(command, "--run-id") == [run_id]
        assert option(command, "--questions") == [str(questions)]
        assert option(command, "--scenarios") == scenarios
        pairs.add((option(command, "--model")[0], option(command, "--condition")[0]))
    assert pairs == {(model, condition) for model in models for condition in conditions}


def test_stage_commands():
    module = load_script()
    cheap_models = {"haiku-5.5", "gpt-6-luna"}
    check_stage(module.stage_commands("smoke", "py"), "pilot-smoke", 1, [HOTPOTQA, LONGSEAL], cheap_models, FOUR_CONDITIONS)
    check_stage(module.stage_commands("cheap", "py"), "pilot-2", 5, [HOTPOTQA, LONGSEAL], cheap_models, FOUR_CONDITIONS)
    # The verifier stage measures the same 5 questions as cheap, with Haiku as the only agent
    check_stage(module.stage_commands("verifier", "py"), "pilot-2", 5, [HOTPOTQA, LONGSEAL], {"haiku-5.5"}, {"llm_verifier"})
    # Opus is only checked on HotpotQA before the final runs; Astra is no longer in the experiment
    check_stage(module.stage_commands("opus-check", "py"), "pilot-2", 2, [HOTPOTQA], {"opus-5.5"}, {"baseline", "harness"})
    assert "big" not in module.STAGES


def test_stages_skip_longseal_contradiction():
    module = load_script()
    for stage in ("smoke", "cheap", "verifier"):
        for command in module.stage_commands(stage, "py"):
            assert option(command, "--skip") == ["longseal:contradiction"]
    for command in module.stage_commands("opus-check", "py"):
        assert "--skip" not in command  # HotpotQA only


def test_stages_leave_concurrency_to_the_runner_unless_given():
    module = load_script()
    for stage in module.STAGES:
        for command in module.stage_commands(stage, "py"):
            assert "--concurrency" not in command  # the runner picks 1 or 4 per model and condition
    for command in module.stage_commands("cheap", "py", concurrency=3):
        assert option(command, "--concurrency") == ["3"]
    assert not hasattr(module, "default_concurrency")


def test_run_id_and_server_url_can_be_overridden():
    module = load_script()
    commands = module.stage_commands("smoke", "py", run_id="pilot-smoke-2", server_url="http://127.0.0.1:9000/mcp")
    for command in commands:
        assert option(command, "--run-id") == ["pilot-smoke-2"]
        assert option(command, "--server-url") == ["http://127.0.0.1:9000/mcp"]
    assert "--server-url" not in module.stage_commands("smoke", "py")[0]


def test_concurrency_can_be_set():
    module = load_script()
    for command in module.stage_commands("cheap", "py", concurrency=1):
        assert option(command, "--concurrency") == ["1"]
    assert "--concurrency" not in module.stage_commands("cheap", "py")[0]


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        load_script().stage_commands("everything", "py")


def test_dry_run_prints_the_commands_without_running_them(monkeypatch, capsys):
    module = load_script()

    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must not run anything")

    monkeypatch.setattr(module.subprocess, "run", refuse)
    assert module.main(["smoke", "--dry-run"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 8


def test_stops_at_the_first_failing_command(monkeypatch):
    module = load_script()
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.main(["cheap"]) == 1
    assert len(calls) == 2
