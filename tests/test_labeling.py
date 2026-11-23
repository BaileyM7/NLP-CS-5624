"""Tests for the hand-labeling packet (#15): picking episodes, rendering them blind, and exporting the labels."""
import json

import pytest
from fakes import gt_row, make_episode, write_run

from toolharness.data.scenario import FaultType
from toolharness.eval.labeling import export, main, pick_episodes, render_episode, write_packet
from toolharness.injector.ground_truth import EpisodeSummary

CONDITIONS = ("baseline", "verify_prompt", "harness", "flags_only")
MODELS = ("haiku-5.5", "gpt-6-luna")


def summary(episode, injected):
    return EpisodeSummary(
        episode_id=episode.episode_id,
        scenario_id=episode.scenario_id,
        fault_type=episode.fault_type.value,
        calls=1,
        fault_injected=injected,
        injected_at_call=0 if injected else None,
        retries_of_corrupted=0,
    )


def pool():
    """Every fault x condition x dataset x model, all faults injected, plus faulted episodes where nothing fired."""
    episodes, gt = [], {}
    for fault in FaultType:
        for condition in CONDITIONS:
            for dataset in ("hotpotqa", "longseal"):
                for model in MODELS:
                    hit = make_episode(model, condition, dataset, fault.value, scenario=f"q-{fault.value}")
                    episodes.append(hit)
                    gt[hit.episode_id] = summary(hit, injected=fault != FaultType.NONE)
                    if fault != FaultType.NONE:
                        miss = make_episode(model, condition, dataset, fault.value, scenario=f"miss-{fault.value}")
                        episodes.append(miss)
                        gt[miss.episode_id] = summary(miss, injected=False)
    return episodes, gt


# Picking

def test_pick_covers_faults_conditions_longseal_and_none():
    episodes, gt = pool()
    picked = pick_episodes(episodes, gt)

    assert len(picked) == 30
    assert len({e.episode_id for e in picked}) == 30
    assert {e.fault_type for e in picked} == set(FaultType)
    assert {e.condition for e in picked} == set(CONDITIONS)
    assert sum(e.dataset == "longseal" for e in picked) >= 6
    assert sum(e.fault_type == FaultType.NONE for e in picked) >= 5
    for episode in picked:
        assert episode.fault_type == FaultType.NONE or gt[episode.episode_id].fault_injected


def test_pick_is_deterministic_for_a_seed():
    episodes, gt = pool()
    first = [e.episode_id for e in pick_episodes(episodes, gt, seed=0)]
    assert first == [e.episode_id for e in pick_episodes(episodes, gt, seed=0)]
    assert first != [e.episode_id for e in pick_episodes(episodes, gt, seed=1)]


def test_pick_with_too_few_episodes_takes_what_exists():
    episodes, gt = pool()
    few = [e for e in episodes if gt[e.episode_id].fault_injected][:3] + [e for e in episodes if e.fault_type == FaultType.NONE][:1]
    assert len(pick_episodes(few, gt)) == 4


# Rendering and the packet

def faulted_episode():
    return make_episode(
        "haiku-5.5", "harness", "hotpotqa", "malformed", question="Who wrote it?", gold="Ann", final="Bob",
        calls=[("get_article", '<html>broken\n\n_harness: {"flags": ["format: not valid JSON"]}', ["format: not valid JSON"])],
    )


def test_rendered_episode_shows_turns_and_the_ground_truth_note():
    episode = faulted_episode()
    rows = [gt_row(episode, 0, "get_article", corrupted=True, original='{"title": "Clean page"}', corrupted_text="<html>broken")]
    text = render_episode(7, episode, rows)

    for part in ["Item 7", "Who wrote it?", "Ann", "Thought: look it up.", "get_article", "<html>broken", "Bob", "What was corrupted"]:
        assert part in text
    assert '{"title": "Clean page"}' in text  # the original response
    assert "Turn 1, tool call 1" in text


def test_rendered_clean_episode_says_nothing_was_corrupted():
    episode = make_episode("haiku-5.5", "baseline", "hotpotqa", "none")
    assert "Nothing was corrupted" in render_episode(1, episode, [])


def test_packet_hides_model_and_condition(tmp_path):
    episodes, gt = pool()
    picked = pick_episodes(episodes, gt)
    write_packet(tmp_path, picked, {})

    folder = tmp_path / "labeling"
    texts = [path.read_text(encoding="utf-8") for path in folder.glob("*.md")]
    texts.append((folder / "labels.jsonl").read_text(encoding="utf-8"))
    assert len(texts) == 31
    for text in texts:
        for hidden in [*MODELS, "verify_prompt", "flags_only", "baseline", "__"]:
            assert hidden not in text
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    assert sorted(key.values()) == sorted(e.episode_id for e in picked)


def test_pick_refuses_without_a_ground_truth_log(tmp_path, capsys):
    run_dir = write_run(tmp_path / "pilot-1", [faulted_episode()])
    (run_dir / "ground_truth.jsonl").unlink()

    assert main(["pick", "--run-id", "pilot-1", "--runs-dir", str(tmp_path)]) == 2
    assert "ground_truth.jsonl" in capsys.readouterr().out
    assert not (run_dir / "labeling").exists()


def test_packet_is_never_overwritten(tmp_path):
    episodes, gt = pool()
    write_packet(tmp_path, pick_episodes(episodes, gt), {})
    with pytest.raises(FileExistsError):
        write_packet(tmp_path, pick_episodes(episodes, gt), {})


# Export

def labeled_run(tmp_path):
    """A run with one HotpotQA and one LongSeal episode, a packet for both, and filled labels."""
    hotpot = faulted_episode()
    longseal = make_episode("gpt-6-luna", "baseline", "longseal", "none")
    rows = [gt_row(hotpot, 0, "get_article", corrupted=True, original="{}", corrupted_text="<html>broken"), gt_row(longseal, 0, "search")]
    run_dir = write_run(tmp_path / "pilot-1", [hotpot, longseal], rows)
    write_packet(run_dir, [hotpot, longseal], {hotpot.episode_id: rows[:1], longseal.episode_id: rows[1:]})

    labels = []
    for line in (run_dir / "labeling" / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.update(labeler="bailey", detected=True, detection_turn=2, relied_on_corrupted=False, answer_correct=True)
        labels.append(row)
    return run_dir, hotpot, longseal, labels


def write_labels(run_dir, labels):
    text = "".join(json.dumps(row) + "\n" for row in labels)
    (run_dir / "labeling" / "labels.jsonl").write_text(text, encoding="utf-8")


def test_export_writes_labels_and_only_hotpotqa_traces(tmp_path):
    run_dir, hotpot, longseal, labels = labeled_run(tmp_path)
    write_labels(run_dir, labels)
    out = tmp_path / "data" / "labels"
    export(run_dir, out)

    exported = [json.loads(line) for line in (out / "pilot-1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(row["episode_id"] for row in exported) == sorted([hotpot.episode_id, longseal.episode_id])
    assert all("item" not in row for row in exported)

    traces = (out / "pilot-1_hotpotqa_traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["episode_id"] for line in traces] == [hotpot.episode_id]
    truth = (out / "pilot-1_hotpotqa_ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["episode_id"] for line in truth] == [hotpot.episode_id]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("labeler", "", "labeler"),
        ("detected", None, "detected"),
        ("relied_on_corrupted", None, "relied_on_corrupted"),
        ("answer_correct", None, "answer_correct"),
    ],
)
def test_export_refuses_unfilled_rows(tmp_path, field, value, message):
    run_dir, hotpot, longseal, labels = labeled_run(tmp_path)
    key = json.loads((run_dir / "labeling" / "key.json").read_text(encoding="utf-8"))
    # detected is required on the injected HotpotQA episode; answer_correct on the LongSeal one
    target = hotpot.episode_id if field != "answer_correct" else longseal.episode_id
    for row in labels:
        if key[str(row["item"])] == target:
            row[field] = value
    write_labels(run_dir, labels)

    with pytest.raises(ValueError, match=message):
        export(run_dir, tmp_path / "out")
