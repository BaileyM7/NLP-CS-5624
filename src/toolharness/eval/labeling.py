"""Hand-labeling packet for the pilot (#15): pick about 30 episodes, render them without model or condition,
and export the filled labels. People label; the labels check the LLM judge (#23), so a model must never fill them in.

Usage:
    python -m toolharness.eval.labeling pick --run-id pilot-1 [--n 30 --seed 0]
    python -m toolharness.eval.labeling export --run-id pilot-1 --out data/labels
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from ..data.scenario import FaultType
from ..injector.ground_truth import EpisodeSummary
from .pilot_report import corrupted_call, load_run
from .trace import Episode


def label_template(item: int) -> dict:
    """An empty label row; the fields follow the experiment spec's definitions."""
    return {
        "item": item,
        "labeler": "",
        "detected": None,  # true/false; null only when nothing was corrupted
        "detection_turn": None,  # absolute trace turn of the first detection, or null
        "false_alarm_turns": [],  # turns that wrongly treat a clean response as unreliable
        "relied_on_corrupted": None,  # true/false; null only when nothing was corrupted
        "answer_correct": None,  # true/false; required for LongSeal (HotpotQA uses exact match)
        "notes": "",  # never quote LongSeal text here
    }


# Picking

def round_robin(items: list, key: Callable, limit: int) -> list:
    """Take one item per key in turn (keys in sorted order) until `limit` items are taken or none are left."""
    groups: dict = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    picked = []
    while len(picked) < limit and any(groups.values()):
        for group_key in sorted(groups):
            if len(picked) == limit:
                break
            if groups[group_key]:
                picked.append(groups[group_key].pop(0))
    return picked


def pick_episodes(
    episodes: list[Episode],
    gt: dict[str, EpisodeSummary],
    n: int = 30,
    n_none: int = 5,
    min_longseal: int = 6,
    seed: int = 0,
) -> list[Episode]:
    """A seeded pick spread over fault types, conditions, models and datasets. Short pools give fewer episodes."""
    rng = random.Random(seed)

    # Clean episodes always qualify; faulted ones only where the fault actually fired
    eligible = []
    for episode in episodes:
        summary = gt.get(episode.episode_id)
        if episode.fault_type == FaultType.NONE or (summary is not None and summary.fault_injected):
            eligible.append(episode)
    eligible.sort(key=lambda e: e.episode_id)
    rng.shuffle(eligible)

    # LongSeal first, one per fault type in turn
    longseal = [e for e in eligible if e.dataset == "longseal"]
    picked = round_robin(longseal, lambda e: e.fault_type.value, min_longseal)
    taken = {e.episode_id for e in picked}

    # Then faulted episodes, one per (fault, condition) cell in turn
    faulted_left = (n - n_none) - sum(e.fault_type != FaultType.NONE for e in picked)
    faulted = [e for e in eligible if e.episode_id not in taken and e.fault_type != FaultType.NONE]
    picked += round_robin(faulted, lambda e: (e.fault_type.value, e.condition), faulted_left)
    taken = {e.episode_id for e in picked}

    # Then clean episodes, alternating models, up to n
    clean = [e for e in eligible if e.episode_id not in taken and e.fault_type == FaultType.NONE]
    picked += round_robin(clean, lambda e: e.model, n - len(picked))

    rng.shuffle(picked)
    return picked


# Rendering

def ground_truth_note(episode: Episode, gt_rows: list[dict]) -> list[str]:
    """Which call was corrupted, with the clean and returned responses: the same facts the judge's behavior pass gets."""
    corrupted = next((row for row in gt_rows if row["corrupted"]), None)
    if corrupted is None:
        return ["Nothing was corrupted in this episode."]
    where = corrupted_call(episode, gt_rows)
    place = f"Turn {where[0]}, tool call {where[1]}" if where else "a call that isn't in the trace"
    return [
        f"Corrupted: {place} (`{corrupted['tool']}` {json.dumps(corrupted['args'])}).",
        "",
        "What a clean call returns:",
        "",
        "```text",
        str(corrupted["original"]),
        "```",
        "",
        "What the agent got:",
        "",
        "```text",
        str(corrupted["corrupted_text"]),
        "```",
    ]


def render_episode(item: int, episode: Episode, gt_rows: list[dict]) -> str:
    """Everything the agent saw, in order, without the model or condition; the ground-truth note goes last."""
    lines = [f"# Item {item}", "", f"**Question:** {episode.question}", "", f"**Gold answer:** {episode.gold_answer}", ""]
    for turn in episode.turns:
        lines += [f"## Turn {turn.index}", ""]
        if turn.assistant_text:
            lines += [turn.assistant_text, ""]
        for number, call in enumerate(turn.tool_calls, start=1):
            lines += [f"**Tool call {number}:** `{call.tool_name}` {json.dumps(call.tool_args)}", "", "```text", call.raw_tool_response, "```", ""]
    lines += ["## Final answer", "", f"{episode.final_answer or '(no answer)'} (stopped by: {episode.stop_reason})", ""]
    lines += ["## What was corrupted (for labeling only; the agent never saw this)", ""]
    lines += ground_truth_note(episode, gt_rows)
    return "\n".join(lines) + "\n"


def write_packet(run_dir, picked: list[Episode], gt_rows: dict[str, list[dict]]) -> Path:
    """Write labeling/NN.md, labels.jsonl (one empty row per item) and key.json (item -> episode ID)."""
    folder = Path(run_dir) / "labeling"
    if (folder / "labels.jsonl").exists():
        raise FileExistsError(f"{folder} already has a labeling packet; move it away before picking again")
    folder.mkdir(parents=True, exist_ok=True)

    key = {}
    with (folder / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as labels:
        for item, episode in enumerate(picked, start=1):
            key[str(item)] = episode.episode_id
            text = render_episode(item, episode, gt_rows.get(episode.episode_id, []))
            (folder / f"{item:02d}.md").write_text(text, encoding="utf-8", newline="\n")
            labels.write(json.dumps(label_template(item)) + "\n")
    (folder / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8", newline="\n")
    return folder


# Export

def check_label(item: int, label: dict, episode: Episode, summary: EpisodeSummary | None) -> None:
    """Raise ValueError if a label row is missing something it needs."""
    if not str(label.get("labeler") or "").strip():
        raise ValueError(f"item {item}: labeler is empty")
    injected = summary is not None and summary.fault_injected
    if injected and label.get("detected") is None:
        raise ValueError(f"item {item}: detected must be true or false when a fault was injected")
    if injected and label.get("relied_on_corrupted") is None:
        raise ValueError(f"item {item}: relied_on_corrupted must be true or false when a fault was injected")
    if episode.dataset == "longseal" and label.get("answer_correct") is None:
        raise ValueError(f"item {item}: answer_correct is required for LongSeal")


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


def export(run_dir, out_dir) -> None:
    """Write <run>.jsonl (labels by episode ID), plus the labeled HotpotQA traces and their ground-truth rows.

    LongSeal traces are never exported: they quote benchmark text and stay in the run folder.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    data = load_run(run_dir)
    folder = run_dir / "labeling"
    key = json.loads((folder / "key.json").read_text(encoding="utf-8"))
    by_id = {episode.episode_id: episode for episode in data.episodes}

    rows = []
    for line in (folder / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        item = label.pop("item")
        episode_id = key[str(item)]
        check_label(item, label, by_id[episode_id], data.gt.get(episode_id))
        rows.append({"episode_id": episode_id, **label})

    hotpot_ids = [row["episode_id"] for row in rows if by_id[row["episode_id"]].dataset == "hotpotqa"]
    truth = []
    for episode_id in hotpot_ids:
        for row in data.gt_rows.get(episode_id, []):
            truth.append(json.dumps(row))

    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    write_lines(out_dir / f"{name}.jsonl", [json.dumps(row) for row in rows])
    write_lines(out_dir / f"{name}_hotpotqa_traces.jsonl", [by_id[i].model_dump_json() for i in hotpot_ids])
    write_lines(out_dir / f"{name}_hotpotqa_ground_truth.jsonl", truth)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or export the pilot's hand-labeling packet.")
    commands = parser.add_subparsers(dest="command", required=True)
    pick = commands.add_parser("pick", help="pick episodes and write runs/<run-id>/labeling/")
    pick.add_argument("--run-id", required=True)
    pick.add_argument("--runs-dir", default="runs")
    pick.add_argument("--n", type=int, default=30)
    pick.add_argument("--seed", type=int, default=0)
    out = commands.add_parser("export", help="check the filled labels and write them, with HotpotQA traces, to --out")
    out.add_argument("--run-id", required=True)
    out.add_argument("--runs-dir", default="runs")
    out.add_argument("--out", default="data/labels")
    args = parser.parse_args(argv)

    run_dir = Path(args.runs_dir) / args.run_id
    try:
        data = load_run(run_dir)
    except FileNotFoundError as error:
        print(error)
        return 2

    if args.command == "pick":
        # Without the ground-truth log every fault looks uninjected, so the pick would hold only clean episodes
        if not data.gt_found:
            print(f"No ground-truth log at {data.gt_path}; can't tell which faults fired. Copy it there first.")
            return 2
        picked = pick_episodes(data.episodes, data.gt, n=args.n, seed=args.seed)
        folder = write_packet(run_dir, picked, data.gt_rows)
        print(f"Wrote {len(picked)} episodes to {folder}. Fill in labels.jsonl (see docs/pilot.md), then run export.")
    else:
        export(run_dir, args.out)
        print(f"Exported labels and HotpotQA traces to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
