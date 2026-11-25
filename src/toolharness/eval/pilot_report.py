"""Pilot report (#15): costs, fault injection, harness flags and reflection for one run, plus a full-run cost projection.

Usage:
    python -m toolharness.eval.pilot_report --run-id pilot-1
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Sequence

from ..agent.runner import INVALID_ARGUMENTS, NOT_RUN
from ..data.scenario import FaultType
from ..harness.checks import HARNESS_CONDITIONS
from ..injector.ground_truth import EpisodeSummary, summarize_gt
from ..llm.models import MAIN_RUN_DATASETS, MAIN_RUN_MODELS, MODELS, VERIFIER_MODEL
from .trace import Episode, read_jsonl

CHEAP_MODELS = ("haiku-5.5", "gpt-6-luna")  # run on every condition and both datasets, so their ratios scale the rest
DATASETS = ("hotpotqa", "longseal")
# Fault types each question runs under, `none` included; LongSeal leaves out contradiction, which almost never fires there
FAULT_TYPES = {"hotpotqa": len(FaultType), "longseal": len(FaultType) - 1}
MAIN_CONDITIONS = ("baseline", "verify_prompt", "harness")
OPTIONAL_CONDITIONS = ("flags_only",)
QUESTION_COUNTS = {"hotpotqa": (25, 50, 100), "longseal": (15, 30)}
CHARS_PER_TOKEN = 4
VERIFIER_OUTPUT_TOKENS = 150


@dataclass
class RunData:
    episodes: list[Episode]  # finished episodes from every trace file
    gt: dict[str, EpisodeSummary]
    gt_rows: dict[str, list[dict]]  # raw ground-truth rows, by episode ID
    errors: list[dict]  # failed attempts from errors.jsonl
    gt_path: Path  # where the tool server's ground-truth log should be
    gt_found: bool  # False when that file doesn't exist


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(run_dir) -> RunData:
    """Read a run folder: traces, the tool server's ground-truth log, and errors."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no run folder at {run_dir}; check --run-id and --runs-dir")
    episodes = []
    for path in sorted((run_dir / "traces").glob("*.jsonl")):
        episodes.extend(read_jsonl(path))

    gt_path = run_dir / "ground_truth.jsonl"
    gt_rows: dict[str, list[dict]] = defaultdict(list)
    for row in read_rows(gt_path):
        gt_rows[row["episode_id"]].append(row)
    gt = summarize_gt(gt_path) if gt_path.exists() else {}
    return RunData(episodes, gt, dict(gt_rows), read_rows(run_dir / "errors.jsonl"), gt_path, gt_path.exists())


def ground_truth_warnings(data: RunData) -> list[str]:
    """Signs that the ground-truth log is missing or belongs to another run, which would make every fault look uninjected."""
    if not data.gt_found:
        return [f"No ground-truth log at {data.gt_path}. Every fault looks uninjected; start the tool server with --gt-log {data.gt_path}."]

    # An episode whose calls reached the server must have ground-truth rows; runner-made results never reach it
    without = 0
    for episode in data.episodes:
        reached_server = False
        for turn in episode.turns:
            for call in turn.tool_calls:
                if call.raw_tool_response not in (NOT_RUN, INVALID_ARGUMENTS):
                    reached_server = True
        if reached_server and episode.episode_id not in data.gt_rows:
            without += 1
    if without:
        return [f"{without} finished episode(s) made tool calls but have no ground-truth rows in {data.gt_path}. Was the server started with another --gt-log?"]
    return []


# Answers

def normalize_answer(text: str) -> str:
    """HotpotQA's normalization: lowercase, no punctuation, no articles, single spaces."""
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, gold: str) -> bool:
    return prediction is not None and normalize_answer(prediction) == normalize_answer(gold)


# Costs and the projection

def mean_costs(episodes: list[Episode]) -> dict[tuple[str, str, str], float]:
    """Mean agent cost per finished episode, by (model, condition, dataset)."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.model, episode.condition, episode.dataset)].append(episode.totals.agent_cost_usd)
    return {key: mean(values) for key, values in groups.items()}


@dataclass(frozen=True)
class Estimate:
    cost: float | None  # USD per episode; None when a measurement it needs is missing
    note: str  # "measured", how it was scaled, or "missing: <what>"


def cheap_ratio(costs: dict, top: tuple[str, str], bottom: tuple[str, str]) -> float | None:
    """Mean over the cheap models of cost(top) / cost(bottom), each a (condition, dataset) pair."""
    ratios = []
    for model in CHEAP_MODELS:
        upper = costs.get((model, *top))
        lower = costs.get((model, *bottom))
        if upper is not None and lower:
            ratios.append(upper / lower)
    return mean(ratios) if ratios else None


def hotpot_cost(costs: dict, model: str, condition: str) -> Estimate:
    """HotpotQA cost per episode: measured, or the model's baseline scaled by the cheap models' condition ratio."""
    measured = costs.get((model, condition, "hotpotqa"))
    if measured is not None:
        return Estimate(measured, "measured")
    base = costs.get((model, "baseline", "hotpotqa"))
    if base is None:
        return Estimate(None, f"missing: {model} baseline on hotpotqa")
    ratio = cheap_ratio(costs, (condition, "hotpotqa"), ("baseline", "hotpotqa"))
    if ratio is None:
        return Estimate(None, f"missing: cheap-model {condition} on hotpotqa")
    return Estimate(base * ratio, f"{model} baseline x{ratio:.2f} (cheap models' {condition}/baseline)")


def estimate_cost(costs: dict, model: str, condition: str, dataset: str) -> Estimate:
    """Cost per episode: measured, or scaled from HotpotQA by the cheap models' dataset ratio."""
    measured = costs.get((model, condition, dataset))
    if measured is not None:
        return Estimate(measured, "measured")
    hotpot = hotpot_cost(costs, model, condition)
    if dataset == "hotpotqa" or hotpot.cost is None:
        return hotpot
    ratio = cheap_ratio(costs, (condition, dataset), (condition, "hotpotqa"))
    if ratio is None:
        return Estimate(None, f"missing: cheap-model {condition} on {dataset}")
    return Estimate(hotpot.cost * ratio, f"{hotpot.note}, then x{ratio:.2f} for {dataset} (cheap models)")


def planned_models(dataset: str) -> list[str]:
    """The main-run models the full experiment runs on this dataset (Opus: HotpotQA only)."""
    return [model for model in MAIN_RUN_MODELS if dataset in MAIN_RUN_DATASETS[model]]


def projected_total(costs: dict, condition: str, dataset: str, n: int) -> tuple[float | None, list[str]]:
    """Cost of N questions x every fault type, summed over the models planned for this dataset; None if any can't be estimated."""
    total = 0.0
    notes = []
    for model in planned_models(dataset):
        estimate = estimate_cost(costs, model, condition, dataset)
        if estimate.cost is None:
            return None, [estimate.note]
        if estimate.note != "measured":
            notes.append(f"{model} {condition} on {dataset}: {estimate.note}")
        total += estimate.cost * n * FAULT_TYPES[dataset]
    return total, notes


def verifier_cost(episodes: list[Episode], dataset: str) -> float | None:
    """Estimated LLM-verifier (#29) cost per episode: one call per tool response, seeing the question and every response so far."""
    spec = MODELS[VERIFIER_MODEL]
    per_episode = []
    for episode in episodes:
        if episode.model not in CHEAP_MODELS or episode.condition != "baseline" or episode.dataset != dataset:
            continue
        seen_chars = len(episode.question)
        cost = 0.0
        for turn in episode.turns:
            for call in turn.tool_calls:
                seen_chars += len(call.raw_tool_response)
                cost += seen_chars / CHARS_PER_TOKEN * spec.usd_per_m_input / 1e6
                cost += VERIFIER_OUTPUT_TOKENS * spec.usd_per_m_output / 1e6
        per_episode.append(cost)
    return mean(per_episode) if per_episode else None


def measured_verifier_cost(episodes: list[Episode], dataset: str) -> float | None:
    """The cheap models' mean measured verifier cost per llm_verifier episode on a dataset."""
    costs = []
    for episode in episodes:
        if episode.model in CHEAP_MODELS and episode.condition == "llm_verifier" and episode.dataset == dataset:
            costs.append(episode.totals.verifier_cost_usd)
    return mean(costs) if costs else None


def llm_verifier_total(data: RunData, costs: dict, dataset: str, n: int) -> tuple[float | None, list[str]]:
    """Projected llm_verifier cost: measured where a model ran it; otherwise a scaled agent part plus a flat verifier part.

    The verifier is always the same model, so its cost is never scaled by the agent model's price.
    """
    verifier_measured = measured_verifier_cost(data.episodes, dataset)
    verifier_estimate = verifier_cost(data.episodes, dataset)
    total = 0.0
    notes = []
    for model in planned_models(dataset):
        measured = [
            e.totals.agent_cost_usd + e.totals.verifier_cost_usd
            for e in data.episodes
            if e.model == model and e.condition == "llm_verifier" and e.dataset == dataset
        ]
        if measured:
            total += mean(measured) * n * FAULT_TYPES[dataset]
            continue

        # Agent part: scaled like any condition (costs hold agent cost only), else the model's baseline
        agent = estimate_cost(costs, model, "llm_verifier", dataset)
        basis = "agent"
        if agent.cost is None:
            agent = estimate_cost(costs, model, "baseline", dataset)
            basis = "agent from baseline (no llm_verifier measurement to scale from)"
        if agent.cost is None:
            return None, [agent.note]

        # Verifier part: the cheap models' measured cost, else the estimate from response sizes
        if verifier_measured is not None:
            verifier, how = verifier_measured, "measured cheap-model verifier cost"
        elif verifier_estimate is not None:
            verifier, how = verifier_estimate, "estimated verifier cost (from response sizes)"
        else:
            return None, [f"missing: cheap-model llm_verifier or baseline on {dataset} (for the verifier part)"]
        notes.append(f"{model} llm_verifier on {dataset}: {basis} {agent.note}, plus {how}")
        total += (agent.cost + verifier) * n * FAULT_TYPES[dataset]
    return total, notes


def money(value: float | None, notes: list[str]) -> str:
    if value is None:
        return notes[0] if notes else "missing"
    return f"${value:,.2f}"


def project_costs(data: RunData) -> str:
    """The projected full-run cost per dataset, condition and question count, with every scaling it applied."""
    costs = mean_costs(data.episodes)
    lines = [
        "## Projected full-run cost",
        f"HotpotQA: {', '.join(planned_models('hotpotqa'))}; LongSeal: {', '.join(planned_models('longseal'))}. "
        f"{FAULT_TYPES['hotpotqa']} fault types per HotpotQA question and {FAULT_TYPES['longseal']} per LongSeal question "
        "(contradiction: HotpotQA only), one seed, no repeats.",
    ]
    all_notes: list[str] = []
    for dataset in DATASETS:
        counts = QUESTION_COUNTS[dataset]
        lines += ["", f"### {dataset}", "| Condition | " + " | ".join(f"N={n}" for n in counts) + " |", "|---" * (len(counts) + 1) + "|"]
        main_totals = [0.0 for _ in counts]

        rows = [(condition, condition) for condition in MAIN_CONDITIONS]
        rows += [(condition, f"{condition} (optional)") for condition in OPTIONAL_CONDITIONS]
        for condition, label in rows:
            cells = []
            for i, n in enumerate(counts):
                total, notes = projected_total(costs, condition, dataset, n)
                all_notes += notes
                cells.append(money(total, notes))
                if condition in MAIN_CONDITIONS:
                    main_totals[i] = None if total is None or main_totals[i] is None else main_totals[i] + total
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        cells = []
        for i, n in enumerate(counts):
            total, notes = llm_verifier_total(data, costs, dataset, n)
            all_notes += notes
            cells.append(money(total, notes))
            main_totals[i] = None if total is None or main_totals[i] is None else main_totals[i] + total
        lines.append("| llm_verifier (measured where available; otherwise estimated) | " + " | ".join(cells) + " |")
        lines.append("| **Main total** (all but flags_only) | " + " | ".join(money(total, []) for total in main_totals) + " |")

    lines += [
        "",
        "- **Not included:** the answer judge (#23: two Opus passes per episode, likely adding about half again on top of the "
        "agent cost), the fact-edit model, and failed attempts that were retried (their spend is in llm_usage.jsonl, not the traces).",
        "- HotpotQA N=100 needs new scenarios plus new fact edits and their review; the current file has 50 questions. LongSeal has 30.",
        "- llm_verifier for models that didn't run it: the agent part is scaled like any condition; the verifier part is the "
        f"cheap models' measured {VERIFIER_MODEL} cost per episode, added unscaled (the verifier is the same model for every agent). "
        f"Without that measurement it is estimated: one call per tool response, input = question + every response so far at "
        f"{CHARS_PER_TOKEN} characters per token, output = {VERIFIER_OUTPUT_TOKENS} tokens.",
    ]
    for note in sorted(set(all_notes)):
        if not note.startswith("missing:"):
            lines.append(f"- Scaled: {note}")
    return "\n".join(lines)


# Fault injection, harness flags and reflection

def injection_counts(data: RunData) -> dict[tuple[str, str], tuple[int, int]]:
    """(injected, episodes) per (fault, dataset), over finished episodes only."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for episode in data.episodes:
        key = (episode.fault_type.value, episode.dataset)
        counts[key][1] += 1
        summary = data.gt.get(episode.episode_id)
        if summary is not None and summary.fault_injected:
            counts[key][0] += 1
    return {key: (value[0], value[1]) for key, value in counts.items()}


def corrupted_call(episode: Episode, gt_rows: list[dict]) -> tuple[int, int] | None:
    """(turn index, call number from 1) of the tool call that got the corrupted response, or None."""
    corrupted = next((row for row in gt_rows if row["corrupted"]), None)
    if corrupted is None:
        return None
    # The first call to that tool whose response starts with the corrupted text (later retries are clean)
    for turn in episode.turns:
        for number, call in enumerate(turn.tool_calls, start=1):
            if call.tool_name == corrupted["tool"] and call.raw_tool_response.startswith(corrupted["corrupted_text"]):
                return turn.index, number
    return None


def flag_counts(data: RunData) -> dict[tuple[str, str, str], dict[str, tuple[int, int]]]:
    """(flagged, total) tool responses per (harness condition, fault, dataset), in four kinds:
    the corrupted response; clean responses before it (or in an episode without one); clean responses after it,
    which can genuinely conflict with it; and tool errors that weren't corrupted (e.g. bad arguments)."""
    counts: dict[tuple[str, str, str], dict[str, list[int]]] = {}
    for episode in data.episodes:
        if episode.condition not in HARNESS_CONDITIONS:
            continue
        key = (episode.condition, episode.fault_type.value, episode.dataset)
        bucket = counts.setdefault(key, {"corrupted": [0, 0], "clean": [0, 0], "after_corrupted": [0, 0], "tool_errors": [0, 0]})
        where = corrupted_call(episode, data.gt_rows.get(episode.episode_id, []))
        for turn in episode.turns:
            for number, call in enumerate(turn.tool_calls, start=1):
                if call.harness_flags is None:  # runner-made results never went through the harness
                    continue
                if where == (turn.index, number):
                    kind = "corrupted"
                elif call.is_error:
                    kind = "tool_errors"
                elif where is not None and (turn.index, number) > where:
                    kind = "after_corrupted"
                else:
                    kind = "clean"
                bucket[kind][1] += 1
                if call.harness_flags:
                    bucket[kind][0] += 1
    return {key: {kind: (v[0], v[1]) for kind, v in value.items()} for key, value in counts.items()}


def verifier_counts(data: RunData) -> dict[tuple[str, str], dict[str, tuple[int, int]]]:
    """(suspicious, total) LLM-verifier verdicts per (fault, dataset), in the same four kinds as the harness flags."""
    counts: dict[tuple[str, str], dict[str, list[int]]] = {}
    for episode in data.episodes:
        if episode.condition != "llm_verifier":
            continue
        key = (episode.fault_type.value, episode.dataset)
        bucket = counts.setdefault(key, {"corrupted": [0, 0], "clean": [0, 0], "after_corrupted": [0, 0], "tool_errors": [0, 0]})
        where = corrupted_call(episode, data.gt_rows.get(episode.episode_id, []))
        for turn in episode.turns:
            for number, call in enumerate(turn.tool_calls, start=1):
                if call.verifier_output is None:  # runner-made results never reached the verifier
                    continue
                if where == (turn.index, number):
                    kind = "corrupted"
                elif call.is_error:
                    kind = "tool_errors"
                elif where is not None and (turn.index, number) > where:
                    kind = "after_corrupted"
                else:
                    kind = "clean"
                bucket[kind][1] += 1
                if call.verifier_output.suspicious:
                    bucket[kind][0] += 1
    return {key: {kind: (v[0], v[1]) for kind, v in value.items()} for key, value in counts.items()}


def reflection_counts(episodes: list[Episode]) -> dict[tuple[str, str], dict[str, int]]:
    """Per model and verification condition: tool-calling turns, those with a thought, and those where every thought starts with "Check:"."""
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for episode in episodes:
        if episode.condition not in ("verify_prompt", "harness"):
            continue
        for turn in episode.turns:
            if not turn.tool_calls:
                continue
            count = counts.setdefault((episode.model, episode.condition), {"turns": 0, "with_thought": 0, "check_first": 0})
            count["turns"] += 1
            thoughts = [line for line in turn.assistant_text.splitlines() if line.startswith("Thought: ")]
            if thoughts:
                count["with_thought"] += 1
            if thoughts and all(line.startswith("Thought: Check:") for line in thoughts):
                count["check_first"] += 1
    return counts


# Tables

def share(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "0/0"


def episode_table(data: RunData) -> str:
    """One row per model, condition and dataset: size, errors, cost, tokens, turns, accuracy and stop reasons."""
    groups: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
    for episode in data.episodes:
        groups[(episode.model, episode.condition, episode.dataset)].append(episode)

    # Failed attempts, by the model and condition in their episode ID
    errors: Counter = Counter()
    for row in data.errors:
        parts = row["episode_id"].split("__")
        errors[(parts[1], parts[2], row["scenario_id"].split("-")[0])] += 1

    lines = [
        "## Episodes",
        "| Model | Condition | Dataset | Episodes | Failed attempts | Total cost/episode | Output tokens/episode | Turns/episode | Exact match | Stop reasons |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(set(groups) | set(errors)):
        episodes = groups.get(key, [])
        model, condition, dataset = key
        if episodes:
            cost = f"${mean(e.totals.agent_cost_usd + e.totals.verifier_cost_usd for e in episodes):.5f}"
            tokens = f"{mean(e.totals.agent_tokens.output_tokens for e in episodes):.0f}"
            turns = f"{mean(len(e.turns) for e in episodes):.1f}"
            stops = ", ".join(f"{reason} {count}" for reason, count in Counter(e.stop_reason for e in episodes).most_common())
        else:
            cost = tokens = turns = stops = "-"
        matched = "-"
        if dataset == "hotpotqa" and episodes:
            matched = share(sum(exact_match(e.final_answer, e.gold_answer) for e in episodes), len(episodes))
        lines.append(f"| {model} | {condition} | {dataset} | {len(episodes)} | {errors[key]} | {cost} | {tokens} | {turns} | {matched} | {stops} |")
    return "\n".join(lines)


def injection_table(data: RunData) -> str:
    lines = ["## Fault injection", "| Fault | Dataset | Injected / episodes |", "|---|---|---|"]
    for (fault, dataset), (injected, total) in sorted(injection_counts(data).items()):
        lines.append(f"| {fault} | {dataset} | {share(injected, total)} |")
    return "\n".join(lines)


def flag_table(data: RunData) -> str:
    lines = [
        "## Harness flags",
        "Share of tool responses with at least one flag. hard_failure, malformed and stale should be flagged every time; "
        "the consistency check catches contradiction and part of silently_wrong and stale. Flags on clean responses are "
        "the checks' false flags. A clean response after the corrupted one can genuinely conflict with it, so it is counted apart. "
        "Tool errors are error results that weren't corrupted (e.g. bad arguments): flagged correctly, but not a fault.",
        "| Condition | Fault | Dataset | Corrupted response flagged | Clean responses flagged | Clean, after the corrupted one, flagged | Tool errors flagged |",
        "|---|---|---|---|---|---|---|",
    ]
    for (condition, fault, dataset), counts in sorted(flag_counts(data).items()):
        cells = " | ".join(share(*counts[kind]) for kind in ("corrupted", "clean", "after_corrupted", "tool_errors"))
        lines.append(f"| {condition} | {fault} | {dataset} | {cells} |")
    return "\n".join(lines)


def verifier_table(data: RunData) -> str:
    lines = [
        "## LLM verifier verdicts",
        "Share of tool responses the verifier called suspicious, in the llm_verifier condition. Suspicious verdicts on clean "
        "responses are its false alarms; a clean response after the corrupted one can genuinely conflict with it.",
        "| Fault | Dataset | Corrupted response suspicious | Clean responses suspicious | Clean, after the corrupted one, suspicious | Tool errors suspicious |",
        "|---|---|---|---|---|---|",
    ]
    for (fault, dataset), counts in sorted(verifier_counts(data).items()):
        cells = " | ".join(share(*counts[kind]) for kind in ("corrupted", "clean", "after_corrupted", "tool_errors"))
        lines.append(f"| {fault} | {dataset} | {cells} |")
    return "\n".join(lines)


def reflection_table(data: RunData) -> str:
    lines = [
        "## Reflection",
        "Tool-calling turns with a thought, and turns where every thought begins with \"Check:\" as the verification prompt "
        "asks (#21). Whether the check itself is meaningful is read by hand.",
        "| Model | Condition | Turns with a thought | Every thought starts with Check: |",
        "|---|---|---|---|",
    ]
    for (model, condition), counts in sorted(reflection_counts(data.episodes).items()):
        with_thought = share(counts["with_thought"], counts["turns"])
        check_first = share(counts["check_first"], counts["turns"])
        lines.append(f"| {model} | {condition} | {with_thought} | {check_first} |")
    return "\n".join(lines)


def report(data: RunData) -> str:
    sections = []
    warnings = ground_truth_warnings(data)
    if warnings:
        sections.append("## Warnings\n" + "\n".join(f"- {warning}" for warning in warnings))
    sections += [
        episode_table(data),
        injection_table(data),
        flag_table(data),
        verifier_table(data),
        reflection_table(data),
        project_costs(data),
    ]
    return "\n\n".join(sections)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a pilot run as markdown.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs", help="folder that holds runs/<run-id>/")
    args = parser.parse_args(argv)

    try:
        data = load_run(Path(args.runs_dir) / args.run_id)
    except FileNotFoundError as error:
        print(error)
        return 2
    print(report(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
