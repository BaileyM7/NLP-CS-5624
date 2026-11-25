# Pilot (#15)

The pilot proves the pipeline works end to end, and sizes the full runs before the budget is spent.

- **`pilot-1` (Nov 22).** Haiku and Luna, `baseline`, `verify_prompt`, `harness` and `flags_only`, before the consistency check (#20) and the `Check:` prompt. It measured injection rates and the format and freshness checks.
- **`pilot-2`.** It covers all five conditions with the final harness (format, freshness and consistency) and the final verification prompt. It also adds the Opus/Astra cost slice, and its report gives the cost projection for choosing N.

Every stage makes paid OpenRouter calls. Run them in order, and look at the results before starting the next stage.

## Before you start
- `OPENROUTER_API_KEY` is set in `.env`.
- These files exist locally:
  - `data/scenarios/hotpotqa.jsonl` and `data/scenarios/longseal.jsonl`
  - `data/edits/hotpotqa.jsonl` and `data/edits/longseal.jsonl`
- Check a stage without spending anything: `python scripts/run_pilot.py cheap --dry-run`.

## Smoke stage (optional, about $0.10)
Haiku 5.5 and GPT-6 Luna each run 1 question per scenario file across the four non-verifier conditions.
- **Use a fresh run ID.** `pilot-smoke` and `pilot-smoke-2` used older prompts, so their manifests refuse the new settings.
```bash
# terminal 1: the tool server, with this stage's ground-truth log
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl data/scenarios/longseal.jsonl \
    --edits data/edits/hotpotqa.jsonl data/edits/longseal.jsonl --gt-log runs/pilot-smoke-3/ground_truth.jsonl

# terminal 2
python scripts/run_pilot.py smoke --run-id pilot-smoke-3
python -m toolharness.eval.pilot_report --run-id pilot-smoke-3
```
Read a few traces in `runs/pilot-smoke-3/traces/` before going on. Check that:
- every tool response in `harness` and `flags_only` ends with a `_harness:` line holding `schema_valid`, `fresh` and `consistent`, and that faulted responses carry flags
- each thought in `verify_prompt` and `harness` starts with `Check:` (#21)
- faults fire, and `failed attempts` is 0

A bug found here is fixed and folded into its issue's commit before you go on. A fix that changes prompt or harness text means the stage has to be rerun under another new run ID.

## pilot-2: cheap, then verifier, then big (about $11–21 in total)
Stop any earlier server first (e.g. the smoke stage's). Its ground-truth log belongs to the smoke run, and `pilot-2`'s ground truth must go to `runs/pilot-2/ground_truth.jsonl`. One new server then serves all three stages:
```bash
# terminal 1 (stop the old server with Ctrl+C first)
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl data/scenarios/longseal.jsonl \
    --edits data/edits/hotpotqa.jsonl data/edits/longseal.jsonl --gt-log runs/pilot-2/ground_truth.jsonl

# terminal 2
python scripts/run_pilot.py cheap
python scripts/run_pilot.py verifier
python scripts/run_pilot.py big
```
- **Cheap (about $0.70):** Haiku and Luna, 5 questions per file, `baseline`, `verify_prompt`, `harness` and `flags_only`. That's 440 episodes.
- **Verifier (about $5.50):** Haiku, the same 5 questions per file as `cheap`, `llm_verifier`. That's 55 episodes.
  - **Why the same questions:** question sizes vary a lot. LongSeal question 3 alone is 3–5x larger than the others, so measuring fewer questions would understate the projection.
  - **Luna:** its `llm_verifier` cost is projected from Haiku's. The verifier is always GPT-6 Astra, so its measured cost carries over.
  - **What costs the money:** GPT-6 Astra checking every tool response. It costs about $0.03 per HotpotQA episode and $0.17 per LongSeal episode, because LongSeal pages are long.
  - **What to inspect afterwards:** the traces and `llm_usage.jsonl`, for `_verifier` assessments, separately recorded verifier cost, tokens and latency, and errors.
- **Big (about $5–15):** Opus 5.5 and GPT-6 Astra, 2 HotpotQA questions, `baseline` and `harness`. That's 48 episodes, and it gives real token counts for the two models that dominate the budget.
- **LongSeal leaves out contradiction.** Every stage that uses LongSeal passes `--skip longseal:contradiction`. In `pilot-1` it fired in only 1 of 40 LongSeal episodes, so those episodes cost money and test nothing. Contradiction still runs on HotpotQA, where it fires 55–65%.
- **Concurrency.**
  - **Who runs one at a time:** GPT models, and every `llm_verifier` episode (its verifier is a GPT model), automatically.
  - **Why:** OpenRouter's per-minute limit counts each call's requested `max_tokens` (up to 7,000), so parallel GPT episodes hit `HTTP 429`.
  - **Everything else** runs 4 at a time. `--concurrency N` overrides this for every command in a stage, and concurrency isn't an experiment setting.
- **If a stage stops early:** a stage stops at the first command that fails. Rerunning the same stage resumes it; finished episodes are skipped and failed ones get a new attempt. For a rate limit, wait a minute first.

## Report
```bash
python -m toolharness.eval.pilot_report --run-id pilot-2 > runs/pilot-2/report.md
```
Post the report on #15, along with the N the team picks. It contains:
- **Warnings:** shown first when the ground-truth log is missing, or when episodes made tool calls but have no rows in it, e.g. because the server was started with another `--gt-log`. Without that log every fault looks uninjected, so fix this before reading anything else.
- **Episodes:** per model, condition and dataset: episodes, failed attempts, total cost (agent plus verifier) and output tokens per episode, turns, HotpotQA exact match, and stop reasons.
- **Fault injection:** how often each fault actually fired, per dataset.
- **Harness flags, per fault:**
  - **The corrupted response.** Every hard_failure, malformed and stale response should be flagged. The consistency check also catches contradiction and part of silently_wrong.
  - **Clean responses before any corrupted one.** Flags here are false flags; LongSeal's old dates and its sources' natural disagreements show up here.
  - **Clean responses after the corrupted one.** They can genuinely conflict with it, e.g. a clean re-fetch.
  - **Tool errors that weren't corrupted,** such as bad arguments.
- **LLM verifier verdicts:** the same four kinds, for the verifier's `suspicious` verdicts.
- **Reflection:** tool-calling turns that have a thought, and turns where every thought starts with `Check:`, in `verify_prompt` and `harness`.
- **Projected full-run cost:**
  - **Grid:** the 4 main models, with 6 fault types per HotpotQA question and 5 per LongSeal question; HotpotQA at N = 25, 50 and 100, LongSeal at N = 15 and 30.
  - **Estimates:** costs that weren't measured are scaled from the cheap models, and every scaling is listed.
  - **llm_verifier for Opus and Astra:**
    - **Agent part:** scaled like any condition.
    - **Verifier part:** the cheap models' measured Astra cost per episode, added unscaled, because the verifier is the same model for every agent.
  - **Optional:** `flags_only` is a separate line.
  - **Not included:** the answer judge (#23), which likely adds about half again on top of the agent cost, and retried failed attempts. Budget for both when choosing N.

## Hand labels
About 30 episodes are labeled **by people**. These labels are how #23 checks the judge, and the judge is Claude Opus 5.5, so no model may fill them in.
```bash
python -m toolharness.eval.labeling pick --run-id pilot-2
```
This writes `runs/pilot-2/labeling/`:
- **`01.md` … `30.md`:** one per episode, showing everything the agent saw, plus a final section on what was corrupted.
- **`labels.jsonl`:** one empty row per item.
- **`key.json`:** which episode each item is.

The pick is spread over fault types, conditions, both models and both datasets, with at least 6 LongSeal episodes and 5 clean (`none`) ones. Faulted episodes are picked only where the fault actually fired.

The model and condition are hidden, as they are from the judge. Don't open `key.json` until labeling is done. `_harness` and `_verifier` lines can still give the condition away; that's expected.

### Labeling rules (from the [experiment spec](experiment-spec.md))
| Field | Fill in |
|---|---|
| `labeler` | your name |
| `detected` | `true` if the agent's own text shows it recognized the corrupted response may be unreliable: it identifies an error, inconsistency, stale information or unusable content, or otherwise questions it. A routine retry, a strategy switch, or a refusal with no stated link to reliability is **not** detection, and neither is a correct answer by coincidence. A `_harness` flag or `_verifier` verdict counts only if the agent takes it up as its own suspicion. `null` only when nothing was corrupted. |
| `detection_turn` | the turn number (the `## Turn k` heading) of the first turn that meets the detection rule, or `null` |
| `false_alarm_turns` | turns where the agent questions or treats as unreliable a response that was **not** corrupted. This includes any such turn in a clean episode, and turns that act on a flag or verdict on a clean response. |
| `relied_on_corrupted` | `true` if the final answer or the reasoning behind it relies on the corrupted claim. `null` only when nothing was corrupted. |
| `answer_correct` | LongSeal only: is the final answer correct against the gold answer? Required for LongSeal; HotpotQA uses exact match. |
| `notes` | anything unclear. **Never quote LongSeal text here**; the labels are committed to a public repo. |

### Export and commit
```bash
python -m toolharness.eval.labeling export --run-id pilot-2 --out data/labels
```
- **Checks first.** Every row needs a labeler, `detected` and `relied_on_corrupted` on every injected episode, and `answer_correct` on every LongSeal episode.
- **`pick` needs the ground-truth log.** It refuses to run without `runs/pilot-2/ground_truth.jsonl`, because without it every fault looks uninjected.
- **Writes:**
  - `data/labels/pilot-2.jsonl` (labels by episode ID)
  - `pilot-2_hotpotqa_traces.jsonl`
  - `pilot-2_hotpotqa_ground_truth.jsonl`
- **Commit:** these three files are folded into the pilot-tooling commit.
- **LongSeal stays private.** LongSeal traces are never exported. Share `runs/pilot-2/` privately with whoever builds #23.
