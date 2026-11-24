# Pilot (#15)

The pilot proves the pipeline works end to end, and sizes the full runs before the budget is spent. The scripted stages cover `baseline`, `verify_prompt`, `harness` and `flags_only`. Run a separate short `llm_verifier` top-up (#29) and a consistency-check top-up once #20 exists.

Every stage makes paid OpenRouter calls. Run them in order, and look at the results before starting the next stage.

## Before you start
- `OPENROUTER_API_KEY` is set in `.env`.
- These files exist locally:
  - `data/scenarios/hotpotqa.jsonl` and `data/scenarios/longseal.jsonl`
  - `data/edits/hotpotqa.jsonl` and `data/edits/longseal.jsonl`
- Check the grid without spending anything: `python scripts/run_pilot.py cheap --dry-run`.

## 1. Smoke stage (about $0.10)
Haiku 5.5 and GPT-6 Luna each run 1 question per scenario file × 6 faults × 4 conditions, which is 96 episodes. It has its own run ID, so it never mixes with the real pilot.
```bash
# terminal 1: the tool server, with this stage's ground-truth log
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl data/scenarios/longseal.jsonl \
    --edits data/edits/hotpotqa.jsonl data/edits/longseal.jsonl --gt-log runs/pilot-smoke/ground_truth.jsonl

# terminal 2
python scripts/run_pilot.py smoke
python -m toolharness.eval.pilot_report --run-id pilot-smoke
```
Read a few traces in `runs/pilot-smoke/traces/` before going on. Check that:
- every tool response in `harness` and `flags_only` ends with a `_harness:` line, and that faulted responses carry flags
- each thought in `verify_prompt` and `harness` starts with a check of the last result (#21)
- faults fire, and `failed attempts` is 0

A bug found here is fixed and folded into its issue's commit before you go on. A fix that changes prompt or harness text means the smoke stage has to be rerun under a new run ID: `python scripts/run_pilot.py smoke --run-id pilot-smoke-2` (with a server logging to `runs/pilot-smoke-2/ground_truth.jsonl`).

## 2. Cheap stage (about $0.60), then 3. Big stage (about $5–15)
Restart the server with the real run's log, then run the two stages:
```bash
# terminal 1
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl data/scenarios/longseal.jsonl \
    --edits data/edits/hotpotqa.jsonl data/edits/longseal.jsonl --gt-log runs/pilot-1/ground_truth.jsonl

# terminal 2
python scripts/run_pilot.py cheap
python scripts/run_pilot.py big
```
- **Cheap:** Haiku and Luna each run 5 questions per file × 6 faults × 4 conditions, which is 480 episodes.
- **Big:** Opus 5.5 and GPT-6 Astra each run 2 HotpotQA questions × 6 faults × `baseline` and `harness`, which is 48 episodes. It gives real token counts for the two models that dominate the budget.
- **If a stage stops early:** a stage stops at the first command that fails. Rerunning the same stage resumes it; finished episodes are skipped.
- **Rate limits (`HTTP 429` in `errors.jsonl`):** GPT-6 Luna hit OpenRouter's per-minute limit in the smoke stage when 4 episodes ran at once. Every call asks for up to 7,000 output tokens, and OpenAI-style limiters count that against the minute. Wait a minute, then rerun the stage with `--concurrency 1`. Concurrency isn't an experiment setting, so the resumed run stays valid.

## 4. Report
```bash
python -m toolharness.eval.pilot_report --run-id pilot-1 > runs/pilot-1/report.md
```
Post the report on #15, along with the N the team picks. It contains:
- **Episodes:** per model, condition and dataset: episodes, failed attempts, cost and output tokens per episode, turns, HotpotQA exact match, and stop reasons.
- **Fault injection:** how often each fault actually fired, per dataset. Contradiction fires less often, and almost never on LongSeal.
- **Warnings:** shown first when the ground-truth log is missing, or when episodes made tool calls but have no rows in it, e.g. because the server was started with another `--gt-log`. Without that log every fault looks uninjected, so fix this before reading anything else.
- **Harness flags, per fault:**
  - **The corrupted response.** It should be flagged every time for hard_failure, malformed and stale; silently_wrong and contradiction can't be caught by these checks.
  - **Clean responses.** Flags here are false flags; LongSeal's old dates show up here.
  - **Tool errors that weren't corrupted,** such as bad arguments. Flagging them is correct, but they aren't faults.
- **Reflection:** tool-calling turns that have a thought, in `verify_prompt` and `harness`.
- **Projected full-run cost:**
  - **Grid:** the 4 main models × 6 faults, for HotpotQA at N = 25, 50 and 100, and LongSeal at N = 15 and 30.
  - **Estimates:** costs that weren't measured are scaled from the cheap models, and every scaling is listed.
  - **Verifier:** measured agent plus verifier cost is used for model/dataset cells with top-up episodes; other cells are estimated from baseline.
  - **Optional:** `flags_only` is a separate line.
  - **Not included:** the answer judge (#23), which likely adds about half again on top of the agent cost, and retried failed attempts. Budget for both when choosing N.

## 5. Hand labels
About 30 episodes are labeled **by people**. These labels are how #23 checks the judge, and the judge is Claude Opus 5.5, so no model may fill them in.
```bash
python -m toolharness.eval.labeling pick --run-id pilot-1
```
This writes `runs/pilot-1/labeling/`:
- **`01.md` … `30.md`:** one per episode, showing everything the agent saw, plus a final section on what was corrupted.
- **`labels.jsonl`:** one empty row per item.
- **`key.json`:** which episode each item is.

The pick is spread over fault types, conditions, both models and both datasets, with at least 6 LongSeal episodes and 5 clean (`none`) ones. Faulted episodes are picked only where the fault actually fired.

The model and condition are hidden, as they are from the judge. Don't open `key.json` until labeling is done. `_harness` lines can still give the condition away; that's expected.

### Labeling rules (from the [experiment spec](experiment-spec.md))
| Field | Fill in |
|---|---|
| `labeler` | your name |
| `detected` | `true` if the agent's own text shows it recognized the corrupted response may be unreliable: it identifies an error, inconsistency, stale information or unusable content, or otherwise questions it. A routine retry, a strategy switch, or a refusal with no stated link to reliability is **not** detection, and neither is a correct answer by coincidence. A `_harness` flag counts only if the agent takes it up as its own suspicion. `null` only when nothing was corrupted. |
| `detection_turn` | the turn number (the `## Turn k` heading) of the first turn that meets the detection rule, or `null` |
| `false_alarm_turns` | turns where the agent questions or treats as unreliable a response that was **not** corrupted. This includes any such turn in a clean episode, and turns that act on a flag on a clean response. |
| `relied_on_corrupted` | `true` if the final answer or the reasoning behind it relies on the corrupted claim. `null` only when nothing was corrupted. |
| `answer_correct` | LongSeal only: is the final answer correct against the gold answer? Required for LongSeal; HotpotQA uses exact match. |
| `notes` | anything unclear. **Never quote LongSeal text here**; the labels are committed to a public repo. |

### Export and commit
```bash
python -m toolharness.eval.labeling export --run-id pilot-1 --out data/labels
```
- **Checks first.** Every row needs a labeler, `detected` and `relied_on_corrupted` on every injected episode, and `answer_correct` on every LongSeal episode.
- **`pick` needs the ground-truth log.** It refuses to run without `runs/pilot-1/ground_truth.jsonl`, because without it every fault looks uninjected.
- **Writes:**
  - `data/labels/pilot-1.jsonl` (labels by episode ID)
  - `pilot-1_hotpotqa_traces.jsonl`
  - `pilot-1_hotpotqa_ground_truth.jsonl`
- **Commit:** these three files go into the #15 commit.
- **LongSeal stays private.** LongSeal traces are never exported. Share `runs/pilot-1/` privately with whoever builds #23.

## Verifier top-up
Use the same scenario and edit files as the main pilot, and a fresh run ID and ground-truth log. For example, start the server as above with `--gt-log runs/pilot-verifier/ground_truth.jsonl`, then run:
```bash
python -m toolharness.agent --run-id pilot-verifier --model haiku-5.5 --condition llm_verifier \
    --scenarios data/scenarios/hotpotqa.jsonl data/scenarios/longseal.jsonl --questions 1
python -m toolharness.eval.pilot_report --run-id pilot-verifier
```
Inspect the traces and usage log for `_verifier` assessments, separately recorded verifier cost/tokens/latency, and errors before expanding the top-up. This makes paid agent and verifier calls. To include measured verifier costs in a broader pilot projection, run the top-up using the broader pilot's run ID and server ground-truth log, with the same scenario files and frozen settings.
