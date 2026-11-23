# Agent runner

The agent runner (`src/toolharness/agent/`) runs one model on one scenario in one condition and writes an `Episode` trace (see [trace format](trace-format.md)). The model works ReAct-style: each turn it either calls `search` / `get_article` through the [tool server](tool-server.md), or gives its final answer.

## Running a batch
Start the tool server with a fresh ground-truth log for the run, then run the agent in a second terminal:
```bash
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl --edits data/edits/hotpotqa.jsonl --gt-log runs/pilot-1/ground_truth.jsonl

python -m toolharness.agent --run-id pilot-1 --model haiku-5.5 --condition baseline \
    --scenarios data/scenarios/hotpotqa.jsonl --questions 5 --faults none hard_failure malformed silently_wrong
```
- `--questions N`: only the first N questions of each scenario file. The files are already a seeded, stratified sample. With HotpotQA and LongSeal together, each gets N.
- `--faults`: only these fault types (default: all six). silently_wrong, stale and contradiction episodes need the server started with `--edits`, as above.
- `--concurrency` (default 4) and `--seed` (default 0) are also available.
- Run the same command again with `--model gpt-6-luna` to add a second model to the same run.
- The command checks the server's `/health` first, and exits if the server isn't running.

## What a run writes (`runs/<run-id>/`)
| File | Contents |
|---|---|
| `traces/<model>__<condition>.jsonl` | finished episodes, one per line |
| `started.jsonl` | one row per episode attempt, written before it runs |
| `errors.jsonl` | failed attempts (API errors after retries, tool server down, any unexpected error), each with its error type and message; one failure never stops the rest of the batch |
| `llm_usage.jsonl` | one row per model call (tokens, cost, latency) |
| `manifest.json` | what each model/condition pair was run with |
| `ground_truth.jsonl` | written by the tool server (`--gt-log`), never by the runner |

## Budget and limits
Every episode gets 12 turns, 3 retries, and 8,000 output tokens (reasoning tokens included), the same in every condition. The runner always keeps 1,000 output tokens back for a final answer.

When a limit is reached, the next turn becomes the **final-answer turn**. Tools stay defined, but `tool_choice="none"` and the model is told: *"You have reached the limit for this question. Give your final answer now…"* Checked before every turn, first match wins:

| Condition | Result |
|---|---|
| No output tokens left | stop with no answer, `stop_reason="token_budget"` |
| The agent tried a 4th retry | final-answer turn, `stop_reason="retry_budget"` |
| 1,000 or fewer tokens left, or the last reply was cut off (`finish_reason="length"`) | final-answer turn, `stop_reason="token_budget"` |
| This is turn 12 | final-answer turn, `stop_reason="max_turns"` |

- **Answering on its own** gives `stop_reason="final_answer"`. A limit's `stop_reason` together with a `final_answer` means the answer was forced.
- **A retry** is a call with the same tool and arguments as an earlier one, with defaults filled in, so `get_article("X")` equals `get_article("X", page=1)`. The cap is shared across tools.
- **The 4th retry is not sent to the server,** and neither is any call after it in the same turn. Those calls get `{"error": "not_run", ...}` back.
- **Arguments that aren't a JSON object** get `{"error": "invalid_arguments", ...}` back, without a server call. The trace records them as `tool_args={"_unparsed": "<raw>"}`.
- **The final answer** is the first non-empty line after the last `Final Answer:` (markdown such as `**Final Answer**:` also counts, and `*` is stripped), or the whole reply if there is no marker.

## The `thought` argument
Every tool the model sees has a required `thought` argument: one or two sentences on what it concluded from the last result, and why it's making this call. It exists for two reasons:
- **The judge (#23) sees only visible text.** Many models, especially the GPT models, send tool calls with no text and keep their reasoning hidden.
- **#21 needs a visible reflection on every turn.**

The runner strips `thought` before calling the server, so the tool server and the harness checks never see it. It is saved in the turn's `assistant_text` as `Thought: ...`, after any text the model wrote.

## Conditions
| Condition | System prompt | Status |
|---|---|---|
| `baseline` | `BASE_PROMPT` | ready |
| `verify_prompt` | `BASE_PROMPT` + verification instruction | ready |
| `llm_verifier` | `BASE_PROMPT` | needs #29 |
| `harness` | `BASE_PROMPT` + verification instruction, plus `_harness` lines | ready |
| `flags_only` | `BASE_PROMPT`, plus `_harness` lines | ready |

- **`BASE_PROMPT`** is in `agent/prompts.py`. It states today's date (November 1, 2026, the mock app's "now") and never mentions faults. It tells the agent that search results are only excerpts, so it should open the articles it relies on: in the first pilot smoke stage, agents often answered from snippets alone, so faults (which fire on article pages) rarely got a chance to fire.
- **Verification instruction (#21):** `VERIFY_INSTRUCTION` in `agent/prompts.py`. It asks the model to check every tool result before relying on it, and to start each thought with that check. `verify_prompt` and `harness` use the identical text, so comparing them isolates the effect of the `_harness` flags. It's frozen experiment text, recorded in the manifest's `system_prompt`.
- **Harness conditions:** `harness` and `flags_only` turn on the harness layer automatically; no extra server or URL is needed. Their trace fields `harness_flags` and `proxy_latency_ms` are filled; see [the harness doc](harness.md).
- **#29:** `verifier_output` stays `None` until the LLM verifier exists.

## Resume and episode IDs
Episode IDs look like `<run-id>__<model>__<condition>__<scenario-id>__s<seed>__a<attempt>`.
- **The server keeps per-episode fault state** (whether the fault already fired), so an ID is never reused.
- **Rerunning the same command:**
  - It skips scenarios that already have a finished episode.
  - It reruns the rest under the next attempt number. That includes failed episodes and ones interrupted with Ctrl-C.
  - Failed attempts never appear in the trace file.

## Manifest
The first run of a model/condition pair records its settings in `manifest.json`:
- the OpenRouter model ID and the condition
- the full system prompt, the thought description and the final-turn message
- `reasoning`: `null`, meaning each model's default
- the budget and the seed
- SHA-256 hashes of the scenario files, compared by content, so the same file under another path is still a match
- the git commit and whether the tree had uncommitted changes

A later run refuses to start if any of those settings changed (except the git commit and the time), so one run never mixes experiment settings. Use a new `--run-id` instead.

## Things to know
- **`is_error=true` doesn't always mean an injected hard failure.** A wrong-typed or missing tool argument also comes back from the server as an error result.
- **Before a paid run, run `scripts/check_final_turn.py`** (a few cents). It confirms every model accepts both kinds of forced final-answer turn: after tool use at the turn limit, and with about 1,000 tokens left at the token limit.
