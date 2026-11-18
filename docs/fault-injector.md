# Fault injector

The server's `EpisodeRegistry` wraps every episode's `MockApp` in a `FaultInjector` (`src/toolharness/injector/injector.py`). The injector passes calls through and corrupts at most one response per episode.

## How a fault fires
- **When:** on the first `get_article` call that returns an **answer page**, i.e. a page holding a supporting-fact sentence (`MockApp.answer_pages()`). Title case doesn't matter. `search` is never corrupted.
- **Once:** faults are transient. Every later call, including an identical retry, gets clean data, so the agent can recover by retrying or cross-checking.
- **Maybe never:** if the agent never fetches an answer page, nothing is injected. `summarize_gt` reports `fault_injected=False`, and detection metrics skip that episode.
- **Same corruption everywhere:** the random generator is seeded by `(X-Fault-Seed, scenario_id)`, not the episode. Every model and condition sees the exact same corrupted text.

## Ground-truth log
The server's `--gt-log` file gets one JSON row per agent tool call:

| Field | Meaning |
|---|---|
| `episode_id`, `scenario_id`, `fault_type` | which episode |
| `call_index`, `tool`, `args` | which call (0-based, in order) |
| `corrupted` | `true` on the one corrupted call |
| `retry_of_corrupted` | `true` when the agent re-fetched the corrupted page |
| `original`, `corrupted_text`, `corrupted_is_error` | clean vs. returned response, **only on the corrupted row** (else `null`) |

`summarize_gt(path)` (`src/toolharness/injector/ground_truth.py`) gives per-episode `fault_injected`, `injected_at_call` and `retries_of_corrupted`. The agent and the proxy must never read this file.

## Adding a fault type (#9–#13)
1. Create `src/toolharness/injector/faults/<fault>.py` with a `Corruptor` subclass (`src/toolharness/injector/corruptors.py`) that implements:
   ```python
   def corrupt(self, call, clean, history, rng, target) -> ToolResponse:
   ```
   - `clean.text`: the clean JSON (`ArticlePage`: title, page, total_pages, sentences, last_updated, source_id)
   - `target`: an `AnswerTarget` with `title`, `page`, `sentence_positions` (indexes into `sentences` that hold supporting facts) and `gold_answer`
   - `history`: earlier `CallRecord`s in this episode (each with `.clean` and `.returned`)
   - `rng`: use it for **every** random choice; never use the global `random` module
2. Register it in `src/toolharness/injector/faults/__init__.py`: one import and one `FaultType.X: YourCorruptor()` entry. Until a fault type is registered, the server rejects its scenarios.
3. Override `should_trigger` only if the default rule doesn't fit (contradiction needs an earlier fact to contradict).
4. Add `tests/test_fault_<fault>.py`: corrupt an answer page from `data/sample/hotpotqa_sample.jsonl` and check the properties below.

| Fault | Return | Must hold |
|---|---|---|
| `hard_failure` | `ToolResponse(text=<realistic error message>, is_error=True)` | message looks like a real API failure (timeout, 500, rate limit) |
| `malformed` | `is_error=False`, broken payload | fails `ArticlePage.model_validate_json` (invalid JSON, missing field, wrong type, or truncated) |
| `silently_wrong` | valid `ArticlePage` JSON | a fact in a `target.sentence_positions` sentence changed, so `target.gold_answer` no longer follows |
| `stale` | valid `ArticlePage` JSON | `last_updated` clearly older than `AS_OF − CLEAN_DATE_WINDOW_DAYS`, plus an outdated fact |
| `contradiction` | valid `ArticlePage` JSON | conflicts with a fact the agent already received earlier in `history` |
