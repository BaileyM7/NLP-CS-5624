# Harness

The harness checks every tool response before the model sees it, and tells the model what it found in one extra line. It is used by two conditions (see the [experiment spec](experiment-spec.md)):

| Condition | `_harness` lines | Verification prompt |
|---|---|---|
| `harness` | yes | yes |
| `flags_only` | yes | no |

The verification prompt is `VERIFY_INSTRUCTION` (`agent/prompts.py`), the same text the `verify_prompt` condition uses. It mentions `_harness` checks in a way that also reads naturally when there are none.

## Where it runs
The harness is a layer inside the agent runner (`src/toolharness/harness/`). It sits between the MCP tools and the model, rather than being a separate MCP server.
- **How it's switched on.** `run_batch` wraps each episode's tool session in a `HarnessSession` when the condition is `harness` or `flags_only`. There is no extra process or URL, so a harness run can't accidentally run without it.
- **What it sees.** Only the tool responses: the current one, plus the episode's earlier ones. It never sees episode or scenario IDs (the scenario ID names the fault) or the ground-truth log.
- **Passive.** It never makes tool calls of its own. With transient faults, a re-fetch would hand it the clean answer for free.

## The `_harness` line
Every tool response from the server gets one line appended, clean responses included, so the line's presence alone never signals a problem:
```
<the tool's response, unchanged>

_harness: {"schema_valid": true, "fresh": false, "consistent": true, "flags": ["stale: last updated 2021-03-04, more than 1 year before today (2026-11-01)"]}
```
- **Contents.** The line has one key per check, in order, then `flags`, every check's short reasons in the same order. A check's value is `true` (passed), `false` (failed) or `null` (doesn't apply to this response, e.g. no date on a search result). With no checks, the line is `_harness: {"flags": []}`.
- **What stays the same.** `is_error` is passed through unchanged.
- **Responses the runner makes itself** (`not_run` after the retry limit, `invalid_arguments` for bad JSON) never reached a tool, so they get no line.
- **In the trace.**
  - `raw_tool_response` is exactly what the model saw, line included.
  - `harness_flags` is the flag list.
  - `proxy_latency_ms` is the time spent running the checks.
- **Frozen settings.** The run manifest's `harness` field records the check list, the freshness window, the consistency threshold and the line prefix, so a resumed run must use the same harness.

## Checks
A check implements `key`, plus `run(tool, result, history) -> CheckResult(value, flags)` (`harness/base.py`). To add one, append it to `HARNESS_CHECKS` in `harness/checks.py`; its key then joins the line and the manifest.

### Format (`schema_valid`, #18)
Does the response match the tool's own output model (`app/backend.py`)? A `search` response should match `SearchResult`. A `get_article` response should match `NotFound` when it says `"error": "not_found"`, and `ArticlePage` otherwise. Tools other than these two get `null`.

| Response | Flag |
|---|---|
| An error result | `format: the tool returned an error` |
| Not JSON (an HTML page, cut-off text) | `format: not valid JSON` |
| JSON that isn't an object | `format: expected a JSON object` |
| A required field is missing | `format: missing field 'sentences'` (nested fields as `hits.0.snippet`) |
| A field has the wrong type | `format: field 'sentences' has the wrong type` |

- **Fault coverage.** It flags every hard_failure (#9) and malformed (#10) variant.
- **No false flags.** It never flags a clean response: searches, pages and not-found results.
- **Scope.** It can't catch silently_wrong, stale or contradiction pages, which are valid JSON; freshness and consistency cover those.
- **A flag isn't always a fault.** Bad tool arguments also come back as error results, so they get flagged too. A flag is a suspicion signal, not proof of a fault.

### Freshness (`fresh`, #19)
Is an article page's `last_updated` more than **365 days** before today (AS_OF, Nov 1, 2026, which the system prompt also states)?
- **Flag:** `stale: last updated 2021-03-04, more than 1 year before today (2026-11-01)`.
- **When it applies:** only to a valid `get_article` page. Searches, not-found results, error results and broken pages get `null`.
- **Fault coverage.** It flags every stale fault (#12), because those are always at least 3 years older than the page's real date.
- **HotpotQA clean pages are never flagged.** They're dated within 90 days of AS_OF.
- **LongSeal clean pages often are.** They keep their real dates, and 41% of them (27% of LongSeal answer pages) are more than a year old. That's a realistic false-alarm test, and the pilot report measures it.
- **Why 1 year.** It's a common rule of thumb, chosen without reference to the stale fault's 3–6 years. With a 2-year window, LongSeal false flags would drop to 23% (4% of answer pages).
- **Frozen.** The window is recorded in the manifest as `freshness_days`.

### Consistency (`consistent`, #20)
Does a page rewrite a sentence the agent was already shown in this episode, in an earlier search snippet or an earlier page (the same article or another)?
- **Flag:** `consistency: says "Dottie West" where an earlier result said "Connie Smith"`, quoting the first words that differ. At most 2 flags per response.
- **What counts as a rewrite.** Two different sentences of at least 6 words whose word sets overlap by **0.7** or more (Jaccard, lowercase words). These are all fine:
  - identical sentences
  - one sentence that starts with the other
  - the same words with different punctuation or spacing

  A search snippet is a sentence cut at 200 characters, so it's compared only with the start of the page sentence.
- **When it applies:** only to a valid `get_article` page. Searches, not-found results, errors and broken pages get `null`. Earlier responses that can't be read are ignored.
- **Measured on the `pilot-1` traces (480 episodes, questions 1–5 of each file):**

  | | HotpotQA | LongSeal |
  |---|---|---|
  | contradiction caught | 24/24 | 1/1 |
  | silently_wrong caught | 18/30 | 0/17 |
  | stale caught | 22/31 | 1/18 |
  | clean pages flagged (before any corrupted one) | 0/84 | 43/280 (15%) |
  | clean pages flagged after the corrupted one | 2/146 | 1/164 |

  The 2 HotpotQA flags after the corrupted page are clean re-fetches that really do disagree with the corrupted page the agent saw first, so they aren't false flags. The report counts them in their own column.
- **HotpotQA false flags beyond the pilot questions.** Questions 1–5 happen to produce none, but they do happen.
  - **How it was measured (offline, all 50 HotpotQA questions):** an agent that searches the question and opens every hit and every answer page.
  - **Result:** the check flags **15 of 272 clean pages (6%)**, in 6 of the 50 questions; 2 of those pages are answer pages.
  - **Why:** related HotpotQA articles share sentence templates, e.g. "Y6 was a series of diesel railcars operated by…" next to "Y3 was…". The full runs (N=25 or 50) will see this, so don't report the pilot's 0/84 as HotpotQA's false-flag rate.
- **Why 0.7.** At 0.6, LongSeal's false flags nearly double (82/280) and HotpotQA catches nothing more. At 0.8, HotpotQA is the same and LongSeal is almost the same: 0/36 caught instead of 2/36, and 41/280 flagged instead of 43/280. The check is the same on both datasets.
- **LongSeal.** Different sources often repeat a sentence with different numbers or names, so 15% of clean LongSeal pages seen before any corruption are flagged (10%, 44/444, counting clean pages after it too). LongSeal pages rarely repeat a sentence the agent saw before, so the check catches little there (2/36). Both rates are reported.
- **Limits.** Consistency with earlier responses doesn't prove a fact is true: the check catches a rewrite of something the agent already saw, not a wrong fact it sees for the first time.
- **Frozen.** The threshold is recorded in the manifest as `consistency_overlap`.

## LLM verifier baseline (#29)
`llm_verifier` uses a separate `VerifierSession` inside the runner rather than the heuristic harness. The agent gets the same base prompt as `baseline`: no verification instruction and no `_harness` line. The verifier is always GPT-6.1 Sol (`VERIFIER_MODEL`), distinct from the Opus 5.5 judge and from every agent model. It has no tools and sees only the question, earlier raw tool responses (with tool names and error status), and the new raw response. It sees neither scenario IDs nor injection ground truth. Earlier verifier assessments are not fed back to it.

After each server response the verifier appends one line:
```
_verifier: {"suspicious":false,"fault_guess":null,"reason":"No evidence of a problem."}
```
The response text and `is_error` are otherwise unchanged. Each assessment is a suspicion, not proof of correctness or corruption. Runner-generated `invalid_arguments` and `not_run` responses do not reach the server and are not assessed. The verifier has no opportunity to make tool calls; only agent calls reach the MCP server.

The verifier requests strict JSON-schema output and validates all three fields, their types and a nonempty reason. An incomplete or invalid reply, a provider failure, or missing token usage/billed cost fails the episode and is recorded in `errors.jsonl`; it is retried as a fresh attempt on resume rather than silently presenting an unverified response. The run manifest freezes the verifier model and OpenRouter ID, prompt, schema, output-token limit, reasoning setting and line prefix.

Traces store the structured `verifier_output`, verifier tokens, cost and successful-call latency per tool call, plus separate verifier token and cost totals per episode. Agent token limits exclude verifier usage. `llm_usage.jsonl` tags calls `agent` or `verifier`; it includes successful calls from failed attempts as well. Episode latency includes the verifier, while verifier call latency excludes provider retries.
