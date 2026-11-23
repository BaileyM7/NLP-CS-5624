# Harness

The harness checks every tool response before the model sees it, and tells the model what it found in one extra line. It is used by two conditions (see the [experiment spec](experiment-spec.md)):

| Condition | `_harness` lines | Verification prompt |
|---|---|---|
| `harness` | yes | yes |
| `flags_only` | yes | no |

## Where it runs
The harness is a layer inside the agent runner (`src/toolharness/harness/`). It sits between the MCP tools and the model, rather than being a separate MCP server.
- **How it's switched on.** `run_batch` wraps each episode's tool session in a `HarnessSession` when the condition is `harness` or `flags_only`. There is no extra process or URL, so a harness run can't accidentally run without it.
- **What it sees.** Only the tool responses: the current one, plus the episode's earlier ones. It never sees episode or scenario IDs (the scenario ID names the fault) or the ground-truth log.
- **Passive.** It never makes tool calls of its own. With transient faults, a re-fetch would hand it the clean answer for free.

## The `_harness` line
Every tool response from the server gets one line appended, clean responses included, so the line's presence alone never signals a problem:
```
<the tool's response, unchanged>

_harness: {"schema_valid": true, "fresh": false, "flags": ["stale: last updated 2021-03-04, more than 1 year before today (2026-11-01)"]}
```
- **Contents.** The line has one key per check, in order, then `flags`, every check's short reasons in the same order. A check's value is `true` (passed), `false` (failed) or `null` (doesn't apply to this response, e.g. no date on a search result). With no checks, the line is `_harness: {"flags": []}`.
- **What stays the same.** `is_error` is passed through unchanged.
- **Responses the runner makes itself** (`not_run` after the retry limit, `invalid_arguments` for bad JSON) never reached a tool, so they get no line.
- **In the trace.**
  - `raw_tool_response` is exactly what the model saw, line included.
  - `harness_flags` is the flag list.
  - `proxy_latency_ms` is the time spent running the checks.
- **Frozen settings.** The run manifest's `harness` field records the check list and the line prefix, so a resumed run must use the same harness.

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
- **Scope.** It can't catch silently_wrong, stale or contradiction pages, which are valid JSON.
- **A flag isn't always a fault.** Bad tool arguments also come back as error results, so they get flagged too. A flag is a suspicion signal, not proof of a fault.

The freshness check (#19) follows.
