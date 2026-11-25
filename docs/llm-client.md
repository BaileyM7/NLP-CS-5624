# LLM client

Every model call (agent #14, verifier #29, judge #23, fact edits) goes through `OpenRouterClient` in `src/toolharness/llm/client.py`. Set `OPENROUTER_API_KEY` in `.env` first (see `.env.example`).

## Models
Defined in `src/toolharness/llm/models.py`. Use the short name everywhere (run IDs, file names, CLI flags).

| Short name | OpenRouter ID | Main run | $ / 1M tokens in / out |
|---|---|---|---|
| `opus-5.5` | `anthropic/claude-opus-5.5` | yes, HotpotQA only (and the judge) | 4 / 20 |
| `haiku-5.5` | `anthropic/claude-haiku-5.5` | yes | 0.10 / 0.50 |
| `gpt-6-astra` | `openai/gpt-6-astra` | no (dropped Nov 24 to cut cost) | 10 / 50 |
| `gpt-6-luna` | `openai/gpt-6-luna` | yes | 0.10 / 0.50 |
| `sonnet-5.5` | `anthropic/claude-sonnet-5.5` | no (wrote the fact edits) | 2 / 10 |
| `gpt-6.1-sol` | `openai/gpt-6.1-sol` | no (the LLM verifier) | 2 / 10 |

OpenRouter lists tool calling, `tool_choice` and reasoning for all six (checked Nov 19, 2026).

Roles fixed by [the experiment spec](experiment-spec.md), as constants in `models.py`: `VERIFIER_MODEL = "gpt-6.1-sol"`, `JUDGE_MODEL = "opus-5.5"`, `FACT_EDIT_MODEL = "sonnet-5.5"`. `MAIN_RUN_DATASETS` records which datasets each main-run model runs (Opus: HotpotQA only). The verifier is not a tested model. The judge (Opus) also judges its own HotpotQA episodes, so the report should mention possible self-preference bias.

## Calling a model with tools
```python
from toolharness.llm.client import OpenRouterClient, tools_from_mcp

tools = tools_from_mcp((await mcp_client.list_tools()).tools)  # our search / get_article
async with OpenRouterClient(usage_log="runs/<run-id>/llm_usage.jsonl") as llm:
    result = await llm.chat("haiku-5.5", messages, tools=tools, tag="agent")

    # Continue the conversation: the assistant message as returned, then one tool message per call
    messages.append(result.message)
    for call in result.tool_calls:  # call.id, call.name, call.arguments (raw JSON string)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_output_text})
```
- **Append `result.message` unchanged.** It carries the model's reasoning details, which OpenRouter needs back on the next turn.
- **`call.arguments` is the raw string the model wrote.** Parsing it, and handling bad JSON, is the caller's job.
- **Optional `chat()` arguments:** `tool_choice`, `max_tokens`, `response_format` (for structured output), and `reasoning` (OpenRouter's settings, e.g. `{"effort": "low"}`; `None` keeps each model's default).

## What every call records
`result.usage`:
- `input_tokens`
- `output_tokens` (includes reasoning; this is what the spec's 8,000-token episode budget counts)
- `reasoning_tokens`
- `cached_tokens`
- `cost_usd` (the cost OpenRouter billed; `None` if it didn't report one)

Plus `result.latency_ms` (the successful attempt only) and `result.attempts`. With `usage_log=...`, every call also appends one JSON row with these fields and the `tag`, which is handy for #32 and for tracking spend.

For the trace (`eval/trace.py`), the usage maps one to one: `TokenUsage(input_tokens=u.input_tokens, output_tokens=u.output_tokens, total_tokens=u.input_tokens + u.output_tokens)` and `cost_usd=u.cost_usd`.

## Retries
- **Retried with exponential backoff** (up to 5 attempts, honoring `Retry-After`): 408/409/429/5xx, connection errors and timeouts, OpenRouter's "200 with no choices" error replies, and replies that end with `finish_reason="error"` (the provider failed mid-reply). These are transport errors and never count as tool failures.
- **Raised as `LLMError`:** any other error, such as a 400 for a too-long context, or giving up after the last attempt.

## Things to know
- **None of the six models accepts `temperature`.** OpenRouter drops it, so there is no temperature-0 setting for the judge (#23). The spec's "freeze decoding settings" can only cover `reasoning` and `max_tokens`, so plan for run-to-run variance.
- **Reasoning tokens are billed as output and count toward the 8,000-token budget.** Freeze the `reasoning` setting per model in the run manifest, so every condition uses the same one.
- **Use `tool_choice="auto"` (the default) for the agent.** Claude Opus 5.5 and Sonnet 5.5 reject forced tool use with HTTP 400 (Anthropic: `tool_choice: type "tool" and "any" are not supported for this model`). So for them, neither `"required"` nor naming a specific tool works. Haiku 5.5 and the GPT models accept `"required"`.
- **Building an `MCPServer` switches on INFO logging for the whole process,** so every HTTP request gets logged. Quiet it with `logging.getLogger("httpx2").setLevel(logging.WARNING)`.

## Smoke test
```bash
python scripts/smoke_models.py                                  # the main-run models (Opus, Haiku, Luna)
python scripts/smoke_models.py --all --usage-log runs/smoke.jsonl  # all six, with a usage log
python scripts/smoke_models.py --models opus-5.5 --tool-choice required  # chosen models, forced tool use
```
Each model gets a sample question and our real tools. It must make a tool call, gets the mock app's result back, and must reply again. The script prints tokens and cost, and exits with code 1 if a main-run model fails.
