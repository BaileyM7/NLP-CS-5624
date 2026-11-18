# Tool server

One long-running MCP server serves every episode's tools over streamable HTTP. The agent runner (#14) and the proxy (#17) connect to it.

## Start it
```bash
python -m toolharness.data.hotpotqa --n 50 --seed 0 --out data/scenarios/hotpotqa.jsonl   # once
python -m toolharness.mcp_server --scenarios data/scenarios/hotpotqa.jsonl --port 8765
```
- MCP endpoint: `http://127.0.0.1:8765/mcp`
- Health check: `GET http://127.0.0.1:8765/health` → `{"status": "ok", "scenarios": N, "episodes": N}`
- Pass several files to `--scenarios` to serve HotpotQA and LongSeal together.
- Episodes live in memory, so restart the server between experiment runs.
- The server only accepts `localhost` / `127.0.0.1` connections (the MCP SDK's default protection).

## Identify the episode on every call
Send these HTTP headers with every request:

| Header | Required | Meaning |
|---|---|---|
| `X-Episode-Id` | yes | Unique ID for one run of one model on one scenario in one condition. Never reuse it. |
| `X-Scenario-Id` | yes | A `scenario_id` from the scenario file, e.g. `hotpotqa-5ab797af5542995dae37e96c-stale` |
| `X-Fault-Seed` | no (default `0`) | Integer seed for the fault injector |

- The first call with a new episode ID creates the episode; later calls with that ID reuse it.
- An episode can't change its scenario or seed.
- A missing or unknown ID fails the request with an `MCPError` (JSON-RPC error). It is never returned to the agent as a tool result.

## Tools
Both tools return **plain JSON text** in `result.content[0].text`. There's no structured output, so a malformed response reaches the agent as-is.

**`search(query: str, k: int = 5)`**: BM25 ranking over article pages, at most 10 hits.
```json
{"query": "...", "hits": [{"title": "Once a Day", "page": 1, "snippet": "It was the first ..."}]}
```

**`get_article(title: str, page: int = 1)`**: one page (≤ 3,000 characters, whole sentences). Title matching ignores case.
```json
{"title": "Once a Day", "page": 1, "total_pages": 1, "sentences": ["..."], "last_updated": "2026-09-14", "source_id": "hotpotqa:Once a Day"}
```
An unknown title or page is a normal result, not an error:
```json
{"error": "not_found", "title": "Rome", "page": null, "message": "No article titled 'Rome'."}
```

`result.is_error` is `true` only for injected hard failures (#8, #9) and unknown tool names.

## Client example (Python, `mcp` 2.3)
```python
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

headers = {"X-Episode-Id": "run1-opus-harness-0001", "X-Scenario-Id": scenario_id}
async with httpx2.AsyncClient(headers=headers, timeout=httpx2.Timeout(30, read=300)) as http:
    async with Client(streamable_http_client("http://127.0.0.1:8765/mcp", http_client=http)) as client:
        tools = (await client.list_tools()).tools
        result = await client.call_tool("search", {"query": "Seine river", "k": 5})
        text, is_error = result.content[0].text, result.is_error
```
- `httpx2` (not `httpx`) is installed with `mcp`. `streamable_http_client` has no `headers=` argument, so set the headers on the `httpx2.AsyncClient`.
- In-memory tests can't send HTTP headers. Pass the same keys as request `_meta` instead: `client.call_tool(name, args, meta={"x-episode-id": ..., "x-scenario-id": ...})`.

## Notes for the proxy (#17)
- Forward all three headers unchanged on every upstream call.
- Never read the fault injector's ground-truth log. The proxy may only see what the agent sees.
