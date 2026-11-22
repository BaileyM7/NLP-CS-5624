"""Shared test helpers: a fake OpenRouter, chat-completion builders, the sample file and a free-port finder."""
import itertools
import json
import socket
from pathlib import Path

import httpx2

from toolharness.llm.client import OpenRouterClient

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample" / "hotpotqa_sample.jsonl"

_call_ids = itertools.count(1)


def completion(message, usage=None, finish_reason="stop"):
    """An OpenAI-style chat completion body."""
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _usage(output_tokens):
    return {"prompt_tokens": 100, "completion_tokens": output_tokens, "total_tokens": 100 + output_tokens, "cost": 0.001}


def tool_reply(*calls, content=None, output_tokens=50, finish_reason="tool_calls"):
    """A reply with one tool call per (name, args) pair; dict args are JSON-encoded, str args are sent raw."""
    tool_calls = []
    for name, args in calls:
        arguments = args if isinstance(args, str) else json.dumps(args)
        call_id = f"call_{next(_call_ids)}"
        tool_calls.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}})
    message = {"role": "assistant", "content": content, "tool_calls": tool_calls}
    return completion(message, usage=_usage(output_tokens), finish_reason=finish_reason)


def answer_reply(text, output_tokens=50, finish_reason="stop"):
    """A plain text reply with no tool calls."""
    return completion({"role": "assistant", "content": text}, usage=_usage(output_tokens), finish_reason=finish_reason)


class FakeOpenRouter:
    """Answers requests with the queued (status, body) pairs in order and records each request body."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return httpx2.Response(status, json=body)


def make_client(fake, **kwargs):
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake))
    return OpenRouterClient(api_key="test-key", http_client=http, base_delay=0, **kwargs)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
