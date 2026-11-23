"""Shared test helpers: a fake OpenRouter, chat-completion builders, the sample file, a free-port finder,
scenario fault variants, and fake episodes and run folders."""
import itertools
import json
import socket
from pathlib import Path

import httpx2

from toolharness.data.scenario import FaultType, Scenario, fault_variant_id
from toolharness.eval.trace import Episode, EpisodeTotals, ToolCallTrace, Turn, append_jsonl, trace_filename
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


def fault_variant(scenario: Scenario, fault: FaultType) -> Scenario:
    """The same question as `scenario`, under another fault type."""
    return scenario.model_copy(update={"scenario_id": fault_variant_id(scenario.scenario_id, fault), "fault_type": fault})


def make_episode(model, condition, dataset, fault, *, cost=0.0, scenario="q1", question="Q?", final="x", gold="x", calls=(), run="r"):
    """A finished episode. `calls` are (tool, raw response, harness flags) made in turn 1; the last turn answers."""
    scenario_id = f"{dataset}-{scenario}-{fault}"
    turns = []
    if calls:
        tool_calls = []
        for tool, raw, flags in calls:
            tool_calls.append(ToolCallTrace(tool_name=tool, tool_args={"query": "q"}, raw_tool_response=raw, harness_flags=flags))
        turns.append(Turn(index=1, assistant_text="Thought: look it up.", tool_calls=tool_calls))
    turns.append(Turn(index=len(turns) + 1, assistant_text=f"Final Answer: {final}"))
    return Episode(
        episode_id=f"{run}__{model}__{condition}__{scenario_id}__s0__a1",
        scenario_id=scenario_id,
        dataset=dataset,
        model=model,
        condition=condition,
        fault_type=FaultType(fault),
        question=question,
        final_answer=final,
        gold_answer=gold,
        stop_reason="final_answer",
        totals=EpisodeTotals(agent_cost_usd=cost),
        turns=turns,
    )


def gt_row(episode, call_index, tool, corrupted=False, original=None, corrupted_text=None):
    """One ground-truth log row, in the format the fault injector writes."""
    return {
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "fault_type": episode.fault_type.value,
        "call_index": call_index,
        "tool": tool,
        "args": {},
        "corrupted": corrupted,
        "retry_of_corrupted": False,
        "original": original,
        "corrupted_text": corrupted_text,
        "corrupted_is_error": False if corrupted else None,
    }


def write_run(run_dir, episodes, gt_rows=(), errors=()):
    """Write a run folder the way the batch runner and the tool server do; returns run_dir."""
    run_dir = Path(run_dir)
    for episode in episodes:
        append_jsonl(episode, run_dir / "traces" / trace_filename(episode.model, episode.condition))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "ground_truth.jsonl").open("w", encoding="utf-8") as stream:
        for row in gt_rows:
            stream.write(json.dumps(row) + "\n")
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as stream:
        for row in errors:
            stream.write(json.dumps(row) + "\n")
    return run_dir
