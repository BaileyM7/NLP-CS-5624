"""The ReAct agent loop: one model answers one scenario through the MCP tools and the result is an Episode trace."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..data.scenario import Scenario
from ..eval.trace import Condition, Episode, EpisodeTotals, TokenUsage, ToolCallTrace, Turn, TurnTotals
from ..llm.client import ChatResult, OpenRouterClient, ToolCallRequest
from .prompts import FINAL_TURN_MESSAGE, system_prompt
from .session import ToolSession
from .tools import add_thought, retry_key, schema_defaults, split_arguments

REASONING = None  # each model's default reasoning setting; frozen for the experiment

# Tool results the runner itself sends back, without calling the server
INVALID_ARGUMENTS = '{"error": "invalid_arguments", "message": "Arguments must be a JSON object."}'
NOT_RUN = '{"error": "not_run", "message": "Retry limit reached."}'

FINAL_ANSWER_MARKER = re.compile(r"final answer\s*\**\s*:", re.IGNORECASE)  # also matches "**Final Answer**:"


@dataclass(frozen=True)
class Budget:
    """Per-episode limits from docs/experiment-spec.md."""

    max_turns: int = 12
    max_retries: int = 3
    max_output_tokens: int = 8000
    final_reserve: int = 1000  # output tokens always kept for the final-answer turn


def extract_final_answer(text: str | None) -> str | None:
    """The first non-empty line after the last "Final Answer:", or the whole text if there is no marker."""
    if not text:
        return None
    markers = list(FINAL_ANSWER_MARKER.finditer(text))
    if not markers:
        return text.strip() or None

    for line in text[markers[-1].end():].split("\n"):
        answer = line.strip().strip("*").strip()
        if answer:
            return answer
    return None


def turn_totals(result: ChatResult) -> TurnTotals:
    """Tokens, cost and latency of one model call."""
    usage = result.usage
    tokens = TokenUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
    )
    return TurnTotals(tokens=tokens, cost_usd=usage.cost_usd or 0.0, latency_ms=result.latency_ms)


def episode_totals(turns: list[Turn], latency_ms: float) -> EpisodeTotals:
    """The agent's summed tokens and cost; latency is the episode's wall-clock time."""
    input_tokens = sum(turn.totals.tokens.input_tokens for turn in turns)
    output_tokens = sum(turn.totals.tokens.output_tokens for turn in turns)
    return EpisodeTotals(
        agent_tokens=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens),
        agent_cost_usd=sum(turn.totals.cost_usd for turn in turns),
        latency_ms=latency_ms,
    )


class EpisodeRun:
    """The state of one episode's tool calls: which calls were seen, and how many retries were used."""

    def __init__(self, session: ToolSession, defaults: dict, max_retries: int):
        self.session = session
        self.defaults = defaults
        self.max_retries = max_retries
        self.seen_calls: set[str] = set()
        self.retries_used = 0
        self.retry_limit_hit = False

    async def run_call(self, call: ToolCallRequest, thoughts: list[str]) -> tuple[dict, ToolCallTrace]:
        """Run one tool call; returns the tool message for the model and the trace entry."""
        parsed = split_arguments(call.arguments)
        if parsed.thought:
            thoughts.append(f"Thought: {parsed.thought}")
        args = parsed.args if parsed.args is not None else {"_unparsed": call.arguments}
        key = retry_key(call.name, parsed.args, self.defaults) if parsed.args is not None else None
        repeats = key in self.seen_calls

        # The 4th retry, and every call after it in the same turn, is not run
        if self.retry_limit_hit or (repeats and self.retries_used == self.max_retries):
            self.retry_limit_hit = True
            trace = ToolCallTrace(tool_name=call.name, tool_args=args, raw_tool_response=NOT_RUN, repeats_earlier_call=repeats)
            return tool_message(call, NOT_RUN), trace

        # Arguments that aren't a JSON object never reach the server
        if parsed.args is None:
            trace = ToolCallTrace(tool_name=call.name, tool_args=args, raw_tool_response=INVALID_ARGUMENTS)
            return tool_message(call, INVALID_ARGUMENTS), trace

        if repeats:
            self.retries_used += 1
        self.seen_calls.add(key)
        result = await self.session.call(call.name, parsed.args)
        trace = ToolCallTrace(
            tool_name=call.name,
            tool_args=parsed.args,
            raw_tool_response=result.text,
            is_error=result.is_error,
            harness_flags=result.harness_flags,
            proxy_latency_ms=result.proxy_latency_ms,
            repeats_earlier_call=repeats,
        )
        return tool_message(call, result.text), trace


def tool_message(call: ToolCallRequest, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call.id, "content": text}


async def run_episode(
    llm: OpenRouterClient,
    session: ToolSession,
    scenario: Scenario,
    model: str,
    condition: Condition,
    episode_id: str,
    budget: Budget = Budget(),
) -> Episode:
    """Run one episode. LLMError and ToolServerError propagate: they are infrastructure failures, not results."""
    start = time.perf_counter()
    server_tools = await session.list_tools()
    shown_tools = add_thought(server_tools)
    run = EpisodeRun(session, schema_defaults(server_tools), budget.max_retries)

    episode = Episode(
        episode_id=episode_id,
        scenario_id=scenario.scenario_id,
        dataset=scenario.dataset,
        model=model,
        condition=condition,
        fault_type=scenario.fault_type,
        question=scenario.question,
        gold_answer=scenario.gold_answer,
    )
    messages = [{"role": "system", "content": system_prompt(condition)}, {"role": "user", "content": scenario.question}]
    output_used = 0
    last_finish_reason = None
    index = 0

    while True:
        index += 1
        remaining = budget.max_output_tokens - output_used

        # No tokens left at all: stop without another model call
        if remaining <= 0:
            episode.stop_reason = "token_budget"
            break

        # A reached limit turns this turn into the final-answer turn (first match wins)
        limit = None
        if run.retry_limit_hit:
            limit = "retry_budget"
        elif remaining <= budget.final_reserve or last_finish_reason == "length":
            limit = "token_budget"
        elif index == budget.max_turns:
            limit = "max_turns"

        if limit is None:
            tool_choice, max_tokens = "auto", remaining - budget.final_reserve
        else:
            messages.append({"role": "user", "content": FINAL_TURN_MESSAGE})
            tool_choice, max_tokens = "none", remaining
        result = await llm.chat(
            model,
            messages,
            tools=shown_tools,  # kept on the final turn: Anthropic rejects tool history without tool definitions
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            reasoning=REASONING,
            tag="agent",
        )
        output_used += result.usage.output_tokens
        last_finish_reason = result.finish_reason
        turn = Turn(index=index, totals=turn_totals(result))
        texts = [result.content.strip()] if result.content and result.content.strip() else []
        episode.turns.append(turn)

        # Final-answer turn: tool calls are not run, but their thoughts are kept as visible text
        if limit is not None:
            for call in result.tool_calls:
                thought = split_arguments(call.arguments).thought
                if thought:
                    texts.append(f"Thought: {thought}")
            turn.assistant_text = "\n".join(texts)
            episode.final_answer = extract_final_answer(result.content)
            episode.stop_reason = limit
            break

        # A reply without tool calls is the final answer, unless it was cut off by max_tokens
        if not result.tool_calls:
            turn.assistant_text = "\n".join(texts)
            if result.finish_reason == "length":
                if texts:  # an empty assistant message would be rejected by some providers
                    messages.append(result.message)
                continue
            episode.final_answer = extract_final_answer(result.content)
            episode.stop_reason = "final_answer"
            break

        # Every tool call gets a tool message before the next model call
        messages.append(result.message)
        for call in result.tool_calls:
            message, trace = await run.run_call(call, texts)
            messages.append(message)
            turn.tool_calls.append(trace)
        turn.assistant_text = "\n".join(texts)

    episode.totals = episode_totals(episode.turns, (time.perf_counter() - start) * 1000)
    return episode
