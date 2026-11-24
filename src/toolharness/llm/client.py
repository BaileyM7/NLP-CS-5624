"""OpenRouter chat client: one async interface for every model, with tool calls, retries and cost tracking.

    async with OpenRouterClient() as llm:
        result = await llm.chat("haiku-5.5", messages, tools=tools)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel

from .models import ModelSpec, get_model

OPENROUTER_URL = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}  # transport errors, never counted as tool failures
MAX_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 30.0
TIMEOUT_S = 300.0


class LLMError(RuntimeError):
    """A model call that failed after every retry, or with an error that retrying won't fix."""


class Usage(BaseModel):
    """Token counts and cost for one call. input/output match the trace's TokenUsage fields."""

    input_tokens: int = 0
    output_tokens: int = 0  # includes reasoning tokens; this is what the 8,000-token episode budget counts
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None  # billed cost reported by OpenRouter; None if it didn't report one


class ToolCallRequest(BaseModel):
    """One tool call the model asked for. `arguments` is the raw JSON string the model wrote."""

    id: str
    name: str
    arguments: str


class ChatResult(BaseModel):
    model: str  # short name
    model_id: str  # OpenRouter ID
    content: str | None
    tool_calls: list[ToolCallRequest]
    finish_reason: str | None
    message: dict[str, Any]  # the assistant message as sent back, ready to append to the conversation
    usage: Usage
    usage_reported: bool = True
    latency_ms: float  # the successful attempt only
    attempts: int  # 1 + number of retries


def openrouter_api_key() -> str:
    """The API key from OPENROUTER_API_KEY (or .env)."""
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def tools_from_mcp(mcp_tools) -> list[dict[str, Any]]:
    """Convert MCP tool definitions (from list_tools) to OpenAI-style function tools."""
    tools = []
    for tool in mcp_tools:
        function = {"name": tool.name, "description": tool.description or "", "parameters": tool.input_schema}
        tools.append({"type": "function", "function": function})
    return tools


def read_usage(usage) -> Usage:
    """Token counts and cost from a response's usage block (fields may be missing)."""
    if usage is None:
        return Usage()
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return Usage(
        input_tokens=usage.prompt_tokens or 0,
        output_tokens=usage.completion_tokens or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", None) or 0,
        cached_tokens=getattr(prompt_details, "cached_tokens", None) or 0,
        cost_usd=getattr(usage, "cost", None),
    )


def to_result(spec: ModelSpec, response, latency_ms: float, attempts: int) -> ChatResult:
    """Turn an OpenAI-style chat completion into a ChatResult."""
    choice = response.choices[0]
    message = choice.message
    tool_calls = []
    for call in message.tool_calls or []:
        function = getattr(call, "function", None)  # skip non-function (custom) tool calls
        if function is not None:
            tool_calls.append(ToolCallRequest(id=call.id, name=function.name, arguments=function.arguments or ""))
    return ChatResult(
        model=spec.name,
        model_id=spec.openrouter_id,
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        message=message.model_dump(exclude_none=True),
        usage=read_usage(response.usage),
        usage_reported=(
            response.usage is not None
            and response.usage.prompt_tokens is not None
            and response.usage.completion_tokens is not None
        ),
        latency_ms=latency_ms,
        attempts=attempts,
    )


def provider_detail(body) -> str:
    """The upstream provider's own error message, which OpenRouter puts in error.metadata.raw."""
    metadata = body.get("metadata") if isinstance(body, dict) else None
    if not isinstance(metadata, dict) or not metadata.get("raw"):
        return ""
    provider = metadata.get("provider_name") or "provider"
    return f" [{provider}: {str(metadata['raw'])[:500]}]"


def retry_after_seconds(response) -> float | None:
    """The Retry-After header in seconds, if the server sent one as a number."""
    if response is None:
        return None
    try:
        return float(response.headers.get("retry-after", ""))
    except ValueError:
        return None


class OpenRouterClient:
    """Async chat client for OpenRouter. Use it as `async with OpenRouterClient() as llm:`."""

    def __init__(
        self,
        api_key: str | None = None,
        usage_log: str | Path | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        base_delay: float = BASE_DELAY_S,
        http_client=None,
    ):
        # The SDK's own retries are off: _create_with_retries handles them so attempts get counted
        self.openai = AsyncOpenAI(
            base_url=OPENROUTER_URL,
            api_key=api_key or openrouter_api_key(),
            max_retries=0,
            timeout=TIMEOUT_S,
            http_client=http_client,
        )
        self.usage_log = Path(usage_log) if usage_log else None
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self.openai.close()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        max_tokens: int | None = None,
        reasoning: dict[str, Any] | None = None,
        tag: str = "",
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Send one chat request and return the reply.

        model: a short name or OpenRouter ID from the registry.
        reasoning: OpenRouter's reasoning settings, e.g. {"effort": "low"}; None = the model's default.
        tag: free text saved in the usage log (e.g. "agent", "verifier", "judge").
        """
        spec = get_model(model)
        request: dict[str, Any] = {
            "model": spec.openrouter_id,
            "messages": messages,
            "extra_body": {"usage": {"include": True}},  # ask OpenRouter to report the billed cost
        }
        if tools:
            request["tools"] = tools
        if tool_choice:
            request["tool_choice"] = tool_choice
        if max_tokens:
            request["max_tokens"] = max_tokens
        if reasoning:
            request["extra_body"]["reasoning"] = reasoning
        if response_format is not None:
            request["response_format"] = response_format

        response, latency_ms, attempts = await self._create_with_retries(request)
        result = to_result(spec, response, latency_ms, attempts)
        self._log(result, tag)
        return result

    async def _create_with_retries(self, request: dict[str, Any]):
        """Call the API, retrying transport errors with backoff. Returns (response, latency_ms, attempts)."""
        for attempt in range(1, self.max_attempts + 1):
            retry_after = None
            start = time.perf_counter()
            try:
                response = await self.openai.chat.completions.create(**request)
            except APIConnectionError as error:  # includes timeouts
                reason = f"{type(error).__name__}: {error}"
            except APIStatusError as error:
                detail = provider_detail(error.body)
                if error.status_code not in RETRY_STATUS:
                    raise LLMError(f"{request['model']}: HTTP {error.status_code}: {error.message}{detail}") from error
                reason = f"HTTP {error.status_code}{detail}"
                retry_after = retry_after_seconds(error.response)
            else:
                # OpenRouter can return status 200 with no choices, or with finish_reason "error" when the provider failed mid-reply
                if not response.choices:
                    reason = f"no choices in the response: {getattr(response, 'error', None)}"
                elif response.choices[0].finish_reason == "error":
                    reason = "the provider ended the reply with finish_reason=error"
                else:
                    return response, (time.perf_counter() - start) * 1000, attempt

            if attempt == self.max_attempts:
                raise LLMError(f"{request['model']}: gave up after {attempt} attempts ({reason})")
            await asyncio.sleep(self.retry_delay(attempt, retry_after))

    def retry_delay(self, attempt: int, retry_after: float | None) -> float:
        """Exponential backoff with jitter; a server's Retry-After wins when it's longer."""
        delay = min(self.base_delay * 2 ** (attempt - 1), MAX_DELAY_S) + random.uniform(0, self.base_delay)
        if retry_after is not None:
            delay = max(delay, min(retry_after, MAX_DELAY_S))
        return delay

    def _log(self, result: ChatResult, tag: str) -> None:
        """Append one row per call to the usage log, if there is one."""
        if self.usage_log is None:
            return
        row = {
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
            "tag": tag,
            "model": result.model,
            "model_id": result.model_id,
            **result.usage.model_dump(),
            "latency_ms": round(result.latency_ms, 1),
            "attempts": result.attempts,
            "finish_reason": result.finish_reason,
        }
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
