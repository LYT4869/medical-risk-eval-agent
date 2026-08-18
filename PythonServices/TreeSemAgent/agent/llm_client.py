from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

try:
    import httpx
except ModuleNotFoundError:  # Unit tests using ScriptedLlmClient need no HTTP stack.
    httpx = None  # type: ignore[assignment]

from .schemas import LlmToolCall, LlmTurn


class LlmError(RuntimeError):
    pass


class LlmClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float) -> LlmTurn: ...


@dataclass(frozen=True)
class OpenAiCompatibleConfig:
    base_url: str
    model: str
    api_key: str = ""
    connect_timeout_seconds: float = 1.0
    request_timeout_seconds: float = 20.0


class OpenAiCompatibleClient:
    def __init__(self, config: OpenAiCompatibleConfig):
        if httpx is None:
            raise RuntimeError("httpx is required for the OpenAI-compatible client")
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float) -> LlmTurn:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        payload = {"model": self._config.model, "messages": messages, "tools": tools, "tool_choice": "auto"}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers,
                    timeout=min(timeout, self._config.request_timeout_seconds),
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise LlmError(f"retryable upstream status {response.status_code}")
                response.raise_for_status()
                body = response.json()
                message = body["choices"][0]["message"]
                calls = []
                for raw in message.get("tool_calls", []):
                    arguments = raw["function"].get("arguments", "{}")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    calls.append(LlmToolCall(id=raw["id"], name=raw["function"]["name"], arguments=arguments))
                content = message.get("content")
                grounding = []
                if content and content.lstrip().startswith("{"):
                    try:
                        structured = json.loads(content)
                        content = structured.get("answer", content)
                        grounding = structured.get("grounding_prediction_ids", [])
                    except (ValueError, TypeError):
                        pass
                return LlmTurn(content=content, tool_calls=calls, grounding_prediction_ids=grounding)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, LlmError) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.05)
        raise LlmError("LLM request failed") from last_error


class ScriptedLlmClient:
    """Deterministic test client; each complete() consumes one scripted turn."""
    def __init__(self, turns: list[LlmTurn]):
        self.turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: float) -> LlmTurn:
        del tools, timeout
        self.requests.append(messages)
        if not self.turns:
            raise LlmError("script exhausted")
        return self.turns.pop(0)
