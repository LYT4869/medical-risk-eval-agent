from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from pydantic import ValidationError

from .intent_frame import IntentFrame
from .intent_router_prompt import (
    ROUTER_SYSTEM_PROMPT,
    ROUTER_TOOL_NAME,
    router_function_definition,
)
from .llm_client import LlmClient, LlmError, LlmToolPolicy
from .reference_extractor import (
    ReferenceExtraction,
    sanitize_router_context_text,
)
from .schemas import LlmUsage, RecentMessage


class StructuredRouterError(RuntimeError):
    _CODES = {"intent_router_unavailable", "invalid_intent_frame"}

    def __init__(self, code: str):
        if code not in self._CODES:
            raise ValueError("unknown structured Router error code")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class StructuredRouterConfig:
    request_timeout_seconds: float = 3.0
    total_deadline_seconds: float = 7.0
    maximum_attempts: int = 2
    retry_backoff_seconds: float = 0.1
    context_messages: int = 4
    context_max_chars: int = 6000

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ValueError("request timeout must be positive")
        if self.maximum_attempts not in {1, 2}:
            raise ValueError("maximum attempts must be 1 or 2")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry backoff must be non-negative")
        required = (
            self.request_timeout_seconds * self.maximum_attempts +
            (self.retry_backoff_seconds
             if self.maximum_attempts > 1 else 0.0))
        if self.total_deadline_seconds < required:
            raise ValueError("total deadline cannot cover the attempt budget")
        if self.context_messages < 0 or self.context_messages > 4:
            raise ValueError("context messages must be between 0 and 4")
        if self.context_max_chars <= 0:
            raise ValueError("context character budget must be positive")


@dataclass(frozen=True)
class RouterContext:
    message: str
    recent_messages: tuple[RecentMessage, ...]
    current_prediction_available: bool
    references: ReferenceExtraction


@dataclass(frozen=True)
class StructuredRoute:
    frame: IntentFrame
    usage: LlmUsage | None
    attempt_count: int
    repaired: bool


def _add_usage(left: LlmUsage | None, right: LlmUsage | None) -> LlmUsage | None:
    if left is None:
        return right
    if right is None:
        return left
    return LlmUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


class StructuredIntentRouter:
    def __init__(
            self, llm: LlmClient,
            config: StructuredRouterConfig | None = None,
            *, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
            monotonic: Callable[[], float] = time.monotonic):
        self._llm = llm
        self._config = config or StructuredRouterConfig()
        self._sleep = sleep
        self._monotonic = monotonic

    def _recent_messages(
            self, values: tuple[RecentMessage, ...]) -> list[dict[str, str]]:
        selected = values[-self._config.context_messages:]
        remaining = self._config.context_max_chars
        result: list[dict[str, str]] = []
        for item in reversed(selected):
            if remaining <= 0:
                break
            content = sanitize_router_context_text(item.content)
            content = content[-remaining:]
            remaining -= len(content)
            result.append({"role": item.role, "content": content})
        result.reverse()
        return result

    def _messages(self, context: RouterContext, *, repair: bool) -> list[dict]:
        payload = {
            "current_message": sanitize_router_context_text(context.message),
            "recent_final_messages": self._recent_messages(
                context.recent_messages),
            "symbolic_session_state": {
                "current_prediction_available":
                    context.current_prediction_available,
            },
            "reference_candidates": {
                "prediction": [
                    f"prediction_ref_{index}"
                    for index in range(len(context.references.prediction_ids))
                ],
                "sample": [
                    f"sample_ref_{index}"
                    for index in range(len(context.references.sample_indexes))
                ],
            },
        }
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"))},
        ]
        if repair:
            messages.append({
                "role": "system",
                "content": (
                    "FORMAT REPAIR: the previous response violated the required "
                    "function or IntentFrame schema. Re-read the same Router input "
                    "and call route_user_request once with strictly valid arguments. "
                    "Do not add prose or new facts."),
            })
        return messages

    @staticmethod
    def _parse(turn) -> IntentFrame:
        if (len(turn.tool_calls) != 1 or
                turn.tool_calls[0].name != ROUTER_TOOL_NAME):
            raise ValueError("required Router function call is missing")
        return IntentFrame.model_validate(turn.tool_calls[0].arguments)

    async def route(self, context: RouterContext) -> StructuredRoute:
        deadline = self._monotonic() + self._config.total_deadline_seconds
        usage: LlmUsage | None = None
        repair = False
        for attempt in range(1, self._config.maximum_attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise StructuredRouterError("intent_router_unavailable")
            timeout = min(self._config.request_timeout_seconds, remaining)
            try:
                turn = await asyncio.wait_for(
                    self._llm.complete(
                        self._messages(context, repair=repair),
                        [router_function_definition()], timeout,
                        LlmToolPolicy.required(ROUTER_TOOL_NAME)),
                    timeout=timeout,
                )
                usage = _add_usage(usage, turn.usage)
                frame = self._parse(turn)
                return StructuredRoute(frame, usage, attempt, repair)
            except asyncio.TimeoutError:
                retryable = True
                protocol_failure = False
            except LlmError as exc:
                retryable = exc.retryable
                protocol_failure = exc.code == "invalid_response"
                if not retryable and not protocol_failure:
                    raise StructuredRouterError(
                        "intent_router_unavailable") from exc
            except (ValidationError, ValueError, TypeError, KeyError):
                retryable = False
                protocol_failure = True

            if attempt >= self._config.maximum_attempts:
                code = (
                    "invalid_intent_frame" if protocol_failure
                    else "intent_router_unavailable")
                raise StructuredRouterError(code)
            if protocol_failure:
                repair = True
                continue
            if not retryable:
                raise StructuredRouterError("intent_router_unavailable")
            if self._config.retry_backoff_seconds:
                await self._sleep(self._config.retry_backoff_seconds)
        raise StructuredRouterError("intent_router_unavailable")
