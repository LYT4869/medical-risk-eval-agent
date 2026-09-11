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

    def __init__(self, code: str, *, attempt_count: int = 1,
                 repaired: bool = False):
        if code not in self._CODES:
            raise ValueError("unknown structured Router error code")
        super().__init__(code)
        self.code = code
        self.attempt_count = attempt_count
        self.repaired = repaired


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


@dataclass(frozen=True)
class _RepairHint:
    reason: str
    paths: tuple[str, ...] = ()


class _RouterFrameError(ValueError):
    def __init__(self, hint: _RepairHint):
        super().__init__(hint.reason)
        self.hint = hint


_SAFE_FRAME_PATH_PARTS = frozenset({
    "schema_version", "goals", "intent", "target", "type",
    "explicit_reference_index", "second_explicit_reference_index",
    "sample_reference_index", "requested_aspects", "knowledge_scope",
    "evidence", "constraints", "excluded_intents", "excluded_aspects",
    "unresolved_references", "needs_clarification", "requested_skill",
})


def _safe_validation_path(location: tuple[object, ...]) -> str:
    path = ""
    for part in location:
        if isinstance(part, int) and 0 <= part <= 7:
            path += f"[{part}]"
        elif isinstance(part, str) and part in _SAFE_FRAME_PATH_PARTS:
            path += ("." if path else "") + part
        else:
            return "unknown_field"
    return path or "intent_frame"


def _validation_hint(error: ValidationError) -> _RepairHint:
    details = error.errors(
        include_url=False, include_context=False, include_input=False)
    kinds = {str(item.get("type", "")) for item in details}
    if "missing" in kinds:
        reason = "missing_required_field"
    elif any(kind == "enum" or kind.endswith("_type") for kind in kinds):
        reason = "invalid_enum_or_type"
    elif "extra_forbidden" in kinds:
        reason = "unexpected_field"
    elif any(kind == "value_error" for kind in kinds):
        reason = "cross_field_violation"
    else:
        reason = "schema_constraint_violation"
    paths = tuple(dict.fromkeys(
        _safe_validation_path(tuple(item.get("loc", ())))
        for item in details))[:4]
    return _RepairHint(reason, paths)


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

    def _messages(
            self, context: RouterContext,
            *, repair: _RepairHint | None) -> list[dict]:
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
        if repair is not None:
            safe_feedback = json.dumps({
                "reason": repair.reason,
                "paths": list(repair.paths),
            }, separators=(",", ":"))
            messages.append({
                "role": "system",
                "content": (
                    "FORMAT REPAIR: the previous response violated the required "
                    "function or IntentFrame schema. Safe validation feedback: " +
                    safe_feedback + ". Re-read the same Router input "
                    "and call route_user_request once with strictly valid arguments. "
                    "Do not add prose or new facts."),
            })
        return messages

    @staticmethod
    def _parse(turn) -> IntentFrame:
        if len(turn.tool_calls) != 1:
            raise _RouterFrameError(_RepairHint("wrong_tool_call_count"))
        if turn.tool_calls[0].name != ROUTER_TOOL_NAME:
            raise _RouterFrameError(_RepairHint("wrong_tool_name"))
        try:
            return IntentFrame.model_validate(turn.tool_calls[0].arguments)
        except ValidationError as exc:
            raise _RouterFrameError(_validation_hint(exc)) from exc

    async def route(self, context: RouterContext) -> StructuredRoute:
        deadline = self._monotonic() + self._config.total_deadline_seconds
        usage: LlmUsage | None = None
        repair: _RepairHint | None = None
        for attempt in range(1, self._config.maximum_attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                raise StructuredRouterError(
                    "intent_router_unavailable", attempt_count=attempt - 1,
                    repaired=repair is not None)
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
                return StructuredRoute(
                    frame, usage, attempt, repair is not None)
            except asyncio.TimeoutError:
                retryable = True
                protocol_failure = False
            except LlmError as exc:
                retryable = exc.retryable
                protocol_failure = exc.code == "invalid_response"
                if not retryable and not protocol_failure:
                    raise StructuredRouterError(
                        "intent_router_unavailable", attempt_count=attempt,
                        repaired=repair is not None) from exc
                repair_candidate = _RepairHint("invalid_upstream_response")
            except _RouterFrameError as exc:
                retryable = False
                protocol_failure = True
                repair_candidate = exc.hint
            except (ValidationError, ValueError, TypeError, KeyError):
                retryable = False
                protocol_failure = True
                repair_candidate = _RepairHint("invalid_upstream_response")

            if attempt >= self._config.maximum_attempts:
                code = (
                    "invalid_intent_frame" if protocol_failure
                    else "intent_router_unavailable")
                raise StructuredRouterError(
                    code, attempt_count=attempt,
                    repaired=repair is not None)
            if protocol_failure:
                repair = repair_candidate
                continue
            if not retryable:
                raise StructuredRouterError(
                    "intent_router_unavailable", attempt_count=attempt,
                    repaired=repair is not None)
            if self._config.retry_backoff_seconds:
                await self._sleep(self._config.retry_backoff_seconds)
        raise StructuredRouterError(
            "intent_router_unavailable",
            attempt_count=self._config.maximum_attempts,
            repaired=repair is not None)
