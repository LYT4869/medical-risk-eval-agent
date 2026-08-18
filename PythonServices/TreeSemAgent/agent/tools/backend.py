from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

from ..observability import TraceState, metrics, trace_event

try:
    import httpx
except ModuleNotFoundError:  # Fake backend tests do not perform HTTP.
    httpx = None  # type: ignore[assignment]


class ToolExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    capability_token: str | None = None
    knowledge_capability_token: str | None = None
    actor_role: str = "patient"
    trace: TraceState | None = None


class BackendToolClient:
    def __init__(self, base_url: str, timeout_seconds: float = 5.0, max_response_bytes: int = 1_048_576):
        if httpx is None:
            raise RuntimeError("httpx is required for backend tool calls")
        self._timeout = timeout_seconds
        self._max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _headers(context: ToolContext) -> dict[str, str]:
        headers = {"X-TreeSem-Session-Id": context.session_id}
        if context.capability_token:
            headers["Authorization"] = f"Bearer {context.capability_token}"
        if context.trace:
            headers["X-Request-Id"] = context.trace.request_id
            headers["traceparent"] = context.trace.traceparent
        return headers

    @staticmethod
    def _validate_finite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ToolExecutionError("tool backend returned a non-finite number")
        if isinstance(value, dict):
            for item in value.values():
                BackendToolClient._validate_finite(item)
        elif isinstance(value, list):
            for item in value:
                BackendToolClient._validate_finite(item)

    async def _request(self, method: str, path: str, context: ToolContext,
                       operation: str,
                       payload: dict[str, Any] | None = None) -> dict[str, Any]:
        attempts = 2 if method == "GET" else 1
        error: Exception | None = None
        for attempt in range(attempts):
            trace = context.trace.child() if context.trace else None
            traced_context = ToolContext(
                context.session_id, context.capability_token,
                context.knowledge_capability_token, context.actor_role, trace)
            started = time.monotonic()
            try:
                response = await self._client.request(
                    method, path, headers=self._headers(traced_context), json=payload)
                if len(response.content) > self._max_response_bytes:
                    raise ToolExecutionError("tool response exceeded the size limit")
                body = response.json()
                if response.status_code >= 400:
                    code = body.get("error", "tool_backend_error") if isinstance(body, dict) else "tool_backend_error"
                    raise ToolExecutionError(f"tool backend rejected request: {code}")
                if not isinstance(body, dict):
                    raise ToolExecutionError("tool backend returned a non-object response")
                self._validate_finite(body)
                metrics.increment("treesem_agent_tool_calls_total",
                                  tool=operation, result="success")
                metrics.observe("treesem_agent_tool_duration_seconds",
                                time.monotonic() - started, tool=operation)
                if trace:
                    trace_event(trace, "agent.native_tool", started, "success",
                                method=method, status=response.status_code)
                return body
            except ToolExecutionError:
                metrics.increment("treesem_agent_tool_calls_total",
                                  tool=operation, result="error")
                metrics.observe("treesem_agent_tool_duration_seconds",
                                time.monotonic() - started, tool=operation)
                if trace:
                    trace_event(trace, "agent.native_tool", started, "error",
                                method=method, error_code="tool_rejected")
                raise
            except (httpx.TransportError, ValueError) as exc:
                error = exc
                if trace:
                    trace_event(trace, "agent.native_tool", started, "error",
                                method=method, error_code="transport_error")
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.03)
        metrics.increment("treesem_agent_tool_calls_total",
                          tool=operation, result="unavailable")
        raise ToolExecutionError("tool backend is unavailable") from error

    async def predict_sample(self, context: ToolContext, sample_index: int) -> dict[str, Any]:
        return await self._request("POST", "/internal/v1/predictions", context,
                                   "predict_sample", {"sample_index": sample_index})

    async def get_prediction(self, context: ToolContext, prediction_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/v1/predictions/{prediction_id}",
                                   context, "get_prediction")

    async def get_explanation(self, context: ToolContext, prediction_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/v1/explanations/{prediction_id}",
                                   context, "get_explanation")

    async def get_history(self, context: ToolContext, limit: int, cursor: str | None) -> dict[str, Any]:
        path = f"/internal/v1/sessions/{context.session_id}/history?limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"
        return await self._request("GET", path, context, "get_prediction_history")

    async def compare(self, context: ToolContext, prediction_id_a: str, prediction_id_b: str) -> dict[str, Any]:
        return await self._request("POST", "/internal/v1/comparisons", context,
                                   "compare_predictions", {
            "prediction_id_a": prediction_id_a, "prediction_id_b": prediction_id_b,
        })
