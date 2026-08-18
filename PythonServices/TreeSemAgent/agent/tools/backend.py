from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

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

    async def _request(self, method: str, path: str, context: ToolContext, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        attempts = 2 if method == "GET" else 1
        error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.request(method, path, headers=self._headers(context), json=payload)
                if len(response.content) > self._max_response_bytes:
                    raise ToolExecutionError("tool response exceeded the size limit")
                body = response.json()
                if response.status_code >= 400:
                    code = body.get("error", "tool_backend_error") if isinstance(body, dict) else "tool_backend_error"
                    raise ToolExecutionError(f"tool backend rejected request: {code}")
                if not isinstance(body, dict):
                    raise ToolExecutionError("tool backend returned a non-object response")
                self._validate_finite(body)
                return body
            except (httpx.TransportError, ValueError) as exc:
                error = exc
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.03)
        raise ToolExecutionError("tool backend is unavailable") from error

    async def predict_sample(self, context: ToolContext, sample_index: int) -> dict[str, Any]:
        return await self._request("POST", "/internal/v1/predictions", context, {"sample_index": sample_index})

    async def get_prediction(self, context: ToolContext, prediction_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/v1/predictions/{prediction_id}", context)

    async def get_explanation(self, context: ToolContext, prediction_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/internal/v1/explanations/{prediction_id}", context)

    async def get_history(self, context: ToolContext, limit: int, cursor: str | None) -> dict[str, Any]:
        path = f"/internal/v1/sessions/{context.session_id}/history?limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"
        return await self._request("GET", path, context)

    async def compare(self, context: ToolContext, prediction_id_a: str, prediction_id_b: str) -> dict[str, Any]:
        return await self._request("POST", "/internal/v1/comparisons", context, {
            "prediction_id_a": prediction_id_a, "prediction_id_b": prediction_id_b,
        })
