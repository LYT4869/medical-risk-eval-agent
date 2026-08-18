from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from .schemas import (
    CompareArgs,
    ComparisonToolOutput,
    ExplanationToolOutput,
    HistoryArgs,
    HistoryToolOutput,
    PredictionToolOutput,
    PredictSampleArgs,
    PredictionIdArgs,
    ToolUse,
)
from .tools import BackendToolClient, ToolContext
from .tools.backend import ToolExecutionError


@dataclass
class ToolResult:
    content: dict[str, Any]
    usage: ToolUse
    prediction_ids: set[str]


class ToolRegistry:
    def __init__(self, backend: BackendToolClient):
        self._backend = backend

    def definitions(self) -> list[dict[str, Any]]:
        def definition(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
            return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
        return [
            definition("predict_sample", "Run treeSem for a non-negative demo sample index.", PredictSampleArgs.model_json_schema()),
            definition("get_prediction", "Read a stored prediction.", PredictionIdArgs.model_json_schema()),
            definition("get_explanation", "Read important features and the decision path.", PredictionIdArgs.model_json_schema()),
            definition("get_prediction_history", "List recent predictions in the bound session.", HistoryArgs.model_json_schema()),
            definition("compare_predictions", "Compute deterministic differences between two predictions.", CompareArgs.model_json_schema()),
        ]

    async def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            if name == "predict_sample":
                args = PredictSampleArgs.model_validate(arguments)
                body = await self._backend.predict_sample(context, args.sample_index)
                body = PredictionToolOutput.model_validate(body).model_dump()
            elif name == "get_prediction":
                args = PredictionIdArgs.model_validate(arguments)
                body = await self._backend.get_prediction(context, args.prediction_id)
                body = PredictionToolOutput.model_validate(body).model_dump()
            elif name == "get_explanation":
                args = PredictionIdArgs.model_validate(arguments)
                body = await self._backend.get_explanation(context, args.prediction_id)
                body = ExplanationToolOutput.model_validate(body).model_dump()
            elif name == "get_prediction_history":
                args = HistoryArgs.model_validate(arguments)
                body = await self._backend.get_history(context, args.limit, args.cursor)
                body = HistoryToolOutput.model_validate(body).model_dump()
            elif name == "compare_predictions":
                args = CompareArgs.model_validate(arguments)
                body = await self._backend.compare(context, args.prediction_id_a, args.prediction_id_b)
                body = ComparisonToolOutput.model_validate(body).model_dump()
            else:
                raise ToolExecutionError("unknown tool")
            ids = self._prediction_ids(body)
            return ToolResult(body, ToolUse(name=name, status="success", duration_ms=int((time.monotonic()-started)*1000)), ids)
        except (ValidationError, ToolExecutionError) as exc:
            safe = {"error": "invalid_tool_arguments" if isinstance(exc, ValidationError) else "tool_execution_failed"}
            return ToolResult(safe, ToolUse(name=name, status="error", duration_ms=int((time.monotonic()-started)*1000)), set())

    @staticmethod
    def _prediction_ids(value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "prediction_id" and isinstance(item, str):
                    found.add(item)
                else:
                    found.update(ToolRegistry._prediction_ids(item))
        elif isinstance(value, list):
            for item in value:
                found.update(ToolRegistry._prediction_ids(item))
        return found
