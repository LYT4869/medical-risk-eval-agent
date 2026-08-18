from __future__ import annotations

import json
import math
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
    KnowledgeCitation,
    KnowledgeSearchArgs,
    KnowledgeToolOutput,
    PredictionToolOutput,
    PredictSampleArgs,
    PredictionIdArgs,
    SkillActivationArgs,
    ToolUse,
)
from .skills import SkillActivation, SkillCatalog
from .tools import BackendToolClient, ToolContext
from .tools.backend import ToolExecutionError
from .tools.knowledge import KnowledgeClient, KnowledgeToolError


@dataclass
class ToolResult:
    content: dict[str, Any]
    usage: ToolUse
    prediction_ids: set[str]
    citations: dict[str, KnowledgeCitation]
    index_version: str | None = None
    skill_activation: SkillActivation | None = None


class ToolRegistry:
    def __init__(self, backend: BackendToolClient,
                 knowledge: KnowledgeClient | None = None,
                 skills: SkillCatalog | None = None):
        self._backend = backend
        self._knowledge = knowledge
        self._skills = skills

    @staticmethod
    def native_tool_names() -> set[str]:
        return {"predict_sample", "get_prediction", "get_explanation",
                "get_prediction_history", "compare_predictions"}

    def available_tool_names(self) -> set[str]:
        result = self.native_tool_names()
        if self._knowledge is not None:
            result.add("search_medical_knowledge")
        return result

    def definitions(self, context: ToolContext | None = None,
                    active_skill: SkillActivation | None = None) -> list[dict[str, Any]]:
        def definition(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
            return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}
        result = [
            definition("predict_sample", "Run treeSem for a non-negative demo sample index.", PredictSampleArgs.model_json_schema()),
            definition("get_prediction", "Read a stored prediction.", PredictionIdArgs.model_json_schema()),
            definition("get_explanation", "Read important features and the decision path.", PredictionIdArgs.model_json_schema()),
            definition("get_prediction_history", "List recent predictions in the bound session.", HistoryArgs.model_json_schema()),
            definition("compare_predictions", "Compute deterministic differences between two predictions.", CompareArgs.model_json_schema()),
        ]
        if self._knowledge is not None:
            result.append(definition(
                "search_medical_knowledge",
                "Search curated treeSem and authoritative PPH knowledge. Never include patient identifiers in the query.",
                KnowledgeSearchArgs.model_json_schema()))
        if self._skills is not None and active_skill is None and context is not None:
            visible = self._skills.summaries(context.actor_role)
            if visible:
                result.append(definition(
                    "activate_skill",
                    "Activate one trusted workflow skill from this catalog: " +
                    json.dumps(visible, ensure_ascii=False),
                    SkillActivationArgs.model_json_schema()))
        if active_skill is not None:
            allowed = active_skill.required_tools
            result = [item for item in result
                      if item["function"]["name"] in allowed]
        names = [item["function"]["name"] for item in result]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate tool name")
        return result

    def skill_catalog_prompt(self, role: str) -> str | None:
        if self._skills is None:
            return None
        summaries = self._skills.summaries(role)
        if not summaries:
            return None
        return "Trusted skill catalog (metadata only): " + json.dumps(
            summaries, ensure_ascii=False, separators=(",", ":"))

    async def execute(self, name: str, arguments: dict[str, Any], context: ToolContext,
                      active_skill: SkillActivation | None = None) -> ToolResult:
        started = time.monotonic()
        try:
            if (active_skill is not None and
                    name not in active_skill.required_tools):
                raise ToolExecutionError("tool is not allowed by the active skill")
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
            elif name == "search_medical_knowledge":
                if self._knowledge is None or not context.knowledge_capability_token:
                    raise ToolExecutionError("knowledge tool is unavailable")
                args = KnowledgeSearchArgs.model_validate(arguments)
                raw = await self._knowledge.search(
                    context.knowledge_capability_token, args.query, args.scope, args.top_k)
                parsed = KnowledgeToolOutput.model_validate(raw)
                if any(not math.isfinite(item.score) for item in parsed.results):
                    raise ToolExecutionError("knowledge tool returned a non-finite score")
                body = parsed.model_dump()
                citations = {
                    item.citation_id: KnowledgeCitation(
                        citation_id=item.citation_id, source_id=item.source_id,
                        title=item.title, section=item.section, page=item.page,
                        publisher=item.publisher, published_at=item.published_at,
                        url=item.url)
                    for item in parsed.results
                }
                return ToolResult(
                    body, ToolUse(name=name, status="success",
                                  duration_ms=int((time.monotonic()-started)*1000)),
                    set(), citations, parsed.index_version)
            elif name == "activate_skill":
                if self._skills is None or active_skill is not None:
                    raise ToolExecutionError("skill activation is unavailable")
                args = SkillActivationArgs.model_validate(arguments)
                activation = self._skills.activate(args.skill_id, context.actor_role)
                body = {"skill_id": activation.skill_id,
                        "version": activation.version,
                        "status": "activated"}
                return ToolResult(
                    body, ToolUse(name=name, status="success",
                                  duration_ms=int((time.monotonic()-started)*1000)),
                    set(), {}, skill_activation=activation)
            else:
                raise ToolExecutionError("unknown tool")
            ids = self._prediction_ids(body)
            return ToolResult(body, ToolUse(name=name, status="success", duration_ms=int((time.monotonic()-started)*1000)), ids, {})
        except (ValidationError, ToolExecutionError, KnowledgeToolError, ValueError) as exc:
            safe = {"error": "invalid_tool_arguments" if isinstance(exc, ValidationError) else "tool_execution_failed"}
            return ToolResult(safe, ToolUse(name=name, status="error", duration_ms=int((time.monotonic()-started)*1000)), set(), {})

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
