from __future__ import annotations

import re
from dataclasses import dataclass

from .intent_frame import KnowledgeScope
from .execution_state import ExecutionState
from .intent_validation import ValidatedIntent
from .reference_extractor import extract_references
from .schemas import KnowledgeCitation, ToolUse
from .skills import SkillActivation
from .tool_result_projection import project_tool_result
from .tool_registry import ToolRegistry
from .tools import ToolContext
from .workflow_registry import (
    ArgumentSource,
    WorkflowRecipe,
    WorkflowStage,
)


@dataclass(frozen=True)
class WorkflowExecution:
    completed: bool
    tool_results: tuple[dict, ...]
    tool_usages: tuple[ToolUse, ...]
    prediction_ids: frozenset[str]
    citations: dict[str, KnowledgeCitation]
    knowledge_index_version: str | None = None
    active_skill: SkillActivation | None = None
    failure_code: str | None = None
    completed_goal_indexes: tuple[int, ...] = ()


class _InsufficientHistory(RuntimeError):
    pass


class DeterministicWorkflowExecutor:
    def __init__(self, tools: ToolRegistry):
        self._tools = tools

    @staticmethod
    def _history_ids(results: list[dict]) -> tuple[str, ...]:
        for result in reversed(results):
            items = result.get("items")
            if isinstance(items, list):
                values = tuple(
                    item["prediction_id"] for item in items
                    if isinstance(item, dict) and
                    isinstance(item.get("prediction_id"), str))
                return values
        return ()

    @staticmethod
    def _bound_prediction_ids(intent: ValidatedIntent) -> tuple[str, ...]:
        for goal in intent.goals:
            if goal.target.prediction_ids:
                return goal.target.prediction_ids
        return ()

    @staticmethod
    def _knowledge_scope(intent: ValidatedIntent) -> str:
        for goal in intent.goals:
            if goal.knowledge_scope is not None:
                return goal.knowledge_scope.value
        return KnowledgeScope.ALL.value

    @staticmethod
    def _knowledge_query(message: str) -> str:
        masked = extract_references(message).router_message
        value = re.sub(
            r"<(?:prediction|sample)_ref_[0-7]>", " ", masked)
        value = re.sub(r"<redacted_[a-z_]+>", " ", value)
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("knowledge query is empty after redaction")
        return value[:500]

    @classmethod
    def _arguments(
            cls, stage: WorkflowStage, recipe: WorkflowRecipe,
            intent: ValidatedIntent, results: list[dict],
            message: str) -> dict:
        source = stage.argument_source
        if source == ArgumentSource.BOUND_SAMPLE:
            sample = next(
                (goal.target.sample_index for goal in intent.goals
                 if goal.target.sample_index is not None), None)
            if sample is None:
                raise ValueError("workflow has no bound sample")
            return {"sample_index": sample}
        if source == ArgumentSource.BOUND_PREDICTION:
            values = cls._bound_prediction_ids(intent)
            if not values:
                raise ValueError("workflow has no bound prediction")
            return {"prediction_id": values[0]}
        if source == ArgumentSource.SESSION_HISTORY:
            return {"limit": 5}
        if source == ArgumentSource.LATEST_TWO_HISTORY:
            return {"limit": 2}
        if source in {
                ArgumentSource.LATEST_FROM_HISTORY,
                ArgumentSource.PREVIOUS_FROM_HISTORY,
                ArgumentSource.PAIR_FROM_HISTORY}:
            values = cls._history_ids(results)
            if len(values) < 2:
                raise _InsufficientHistory()
            if source == ArgumentSource.LATEST_FROM_HISTORY:
                return {"prediction_id": values[0]}
            if source == ArgumentSource.PREVIOUS_FROM_HISTORY:
                return {"prediction_id": values[1]}
            return {
                "prediction_id_a": values[0],
                "prediction_id_b": values[1],
            }
        if source == ArgumentSource.BOUND_EXPLICIT_PAIR:
            values = cls._bound_prediction_ids(intent)
            if len(values) != 2:
                raise ValueError("workflow has no bound prediction pair")
            return {
                "prediction_id_a": values[0],
                "prediction_id_b": values[1],
            }
        if source == ArgumentSource.KNOWLEDGE_QUERY:
            return {
                "query": cls._knowledge_query(message),
                "scope": cls._knowledge_scope(intent),
                "top_k": 5,
            }
        if source == ArgumentSource.TRUSTED_SKILL_ID:
            if recipe.trusted_skill_id is None:
                raise ValueError("workflow has no trusted skill")
            return {"skill_id": recipe.trusted_skill_id}
        raise ValueError("unsupported workflow argument source")

    @staticmethod
    def _result(
            completed: bool, results: list[dict], usages: list[ToolUse],
            prediction_ids: set[str],
            citations: dict[str, KnowledgeCitation],
            index_version: str | None, active_skill: SkillActivation | None,
            failure_code: str | None = None,
            state: ExecutionState | None = None) -> WorkflowExecution:
        return WorkflowExecution(
            completed=completed,
            tool_results=tuple(results),
            tool_usages=tuple(usages),
            prediction_ids=frozenset(prediction_ids),
            citations=dict(citations),
            knowledge_index_version=index_version,
            active_skill=active_skill,
            failure_code=failure_code,
            completed_goal_indexes=tuple(state.completed_goal_indexes) if state else (),
        )

    async def execute(
            self, recipe: WorkflowRecipe, intent: ValidatedIntent,
            context: ToolContext, message: str,
            max_tool_calls: int | None = None) -> WorkflowExecution:
        results: list[dict] = []
        llm_results: list[dict] = []
        usages: list[ToolUse] = []
        prediction_ids: set[str] = set()
        citations: dict[str, KnowledgeCitation] = {}
        index_version: str | None = None
        active_skill: SkillActivation | None = None
        domain_calls = 0
        state = ExecutionState(intent, 0, max_tool_calls or len(recipe.stages))
        for stage in recipe.stages:
            if stage.tool_name != "activate_skill":
                if max_tool_calls is not None and domain_calls >= max_tool_calls:
                    return self._result(
                        False, llm_results, usages, prediction_ids, citations,
                        index_version, active_skill, "tool_call_limit", state=state)
                domain_calls += 1
            try:
                arguments = self._arguments(
                    stage, recipe, intent, results, message)
            except _InsufficientHistory:
                return self._result(
                    False, llm_results, usages, prediction_ids, citations,
                    index_version, active_skill, "insufficient_history", state=state)
            except ValueError:
                return self._result(
                    False, llm_results, usages, prediction_ids, citations,
                    index_version, active_skill, "workflow_binding_failed", state=state)
            result = await self._tools.execute(
                stage.tool_name, arguments, context, active_skill)
            state.record(stage.tool_name, arguments, result.content,
                         result.usage, {})
            results.append(result.content)
            llm_results.append(project_tool_result(
                stage.tool_name, result.content, intent))
            usages.append(result.usage)
            prediction_ids.update(result.prediction_ids)
            citations.update(result.citations)
            if result.index_version is not None:
                index_version = result.index_version
            if result.skill_activation is not None:
                active_skill = result.skill_activation
            if result.usage.status != "success":
                return self._result(
                    False, llm_results, usages, prediction_ids, citations,
                    index_version, active_skill,
                    str(result.content.get(
                        "error", "tool_execution_failed")), state=state)
        return self._result(
            True, llm_results, usages, prediction_ids, citations,
            index_version, active_skill, state=state)
