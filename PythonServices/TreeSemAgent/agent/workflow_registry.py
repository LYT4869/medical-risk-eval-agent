from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from .intent_frame import IntentKind, TargetKind
from .intent_validation import BoundGoal, ValidatedIntent


class WorkflowRegistryError(ValueError):
    pass


class ArgumentSource(str, Enum):
    BOUND_SAMPLE = "bound_sample"
    BOUND_PREDICTION = "bound_prediction"
    SESSION_HISTORY = "session_history"
    LATEST_TWO_HISTORY = "latest_two_history"
    LATEST_FROM_HISTORY = "latest_from_history"
    PREVIOUS_FROM_HISTORY = "previous_from_history"
    PAIR_FROM_HISTORY = "pair_from_history"
    BOUND_EXPLICIT_PAIR = "bound_explicit_pair"
    KNOWLEDGE_QUERY = "knowledge_query"
    TRUSTED_SKILL_ID = "trusted_skill_id"


class RendererKind(str, Enum):
    PREDICTION_COMPLETED = "prediction_completed"
    PREDICTION_DATA_AVAILABLE = "prediction_data_available"
    EXPLANATION_DATA_AVAILABLE = "explanation_data_available"
    HISTORY_DATA_AVAILABLE = "history_data_available"
    COMPARISON_DATA_AVAILABLE = "comparison_data_available"
    KNOWLEDGE_TEMPORARILY_UNRENDERED = "knowledge_temporarily_unrendered"


@dataclass(frozen=True)
class GoalPattern:
    intent: IntentKind
    target_kinds: frozenset[TargetKind]

    def __post_init__(self) -> None:
        if not self.target_kinds:
            raise WorkflowRegistryError("goal pattern target set is empty")


@dataclass(frozen=True)
class WorkflowStage:
    tool_name: str
    argument_source: ArgumentSource


@dataclass(frozen=True)
class WorkflowRecipe:
    recipe_id: str
    goal_patterns: tuple[GoalPattern, ...]
    stages: tuple[WorkflowStage, ...]
    renderer: RendererKind
    trusted_skill_id: str | None = None
    require_same_target: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", self.recipe_id):
            raise WorkflowRegistryError("invalid recipe id")
        if not self.goal_patterns or not self.stages:
            raise WorkflowRegistryError("recipe requires patterns and stages")
        if (self.trusted_skill_id is not None and
                not re.fullmatch(
                    r"[a-z][a-z0-9_]{2,63}", self.trusted_skill_id)):
            raise WorkflowRegistryError("invalid trusted skill id")


_TOOL_SOURCES = {
    "predict_sample": {ArgumentSource.BOUND_SAMPLE},
    "get_prediction": {
        ArgumentSource.BOUND_PREDICTION,
        ArgumentSource.LATEST_FROM_HISTORY,
        ArgumentSource.PREVIOUS_FROM_HISTORY,
    },
    "get_explanation": {
        ArgumentSource.BOUND_PREDICTION,
        ArgumentSource.LATEST_FROM_HISTORY,
        ArgumentSource.PREVIOUS_FROM_HISTORY,
    },
    "get_prediction_history": {
        ArgumentSource.SESSION_HISTORY,
        ArgumentSource.LATEST_TWO_HISTORY,
    },
    "compare_predictions": {
        ArgumentSource.PAIR_FROM_HISTORY,
        ArgumentSource.BOUND_EXPLICIT_PAIR,
    },
    "search_medical_knowledge": {ArgumentSource.KNOWLEDGE_QUERY},
    "activate_skill": {ArgumentSource.TRUSTED_SKILL_ID},
}


class WorkflowRegistry:
    def __init__(self, recipes: tuple[WorkflowRecipe, ...]):
        self._recipes = recipes
        self._validate()
        self._version = self._calculate_version()

    @property
    def recipes(self) -> tuple[WorkflowRecipe, ...]:
        return self._recipes

    @property
    def version(self) -> str:
        return self._version

    def _validate(self) -> None:
        ids = [item.recipe_id for item in self._recipes]
        if len(ids) != len(set(ids)):
            raise WorkflowRegistryError("duplicate recipe id")
        expanded: dict[tuple[tuple[str, str], ...], str] = {}
        for recipe in self._recipes:
            for stage in recipe.stages:
                if stage.tool_name not in _TOOL_SOURCES:
                    raise WorkflowRegistryError("unknown tool in recipe")
                if stage.argument_source not in _TOOL_SOURCES[stage.tool_name]:
                    raise WorkflowRegistryError(
                        "argument source is incompatible with tool")
            if ((recipe.trusted_skill_id is None) !=
                    all(stage.tool_name != "activate_skill"
                        for stage in recipe.stages)):
                raise WorkflowRegistryError(
                    "trusted skill metadata does not match recipe stages")
            for key in self._expanded_keys(recipe):
                if key in expanded:
                    raise WorkflowRegistryError("overlapping recipe patterns")
                expanded[key] = recipe.recipe_id

    @staticmethod
    def _expanded_keys(
            recipe: WorkflowRecipe) -> set[tuple[tuple[str, str], ...]]:
        result: set[tuple[tuple[str, str], ...]] = set()
        choices = [pattern.target_kinds for pattern in recipe.goal_patterns]
        for targets in itertools.product(*choices):
            if recipe.require_same_target and len(set(targets)) != 1:
                continue
            key = tuple(sorted(
                (pattern.intent.value, target.value)
                for pattern, target in zip(recipe.goal_patterns, targets)))
            result.add(key)
        return result

    def _calculate_version(self) -> str:
        payload = [{
            "recipe_id": item.recipe_id,
            "patterns": [{
                "intent": pattern.intent.value,
                "targets": sorted(value.value for value in pattern.target_kinds),
            } for pattern in item.goal_patterns],
            "stages": [{
                "tool": stage.tool_name,
                "source": stage.argument_source.value,
            } for stage in item.stages],
            "renderer": item.renderer.value,
            "trusted_skill_id": item.trusted_skill_id,
            "require_same_target": item.require_same_target,
        } for item in self._recipes]
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def match(self, intent: ValidatedIntent) -> WorkflowRecipe | None:
        key = tuple(sorted(
            (goal.intent.value, goal.target.kind.value)
            for goal in intent.goals))
        matches = [recipe for recipe in self._recipes
                   if key in self._expanded_keys(recipe)]
        if len(matches) > 1:
            raise WorkflowRegistryError("multiple recipes matched one intent")
        return matches[0] if matches else None


def _pattern(intent: IntentKind, *targets: TargetKind) -> GoalPattern:
    return GoalPattern(intent, frozenset(targets))


def _stage(tool: str, source: ArgumentSource) -> WorkflowStage:
    return WorkflowStage(tool, source)


@lru_cache(maxsize=1)
def default_workflow_registry() -> WorkflowRegistry:
    current_or_explicit = (
        TargetKind.CURRENT_PREDICTION,
        TargetKind.EXPLICIT_PREDICTION,
    )
    return WorkflowRegistry((
        WorkflowRecipe(
            "predict_demo_sample",
            (_pattern(IntentKind.PREDICTION, TargetKind.DEMO_SAMPLE),),
            (_stage("predict_sample", ArgumentSource.BOUND_SAMPLE),),
            RendererKind.PREDICTION_COMPLETED),
        WorkflowRecipe(
            "read_current_or_explicit_prediction",
            (_pattern(IntentKind.SUMMARY, *current_or_explicit),),
            (_stage("get_prediction", ArgumentSource.BOUND_PREDICTION),),
            RendererKind.PREDICTION_DATA_AVAILABLE),
        WorkflowRecipe(
            "explain_current_or_explicit_prediction",
            (_pattern(IntentKind.EXPLANATION, *current_or_explicit),),
            (_stage("get_explanation", ArgumentSource.BOUND_PREDICTION),),
            RendererKind.EXPLANATION_DATA_AVAILABLE),
        WorkflowRecipe(
            "explain_previous_prediction",
            (_pattern(IntentKind.EXPLANATION,
                      TargetKind.PREVIOUS_PREDICTION),),
            (
                _stage("get_prediction_history",
                       ArgumentSource.LATEST_TWO_HISTORY),
                _stage("get_explanation",
                       ArgumentSource.PREVIOUS_FROM_HISTORY),
            ),
            RendererKind.EXPLANATION_DATA_AVAILABLE),
        WorkflowRecipe(
            "list_session_history",
            (_pattern(IntentKind.HISTORY, TargetKind.SESSION_HISTORY),),
            (_stage("get_prediction_history",
                    ArgumentSource.SESSION_HISTORY),),
            RendererKind.HISTORY_DATA_AVAILABLE),
        WorkflowRecipe(
            "compare_latest_two",
            (_pattern(IntentKind.COMPARISON,
                      TargetKind.LATEST_TWO_PREDICTIONS),),
            (
                _stage("get_prediction_history",
                       ArgumentSource.LATEST_TWO_HISTORY),
                _stage("compare_predictions", ArgumentSource.PAIR_FROM_HISTORY),
            ),
            RendererKind.COMPARISON_DATA_AVAILABLE),
        WorkflowRecipe(
            "compare_explicit_pair",
            (_pattern(IntentKind.COMPARISON,
                      TargetKind.EXPLICIT_PREDICTION_PAIR),),
            (_stage("compare_predictions",
                    ArgumentSource.BOUND_EXPLICIT_PAIR),),
            RendererKind.COMPARISON_DATA_AVAILABLE),
        WorkflowRecipe(
            "search_general_knowledge",
            (_pattern(IntentKind.KNOWLEDGE,
                      TargetKind.GENERAL_KNOWLEDGE),),
            (_stage("search_medical_knowledge",
                    ArgumentSource.KNOWLEDGE_QUERY),),
            RendererKind.KNOWLEDGE_TEMPORARILY_UNRENDERED),
        WorkflowRecipe(
            "activate_explanation_skill",
            (_pattern(IntentKind.SKILL, *current_or_explicit),),
            (
                _stage("activate_skill", ArgumentSource.TRUSTED_SKILL_ID),
                _stage("get_prediction", ArgumentSource.BOUND_PREDICTION),
                _stage("get_explanation", ArgumentSource.BOUND_PREDICTION),
                _stage("search_medical_knowledge",
                       ArgumentSource.KNOWLEDGE_QUERY),
            ),
            RendererKind.EXPLANATION_DATA_AVAILABLE,
            trusted_skill_id="explain_prediction"),
        WorkflowRecipe(
            "activate_comparison_skill",
            (_pattern(
                IntentKind.SKILL,
                TargetKind.LATEST_TWO_PREDICTIONS),),
            (
                _stage("activate_skill", ArgumentSource.TRUSTED_SKILL_ID),
                _stage("get_prediction_history",
                       ArgumentSource.LATEST_TWO_HISTORY),
                _stage("compare_predictions", ArgumentSource.PAIR_FROM_HISTORY),
            ),
            RendererKind.COMPARISON_DATA_AVAILABLE,
            trusted_skill_id="compare_prediction_history"),
        WorkflowRecipe(
            "activate_education_skill",
            (_pattern(IntentKind.SKILL, TargetKind.GENERAL_KNOWLEDGE),),
            (
                _stage("activate_skill", ArgumentSource.TRUSTED_SKILL_ID),
                _stage("search_medical_knowledge",
                       ArgumentSource.KNOWLEDGE_QUERY),
            ),
            RendererKind.KNOWLEDGE_TEMPORARILY_UNRENDERED,
            trusted_skill_id="pph_evidence_education"),
        WorkflowRecipe(
            "compare_and_explain_latest_two",
            (
                _pattern(IntentKind.COMPARISON,
                         TargetKind.LATEST_TWO_PREDICTIONS),
                _pattern(IntentKind.EXPLANATION,
                         TargetKind.LATEST_TWO_PREDICTIONS),
            ),
            (
                _stage("get_prediction_history",
                       ArgumentSource.LATEST_TWO_HISTORY),
                _stage("compare_predictions", ArgumentSource.PAIR_FROM_HISTORY),
                _stage("get_explanation",
                       ArgumentSource.LATEST_FROM_HISTORY),
                _stage("get_explanation",
                       ArgumentSource.PREVIOUS_FROM_HISTORY),
            ),
            RendererKind.COMPARISON_DATA_AVAILABLE,
            require_same_target=True),
    ))
