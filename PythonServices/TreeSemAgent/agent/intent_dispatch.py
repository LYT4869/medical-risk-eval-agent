from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .intent_frame import IntentKind
from .intent_validation import IntentValidationResult, ValidatedIntent
from .workflow_registry import (
    WorkflowRecipe,
    WorkflowRegistry,
    default_workflow_registry,
)


class DispatchKind(str, Enum):
    WORKFLOW = "workflow"
    COMPOSITE_WORKFLOW = "composite_workflow"
    CLARIFICATION = "clarification"
    OPEN_AGENT = "open_agent"


@dataclass(frozen=True)
class DispatchDecision:
    kind: DispatchKind
    recipe: WorkflowRecipe | None
    validated_intent: ValidatedIntent | None
    clarification_code: str | None
    allowed_tools: frozenset[str]


_OPEN_TOOLS = {
    IntentKind.PREDICTION: frozenset(),
    IntentKind.SUMMARY: frozenset({"get_prediction"}),
    IntentKind.EXPLANATION: frozenset({
        "get_prediction_history", "get_explanation"}),
    IntentKind.HISTORY: frozenset({"get_prediction_history"}),
    IntentKind.COMPARISON: frozenset({
        "get_prediction_history", "compare_predictions"}),
    IntentKind.KNOWLEDGE: frozenset({"search_medical_knowledge"}),
    IntentKind.SKILL: frozenset({"activate_skill"}),
    IntentKind.OTHER: frozenset(),
}


class IntentDispatcher:
    def __init__(self, registry: WorkflowRegistry | None = None):
        self._registry = registry or default_workflow_registry()

    def dispatch(self, result: IntentValidationResult) -> DispatchDecision:
        if result.validated is None:
            return DispatchDecision(
                DispatchKind.CLARIFICATION, None, None,
                result.clarification_code, frozenset())
        recipe = self._registry.match(result.validated)
        if recipe is not None:
            tools = frozenset(stage.tool_name for stage in recipe.stages)
            kind = (DispatchKind.COMPOSITE_WORKFLOW
                    if len(result.validated.goals) > 1
                    else DispatchKind.WORKFLOW)
            return DispatchDecision(
                kind, recipe, result.validated, None, tools)
        allowed: set[str] = set()
        for goal in result.validated.goals:
            allowed.update(_OPEN_TOOLS[goal.intent])
        allowed.discard("predict_sample")
        return DispatchDecision(
            DispatchKind.OPEN_AGENT, None, result.validated, None,
            frozenset(allowed))
