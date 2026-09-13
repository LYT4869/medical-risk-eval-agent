from __future__ import annotations

from dataclasses import dataclass, replace
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
        if recipe is None:
            recipe = self._compose_business_knowledge(result.validated)
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

    def _compose_business_knowledge(
            self, intent: ValidatedIntent) -> WorkflowRecipe | None:
        """Compose two known read-only goals; never synthesize new capabilities."""
        if len(intent.goals) != 2 or intent.requested_skill is not None:
            return None
        knowledge = [goal for goal in intent.goals
                     if goal.intent == IntentKind.KNOWLEDGE]
        business = [goal for goal in intent.goals if goal.intent in {
            IntentKind.SUMMARY, IntentKind.EXPLANATION,
            IntentKind.HISTORY, IntentKind.COMPARISON}]
        if len(knowledge) != 1 or len(business) != 1:
            return None
        business_recipe = self._registry.match(replace(
            intent, goals=(business[0],)))
        knowledge_recipe = self._registry.match(replace(
            intent, goals=(knowledge[0],)))
        if business_recipe is None or knowledge_recipe is None:
            return None
        stages = business_recipe.stages + knowledge_recipe.stages
        if any(stage.tool_name in {"predict_sample", "activate_skill"}
               for stage in stages):
            return None
        return WorkflowRecipe(
            recipe_id="composed_business_knowledge",
            goal_patterns=(business_recipe.goal_patterns +
                           knowledge_recipe.goal_patterns),
            stages=stages, renderer=business_recipe.renderer)
