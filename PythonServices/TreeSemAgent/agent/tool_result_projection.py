from __future__ import annotations

from typing import Any

from .intent_frame import IntentKind, RequestedAspect
from .intent_validation import ValidatedIntent


_EXPLANATION_FIELDS = {
    RequestedAspect.IMPORTANT_FEATURES: "important_features",
    RequestedAspect.DECISION_PATH: "decision_path",
}
_EXPLANATION_IDENTITY_FIELDS = ("prediction_id", "model_version")


def project_tool_result(
        tool_name: str, content: dict[str, Any],
        intent: ValidatedIntent) -> dict[str, Any]:
    """Return the least explanation data needed by the final LLM.

    The typed Tool result remains unchanged for binding and grounding. The
    projection is only the copy placed in the model's conversational context.
    """
    if tool_name != "get_explanation" or "error" in content:
        return dict(content)

    goals = tuple(
        goal for goal in intent.goals
        if goal.intent == IntentKind.EXPLANATION)
    if not goals:
        return dict(content)

    requested = {
        aspect for goal in goals for aspect in goal.requested_aspects
        if aspect in _EXPLANATION_FIELDS
    }
    excluded = intent.excluded_aspects.intersection(_EXPLANATION_FIELDS)
    if not requested and not excluded:
        return dict(content)

    visible = requested or set(_EXPLANATION_FIELDS)
    visible.difference_update(excluded)
    fields = set(_EXPLANATION_IDENTITY_FIELDS)
    fields.update(_EXPLANATION_FIELDS[aspect] for aspect in visible)
    return {key: value for key, value in content.items() if key in fields}
