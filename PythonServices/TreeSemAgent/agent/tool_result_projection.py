from __future__ import annotations

from typing import Any

from .intent_frame import IntentKind, RequestedAspect, TargetKind
from .intent_validation import ValidatedIntent


_EXPLANATION_FIELDS = {
    RequestedAspect.IMPORTANT_FEATURES: "important_features",
    RequestedAspect.DECISION_PATH: "decision_path",
}
_EXPLANATION_IDENTITY_FIELDS = ("prediction_id", "model_version")


def project_tool_result(
        tool_name: str, content: dict[str, Any],
        intent: ValidatedIntent, *,
        history_head_prediction_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    """Return the least explanation data needed by the final LLM.

    The typed Tool result remains unchanged for binding and grounding. The
    projection is only the copy placed in the model's conversational context.
    """
    if "error" in content:
        return dict(content)
    if tool_name == "compare_predictions":
        return _project_comparison(content, intent, history_head_prediction_ids)
    if tool_name != "get_explanation":
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


def _direction(delta: float) -> str:
    return "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"


def _metric_change(metric: str, before: float, after: float,
                   delta: float) -> dict[str, Any]:
    """Describe a [0, 1] model score without turning it into prose."""
    display = lambda value: round(100 * value, 10)
    return {
        "metric": metric,
        "from": {"value": before, "display_value": display(before),
                 "display_unit": "percent"},
        "to": {"value": after, "display_value": display(after),
               "display_unit": "percent"},
        "absolute_delta": {"value": abs(delta),
                           "display_value": display(abs(delta)),
                           "display_unit": "percentage_points"},
        "direction": _direction(delta),
        # Relative change has a different denominator and is deliberately not
        # inferred when the business comparison API did not request it.
        "relative_change": "not_computed",
    }


def _model_class(code: Any) -> dict[str, Any]:
    name = {0: "model_negative_class", 1: "model_positive_class"}.get(
        code, "model_class_" + str(code))
    return {"code": code, "name": name}


def _project_comparison(content: dict[str, Any], intent: ValidatedIntent,
                        head: tuple[str, ...]) -> dict[str, Any]:
    """Normalize only the LLM evidence copy, never the backend business result."""
    a, b = content["prediction_a"], content["prediction_b"]
    pair = (a["prediction_id"], b["prediction_id"])
    goals = [goal for goal in intent.goals if goal.intent == IntentKind.COMPARISON]
    relative = bool(goals) and all(
        goal.target.kind == TargetKind.LATEST_TWO_PREDICTIONS for goal in goals)
    chronological = (relative and len(head) >= 2 and head[0] != head[1]
                     and set(pair) == set(head[:2]))
    reverse = chronological and pair[0] == head[0]
    sign = -1 if reverse else 1
    result = {key: value for key, value in content.items()
              if key not in {"prediction_a", "prediction_b", "changed_features"}}
    result.update({
        "comparison_order": "previous_to_latest" if chronological else "requested_a_to_b",
        "from_prediction": dict(b if reverse else a),
        "to_prediction": dict(a if reverse else b),
        "delta_definition": "to_prediction minus from_prediction",
    })
    for field in ("positive_probability_delta", "confidence_delta"):
        delta = sign * content[field]
        result[field] = 0.0 if delta == 0 else delta
        result[field.removesuffix("_delta") + "_direction"] = _direction(delta)
    result["positive_probability_delta_percentage_points"] = (
        100 * result["positive_probability_delta"])
    for field, metric in (
            ("positive_probability", "positive_class_probability"),
            ("confidence", "predicted_class_confidence")):
        if (result["from_prediction"].get(field) is not None and
                result["to_prediction"].get(field) is not None):
            result[field + "_change"] = _metric_change(
                metric, result["from_prediction"][field],
                result["to_prediction"][field], result[field + "_delta"])
    from_label = result["from_prediction"].get("label")
    to_label = result["to_prediction"].get("label")
    if from_label is not None and to_label is not None:
        result["model_class_change"] = {
            "changed": from_label != to_label,
            "from": _model_class(from_label),
            "to": _model_class(to_label),
        }
    result["interpretation_scope"] = {
        "probability": "model_positive_class_probability",
        "class_labels": "model_encoding_only",
        "clinical_interpretation_supported": False,
        "feature_threshold_is_clinical_reference_range": False,
        "causal_attribution_supported": False,
    }

    features = []
    for original in content.get("changed_features", []):
        feature = dict(original)
        for prefix in ("original", "standardized"):
            source, destination = ("b", "a") if reverse else ("a", "b")
            for suffix, key in (("from", source), ("to", destination)):
                old_key = prefix + "_value_" + key
                if old_key in original:
                    feature[prefix + "_value_" + suffix] = original[old_key]
            feature.pop(prefix + "_value_a", None)
            feature.pop(prefix + "_value_b", None)
            delta_key = prefix + "_delta"
            if original.get(delta_key) is not None:
                feature[delta_key] = sign * original[delta_key]
        features.append(feature)
    result["changed_features"] = features
    return result
