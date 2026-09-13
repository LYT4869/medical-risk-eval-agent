"""Request-local execution facts; never generated or updated by the Router."""
from __future__ import annotations

from dataclasses import dataclass, field

from .intent_frame import IntentKind, TargetKind
from .intent_validation import BoundGoal, ValidatedIntent
from .schemas import ToolUse
from .tool_result_projection import project_tool_result


_SAFE_TOOL_NAMES = frozenset({
    "predict_sample", "get_prediction", "get_explanation",
    "get_prediction_history", "compare_predictions",
    "search_medical_knowledge", "activate_skill",
})


@dataclass
class ExecutionState:
    intent: ValidatedIntent
    max_steps: int
    max_tool_calls: int
    llm_calls: int = 0
    llm_step_kinds: list[str] = field(default_factory=list)
    tool_attempts: int = 0
    tool_usages: list[ToolUse] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    _facts: list[tuple[str, dict, dict, str]] = field(default_factory=list)

    def record(self, name: str, arguments: dict, content: dict,
               usage: ToolUse, evidence: dict) -> None:
        safe_name = name if name in _SAFE_TOOL_NAMES else "unknown_tool"
        self.tool_usages.append(usage.model_copy(update={"name": safe_name}))
        self.evidence.append({"tool": safe_name, "data": evidence})
        self._facts.append((name, arguments, content, usage.status))

    @property
    def history_head_prediction_ids(self) -> tuple[str, ...]:
        history = next((content for name, args, content, status in reversed(
            self._facts) if name == "get_prediction_history" and
            status == "success" and args.get("cursor") is None), {})
        return tuple(item["prediction_id"] for item in history.get("items", [])
                    if isinstance(item, dict) and
                    isinstance(item.get("prediction_id"), str))

    def _target_ids(self, goal: BoundGoal) -> tuple[str, ...]:
        if goal.target.prediction_ids:
            return goal.target.prediction_ids
        ids = self.history_head_prediction_ids
        if goal.target.kind == TargetKind.PREVIOUS_PREDICTION:
            return ids[1:2]
        if goal.target.kind == TargetKind.LATEST_TWO_PREDICTIONS:
            return ids[:2] if len(ids) >= 2 else ()
        return ()

    def goal_completed(self, goal: BoundGoal) -> bool:
        successful = [(name, args, content) for name, args, content, status
                      in self._facts if status == "success"]
        if goal.intent == IntentKind.PREDICTION:
            return any(name == "predict_sample" and
                       goal.target.sample_index is not None and
                       args.get("sample_index") == goal.target.sample_index and
                       isinstance(content.get("prediction_id"), str)
                       for name, args, content in successful)
        if goal.intent == IntentKind.HISTORY:
            return any(name == "get_prediction_history"
                       for name, _, _ in successful)
        if goal.intent == IntentKind.KNOWLEDGE:
            return any(name == "search_medical_knowledge" and
                       (goal.knowledge_scope is None or
                        args.get("scope", "all") == goal.knowledge_scope.value)
                       for name, args, _ in successful)
        ids = self._target_ids(goal)
        if goal.intent == IntentKind.COMPARISON:
            return bool(len(ids) == 2 and any(
                name == "compare_predictions" and
                {content.get("prediction_a", {}).get("prediction_id"),
                 content.get("prediction_b", {}).get("prediction_id")} == set(ids)
                for name, _, content in successful))
        tool = {IntentKind.SUMMARY: "get_prediction",
                IntentKind.EXPLANATION: "get_explanation"}.get(goal.intent)
        if tool is not None:
            returned = {content.get("prediction_id") for name, _, content
                        in successful if name == tool}
            return bool(ids and set(ids) <= returned)
        return False

    @property
    def completed_goal_indexes(self) -> list[int]:
        return [index for index, goal in enumerate(self.intent.goals)
                if self.goal_completed(goal)]

    @property
    def all_goals_completed(self) -> bool:
        return len(self.completed_goal_indexes) == len(self.intent.goals)

    def summary(self, reason: str) -> dict:
        completed = self.completed_goal_indexes
        return {
            "termination_reason": reason,
            "llm_call_count": self.llm_calls,
            "llm_step_kinds": list(self.llm_step_kinds),
            "tool_calls": [{"name": usage.name, "status": usage.status}
                           for usage in self.tool_usages],
            "completed_goal_indexes": completed,
            "pending_goal_indexes": [i for i in range(len(self.intent.goals))
                                     if i not in completed],
            "remaining_steps": max(0, self.max_steps - self.llm_calls),
            "remaining_tool_calls": max(
                0, self.max_tool_calls - self.tool_attempts),
        }

    def finalization_context(self, original_request: str) -> dict:
        # A later head-history read may bind relative targets after comparison.
        # Refresh from raw facts, never reverse an already projected delta twice.
        evidence = [
            {"tool": entry["tool"], "data": project_tool_result(
                name, content, self.intent,
                history_head_prediction_ids=self.history_head_prediction_ids)}
            if name == "compare_predictions" and status == "success" else entry
            for entry, (name, _, content, status) in zip(self.evidence, self._facts)
        ]
        return finalization_context(self.intent, original_request,
                                    self.completed_goal_indexes, evidence)


def finalization_context(intent: ValidatedIntent, original_request: str,
                         completed: list[int], evidence: list[dict]) -> dict:
    return {
            "original_request": original_request,
            "subgoals": [{
                "intent": goal.intent.value,
                "target": goal.target.kind.value,
                "requested_aspects": [a.value for a in goal.requested_aspects],
                "evidence_fields_to_cover": _evidence_fields(goal, intent),
                "status": "completed" if i in completed else "pending",
            } for i, goal in enumerate(intent.goals)],
            "response_constraints": {
                "excluded_intents": sorted(i.value for i in
                                           intent.excluded_intents),
                "excluded_aspects": sorted(a.value for a in
                                           intent.excluded_aspects),
                "preserve_original_request_constraints": True,
            },
            "current_evidence": list(evidence),
        }


def _evidence_fields(goal: BoundGoal, intent: ValidatedIntent) -> list[str]:
    if goal.intent == IntentKind.COMPARISON:
        fields = ["positive_probability_delta", "label_changed",
                  "model_version_changed", "path_changed"]
    elif goal.requested_aspects:
        fields = [aspect.value for aspect in goal.requested_aspects]
    else:
        fields = {
            IntentKind.SUMMARY: ["prediction_summary"],
            IntentKind.EXPLANATION: ["important_features", "decision_path"],
            IntentKind.HISTORY: ["items"],
            IntentKind.KNOWLEDGE: ["supported_claims", "citation_id"],
        }.get(goal.intent, [])
    excluded = {aspect.value for aspect in intent.excluded_aspects}
    alias = {"positive_probability_delta": "probability",
             "label_changed": "label", "model_version_changed": "model_version",
             "path_changed": "decision_path"}
    return [field for field in fields
            if field not in excluded and alias.get(field, field) not in excluded]
