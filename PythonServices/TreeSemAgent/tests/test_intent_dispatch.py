from __future__ import annotations

import unittest

from agent.intent_dispatch import DispatchKind, IntentDispatcher
from agent.intent_frame import IntentFrame
from agent.intent_validation import validate_and_bind_intent
from agent.reference_extractor import extract_references
from agent.schemas import AgentRunRequest, PredictionContext


def request(message: str, *, current=True):
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message=message,
        current_prediction=(PredictionContext(
            prediction_id="pred_" + "c" * 32,
            model_version="v1") if current else None),
    )


def goal(intent, target, evidence, aspects=None, **indexes):
    return {
        "intent": intent,
        "target": {
            "type": target,
            "explicit_reference_index": indexes.get("explicit"),
            "second_explicit_reference_index": indexes.get("second"),
            "sample_reference_index": indexes.get("sample"),
        },
        "requested_aspects": aspects or [],
        "knowledge_scope": (
            "clinical" if intent in {"knowledge", "skill"} and
            target == "general_knowledge" else None),
        "evidence": evidence,
    }


def validation(user_request, goals, *, unresolved=None, clarify=False):
    value = IntentFrame.model_validate({
        "schema_version": 1,
        "goals": goals,
        "constraints": {"excluded_intents": [], "excluded_aspects": []},
        "unresolved_references": unresolved or [],
        "needs_clarification": clarify,
    })
    references = extract_references(user_request.message)
    return validate_and_bind_intent(value, references, user_request)


class IntentDispatcherTest(unittest.TestCase):
    def setUp(self):
        self.dispatcher = IntentDispatcher()

    def test_maps_one_complete_goal_to_a_stable_recipe(self):
        user_request = request("解释当前结果的决策路径")
        validated = validation(user_request, [goal(
            "explanation", "current_prediction", ["解释", "决策路径"],
            ["decision_path"])])

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(result.kind, DispatchKind.WORKFLOW)
        self.assertEqual(
            result.recipe.recipe_id,
            "explain_current_or_explicit_prediction")
        self.assertEqual(result.allowed_tools, frozenset({"get_explanation"}))

    def test_maps_registered_compare_and_explain_composite(self):
        user_request = request("比较最近两次并分别解释决策路径")
        validated = validation(user_request, [
            goal("comparison", "latest_two_predictions", ["比较最近两次"],
                 ["comparison_changes"]),
            goal("explanation", "latest_two_predictions", ["分别解释决策路径"],
                 ["decision_path"]),
        ])

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(result.kind, DispatchKind.COMPOSITE_WORKFLOW)
        self.assertEqual(
            result.recipe.recipe_id, "compare_and_explain_latest_two")
        self.assertEqual(result.allowed_tools, frozenset({
            "get_prediction_history", "compare_predictions",
            "get_explanation",
        }))

    def test_preserves_clarification_as_a_distinct_branch(self):
        user_request = request("解释那个", current=False)
        validated = validation(
            user_request,
            [goal("explanation", "none", ["那个"])],
            unresolved=["ambiguous_reference"], clarify=True)

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(result.kind, DispatchKind.CLARIFICATION)
        self.assertEqual(result.clarification_code, "ambiguous_reference")
        self.assertEqual(result.allowed_tools, frozenset())

    def test_other_request_enters_open_agent_without_side_effect_tools(self):
        user_request = request("帮我组织一下表达")
        validated = validation(user_request, [
            goal("other", "none", ["组织一下"]),
        ])

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(result.kind, DispatchKind.OPEN_AGENT)
        self.assertEqual(result.allowed_tools, frozenset())

    def test_unregistered_valid_composition_enters_narrow_open_agent(self):
        user_request = request("列出历史并解释当前决策路径")
        validated = validation(user_request, [
            goal("history", "session_history", ["列出历史"],
                 ["history_items"]),
            goal("explanation", "current_prediction", ["解释当前决策路径"],
                 ["decision_path"]),
        ])

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(result.kind, DispatchKind.OPEN_AGENT)
        self.assertEqual(result.allowed_tools, frozenset({
            "get_prediction_history", "get_explanation"}))
        self.assertNotIn("predict_sample", result.allowed_tools)

    def test_skill_target_selects_a_trusted_recipe_not_a_model_skill_id(self):
        user_request = request("使用稳定流程解释当前结果")
        validated = validation(user_request, [goal(
            "skill", "current_prediction", ["稳定流程", "解释"],
            ["decision_path"])])

        result = self.dispatcher.dispatch(validated)

        self.assertEqual(
            result.recipe.recipe_id, "activate_explanation_skill")
        self.assertEqual(result.recipe.trusted_skill_id, "explain_prediction")


if __name__ == "__main__":
    unittest.main()
