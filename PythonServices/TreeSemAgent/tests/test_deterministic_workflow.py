from __future__ import annotations

import asyncio
import unittest

from agent.deterministic_workflow import DeterministicWorkflowExecutor
from agent.intent_dispatch import IntentDispatcher
from agent.intent_frame import IntentFrame
from agent.intent_validation import validate_and_bind_intent
from agent.reference_extractor import extract_references
from agent.schemas import AgentRunRequest, PredictionContext
from agent.tool_registry import ToolRegistry
from agent.tools import ToolContext


class FakeBackend:
    def __init__(self, *, history_count=2, fail_prediction=False):
        self.calls = []
        self.history_count = history_count
        self.fail_prediction = fail_prediction

    async def predict_sample(self, context, sample_index):
        self.calls.append(("predict_sample", sample_index))
        if self.fail_prediction:
            raise ValueError("injected failure")
        return {"prediction_id": "pred_" + "d" * 32}

    async def get_prediction(self, context, prediction_id):
        self.calls.append(("get_prediction", prediction_id))
        return {"prediction_id": prediction_id}

    async def get_explanation(self, context, prediction_id):
        self.calls.append(("get_explanation", prediction_id))
        return {
            "prediction_id": prediction_id,
            "important_features": [],
            "decision_path": [],
        }

    async def get_history(self, context, limit, cursor):
        self.calls.append(("get_prediction_history", limit, cursor))
        ids = ["pred_" + value * 32 for value in ("a", "b")]
        return {
            "items": [{"prediction_id": value}
                      for value in ids[:self.history_count]],
            "next_cursor": None,
        }

    async def compare(self, context, prediction_id_a, prediction_id_b):
        self.calls.append((
            "compare_predictions", prediction_id_a, prediction_id_b))
        return {
            "prediction_a": {"prediction_id": prediction_id_a},
            "prediction_b": {"prediction_id": prediction_id_b},
            "label_changed": False,
            "model_version_changed": False,
            "positive_probability_delta": 0.0,
            "confidence_delta": 0.0,
            "cluster_changed": False,
            "tree_leaf_changed": False,
            "path_changed": False,
            "changed_features": [],
        }


class FakeKnowledge:
    def __init__(self):
        self.calls = []

    async def search(self, token, query, scope, top_k, trace=None):
        self.calls.append((token, query, scope, top_k))
        return {
            "index_version": "knowledge-v1",
            "retrieval_mode": "hybrid",
            "results": [{
                "citation_id": "cite_" + "a" * 20,
                "source_id": "src_who",
                "title": "WHO guide",
                "section": "Overview",
                "page": 1,
                "excerpt": "Evidence",
                "publisher": "WHO",
                "published_at": "2025-01-01",
                "url": "https://www.who.int/example",
                "content_sha256": "b" * 64,
                "score": 0.9,
            }],
        }


def request(message, *, current=True):
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message=message,
        current_prediction=(PredictionContext(
            prediction_id="pred_" + "c" * 32,
            model_version="v1") if current else None),
        capability_token="capability",
        knowledge_capability_token="knowledge-capability",
    )


def goal(intent, target, evidence, aspects=None, scope=None, **indexes):
    return {
        "intent": intent,
        "target": {
            "type": target,
            "explicit_reference_index": indexes.get("explicit"),
            "second_explicit_reference_index": indexes.get("second"),
            "sample_reference_index": indexes.get("sample"),
        },
        "requested_aspects": aspects or [],
        "knowledge_scope": scope,
        "evidence": evidence,
    }


def recipe_for(user_request, goals, *, requested_skill=None):
    value = IntentFrame.model_validate({
        "schema_version": 2,
        "goals": goals,
        "constraints": {"excluded_intents": [], "excluded_aspects": []},
        "unresolved_references": [],
        "needs_clarification": False,
        "requested_skill": requested_skill,
    })
    validated = validate_and_bind_intent(
        value, extract_references(user_request.message), user_request)
    dispatch = IntentDispatcher().dispatch(validated)
    return dispatch.recipe, dispatch.validated_intent


class DeterministicWorkflowTest(unittest.TestCase):
    @staticmethod
    def execute(user_request, goals, backend=None, knowledge=None):
        recipe, intent = recipe_for(user_request, goals)
        backend = backend or FakeBackend()
        tools = ToolRegistry(backend, knowledge)
        context = ToolContext(
            user_request.session_id, user_request.capability_token,
            user_request.knowledge_capability_token,
            user_request.actor_role)
        result = asyncio.run(DeterministicWorkflowExecutor(tools).execute(
            recipe, intent, context, user_request.message))
        return result, backend

    def test_predicts_a_bound_demo_sample_without_an_llm(self):
        user_request = request("预测样本 12")

        result, backend = self.execute(user_request, [
            goal("prediction", "demo_sample", ["预测", "<sample_ref_0>"],
                 sample=0),
        ])

        self.assertTrue(result.completed)
        self.assertEqual(backend.calls, [("predict_sample", 12)])
        self.assertEqual(
            result.prediction_ids, frozenset({"pred_" + "d" * 32}))

    def test_explains_the_trusted_current_prediction_directly(self):
        user_request = request("解释当前结果")

        result, backend = self.execute(user_request, [
            goal("explanation", "current_prediction", ["解释"]),
        ])

        self.assertTrue(result.completed)
        self.assertEqual(backend.calls, [
            ("get_explanation", user_request.current_prediction.prediction_id),
        ])

    def test_resolves_previous_prediction_from_history(self):
        user_request = request("解释上次结果")

        result, backend = self.execute(user_request, [
            goal("explanation", "previous_prediction", ["上次"]),
        ])

        self.assertTrue(result.completed)
        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None),
            ("get_explanation", "pred_" + "b" * 32),
        ])

    def test_compares_latest_two_ids_returned_by_history(self):
        user_request = request("比较最近两次")

        result, backend = self.execute(user_request, [
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
        ])

        self.assertTrue(result.completed)
        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None),
            ("compare_predictions", "pred_" + "a" * 32,
             "pred_" + "b" * 32),
        ])

    def test_compares_exact_explicit_source_ids_without_history(self):
        first = "pred_" + "e" * 32
        second = "pred_" + "f" * 32
        user_request = request(f"比较 {first} 和 {second}")

        result, backend = self.execute(user_request, [
            goal("comparison", "explicit_prediction_pair",
                 ["比较", "<prediction_ref_0>", "<prediction_ref_1>"],
                 explicit=0, second=1),
        ])

        self.assertTrue(result.completed)
        self.assertEqual(backend.calls, [
            ("compare_predictions", first, second),
        ])

    def test_knowledge_query_does_not_forward_business_ids(self):
        knowledge = FakeKnowledge()
        business_id = "pred_" + "e" * 32
        user_request = request(f"查询产后出血资料，不要使用 {business_id}")

        result, _ = self.execute(user_request, [
            goal("knowledge", "general_knowledge", ["查询", "资料"],
                 scope="clinical"),
        ], knowledge=knowledge)

        self.assertTrue(result.completed)
        self.assertEqual(len(knowledge.calls), 1)
        self.assertNotIn(business_id, knowledge.calls[0][1])
        self.assertEqual(knowledge.calls[0][2:], ("clinical", 5))
        self.assertEqual(result.knowledge_index_version, "knowledge-v1")

    def test_stops_when_history_has_fewer_than_two_predictions(self):
        backend = FakeBackend(history_count=1)
        user_request = request("比较最近两次")

        result, backend = self.execute(user_request, [
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
        ], backend=backend)

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_code, "insufficient_history")
        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None),
        ])

    def test_failed_stage_stops_the_recipe(self):
        backend = FakeBackend(fail_prediction=True)
        user_request = request("预测样本 1")

        result, backend = self.execute(user_request, [
            goal("prediction", "demo_sample", ["预测", "<sample_ref_0>"],
                 sample=0),
        ], backend=backend)

        self.assertFalse(result.completed)
        self.assertEqual(result.failure_code, "tool_execution_failed")
        self.assertEqual(len(backend.calls), 1)

    def test_tool_definition_names_respect_runtime_availability(self):
        context = ToolContext("ses_" + "2" * 32)
        names = ToolRegistry(FakeBackend()).definition_names(context)

        self.assertIn("predict_sample", names)
        self.assertNotIn("search_medical_knowledge", names)
        self.assertNotIn("activate_skill", names)


if __name__ == "__main__":
    unittest.main()
