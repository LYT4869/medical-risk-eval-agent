"""Shared target binding and bounded semantic-contract repair regressions."""
from __future__ import annotations

import asyncio
import json
import unittest

from agent.intent_dispatch import IntentDispatcher
from agent.execution_state import finalization_context
from agent.intent_validation import validate_and_bind_intent
from agent.loop import AgentLoop, AgentExecutionError
from agent.reference_extractor import extract_references
from agent.schemas import LlmTurn
from agent.structured_router import StructuredIntentRouter, StructuredRouterConfig
from agent.tool_registry import ToolRegistry
from test_structured_agent_loop import (
    CURRENT, FakeBackend, FakeStructuredRouter, RecordingLlm, PREVIOUS, frame, goal, request,
)
from test_structured_router import RecordingLlm as RouterLlm, route_turn


class ContractConsistencyTest(unittest.TestCase):
    def test_previous_summary_binds_history_then_reads_previous_not_current(self):
        backend = FakeBackend()
        value = frame([goal("summary", "previous_prediction", ["上一次"], ["label", "probability"])])
        result = asyncio.run(AgentLoop(
            RecordingLlm([LlmTurn(content="上一次记录 " + PREVIOUS,
                                  grounding_prediction_ids=[PREVIOUS])]),
            ToolRegistry(backend), structured_router=FakeStructuredRouter(value),
            routing_mode="structured_llm").run(request("看上一次的标签和概率")))
        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None), ("get_prediction", PREVIOUS)])
        self.assertEqual(result.grounding_prediction_ids, [PREVIOUS])

    def test_previous_summary_composes_with_knowledge_without_open_planning(self):
        req = request("看上一次标签，再解释标签编码")
        knowledge = goal("knowledge", "general_knowledge", ["解释标签编码"], ["knowledge_overview"])
        knowledge["knowledge_scope"] = "model"
        bound = validate_and_bind_intent(frame([
            goal("summary", "previous_prediction", ["上一次标签"], ["label"]), knowledge]),
            extract_references(req.message), req)
        decision = IntentDispatcher().dispatch(bound)
        self.assertEqual(decision.kind.value, "composite_workflow")
        self.assertEqual([s.tool_name for s in decision.recipe.stages],
                         ["get_prediction_history", "get_prediction", "search_medical_knowledge"])

    @staticmethod
    def router(client, attempts=2):
        return StructuredIntentRouter(client, StructuredRouterConfig(
            maximum_attempts=attempts, request_timeout_seconds=1,
            total_deadline_seconds=3, retry_backoff_seconds=0))

    def test_cross_field_error_is_repaired_before_any_business_tool(self):
        req = request("比较两次，再解释概率差是否代表病情变化").model_copy(
            update={"capability_token": "private-test-capability-123",
                    "knowledge_capability_token": "private-test-knowledge-123"})
        bad = frame([
            goal("comparison", "latest_two_predictions", ["比较两次"], ["comparison_changes"]),
            goal("explanation", "current_prediction", ["概率差是否代表病情变化"], ["probability"])])
        knowledge = goal("knowledge", "general_knowledge", ["概率差是否代表病情变化"], ["knowledge_overview"])
        knowledge["knowledge_scope"] = "model"
        good = frame([bad.goals[0].model_dump(mode="json"), knowledge])
        client = RouterLlm([route_turn(bad.model_dump(mode="json")), route_turn(good.model_dump(mode="json"))])
        from test_structured_agent_loop import FakeKnowledge
        backend = FakeBackend()
        result = asyncio.run(AgentLoop(
            RecordingLlm([LlmTurn(content="模型差异不能直接代表病情变化 cite_" + "e" * 20,
                                  grounding_source_ids=["cite_" + "e" * 20],
                                  grounding_prediction_ids=["pred_" + "a" * 32, PREVIOUS])]),
            ToolRegistry(backend, knowledge=FakeKnowledge()),
            structured_router=self.router(client), routing_mode="structured_llm").run(req))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual([name for name, *_ in backend.calls],
                         ["get_prediction_history", "compare_predictions"])
        self.assertEqual(result.grounding_source_ids, ["cite_" + "e" * 20])
        feedback = client.requests[1]["messages"][-1]["content"]
        self.assertIn("incompatible_aspect", feedback)
        prompts = json.dumps([r["messages"] for r in client.requests])
        self.assertNotIn("private-test-capability-123", prompts)
        self.assertNotIn("private-test-knowledge-123", prompts)

    def test_repeated_cross_field_error_stops_without_open_agent_or_tools(self):
        bad = frame([goal("explanation", "current_prediction", ["解释"], ["probability"])])
        client = RouterLlm([route_turn(bad.model_dump(mode="json"))] * 3)
        backend = FakeBackend()
        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(AgentLoop(RecordingLlm([]), ToolRegistry(backend),
                structured_router=self.router(client), routing_mode="structured_llm")
                .run(request("解释概率含义")))
        self.assertEqual(caught.exception.code, "invalid_intent_frame")
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(backend.calls, [])

    def test_knowledge_search_completion_does_not_claim_answer_coverage(self):
        req = request("解释标签编码")
        knowledge = goal("knowledge", "general_knowledge", ["标签编码"])
        knowledge["knowledge_scope"] = "model"
        intent = validate_and_bind_intent(frame([knowledge]), extract_references(req.message), req).validated
        for data, expected in (
            ({"results": []}, "not_found"),
            ({"results": [{"excerpt": "AUC describes ranking only."}]}, "retrieved"),
            ({"error": "knowledge_unavailable"}, "unavailable"),
        ):
            with self.subTest(expected=expected):
                payload = finalization_context(intent, req.message, [0],
                    [{"tool": "search_medical_knowledge", "data": data}])
                self.assertEqual(payload["knowledge_evidence"]["availability"], expected)
                self.assertEqual(payload["knowledge_evidence"]["answer_coverage"], "not_verified")
                self.assertNotIn("evidence_availability", payload["subgoals"][0])

    def test_resolvable_explicit_missing_state_still_repairs_invalid_aspect(self):
        bad_goal = goal("explanation", "explicit_prediction", ["概率"], ["probability"])
        bad_goal["target"]["explicit_reference_index"] = 0
        bad = frame([bad_goal], unresolved=["missing_prediction_target"], clarify=True)
        good_goal = dict(bad_goal, intent="summary")
        good = frame([good_goal], unresolved=["missing_prediction_target"], clarify=True)
        client = RouterLlm([route_turn(bad.model_dump(mode="json")), route_turn(good.model_dump(mode="json"))])
        backend = FakeBackend()
        result = asyncio.run(AgentLoop(
            RecordingLlm([LlmTurn(content="已读取该记录", grounding_prediction_ids=[CURRENT])]),
            ToolRegistry(backend), structured_router=self.router(client), routing_mode="structured_llm")
            .run(request("看看这条预测概率 " + CURRENT)))
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(backend.calls, [("get_prediction", CURRENT)])
        self.assertEqual(result.grounding_prediction_ids, [CURRENT])

    def test_invalid_source_index_is_not_repaired_or_executed(self):
        invalid = goal("summary", "explicit_prediction", ["概率"], ["probability"])
        invalid["target"]["explicit_reference_index"] = 7
        value = frame([invalid])
        client = RouterLlm([route_turn(value.model_dump(mode="json"))] * 2)
        backend = FakeBackend()
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(RecordingLlm([]), ToolRegistry(backend),
                structured_router=self.router(client), routing_mode="structured_llm")
                .run(request("看看这条预测概率 " + CURRENT)))
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(backend.calls, [])

    def test_missing_previous_summary_does_not_claim_explanation_or_current_facts(self):
        class OneRecord(FakeBackend):
            async def get_history(self, context, limit, cursor):
                self.calls.append(("get_prediction_history", limit, cursor))
                return {"items": [{"prediction_id": CURRENT}], "next_cursor": None}
        backend = OneRecord()
        value = frame([goal("summary", "previous_prediction", ["上一次"], ["label"])])
        result = asyncio.run(AgentLoop(RecordingLlm([]), ToolRegistry(backend),
            structured_router=FakeStructuredRouter(value), routing_mode="structured_llm")
            .run(request("看上一次的标签")))
        self.assertIn("上一次", result.answer)
        self.assertNotIn("解释", result.answer)
        self.assertEqual(result.grounding_prediction_ids, [])
        self.assertEqual(backend.calls, [("get_prediction_history", 2, None)])


if __name__ == "__main__":
    unittest.main()
