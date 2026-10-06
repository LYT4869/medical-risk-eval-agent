from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from agent.intent_frame import IntentFrame
from agent.llm_client import ScriptedLlmClient
from agent.loop import AgentLoop
from agent.observability import Metrics
from agent.schemas import (
    AgentRunRequest,
    LlmTurn,
    LlmUsage,
    PredictionContext,
)
from agent.structured_router import StructuredRoute, StructuredRouterError
from agent.tool_registry import ToolRegistry


PREDICTION_ID = "pred_" + "a" * 32
SESSION_ID = "ses_" + "b" * 32
REQUEST_ID = "req_" + "c" * 32


class ExplanationBackend:
    async def get_explanation(self, context, prediction_id):
        del context
        return {
            "prediction_id": prediction_id,
            "important_features": [],
            "decision_path": [],
        }


class MeasuredRouter:
    def __init__(self, *, attempts=1, repaired=False, failure=None):
        self.calls = 0
        self.failure = failure
        self.result = StructuredRoute(
            explanation_frame(),
            usage=LlmUsage(
                prompt_tokens=7,
                completion_tokens=3,
                total_tokens=10 * attempts),
            attempt_count=attempts,
            repaired=repaired,
        )

    async def route(self, context):
        del context
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        return self.result


def explanation_frame() -> IntentFrame:
    return IntentFrame.model_validate({
        "schema_version": 2,
        "goals": [{
            "intent": "explanation",
            "target": {
                "type": "current_prediction",
                "explicit_reference_index": None,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": ["decision_path"],
            "knowledge_scope": None,
            "evidence": ["private explanation words"],
        }],
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": [],
        },
        "unresolved_references": [],
        "needs_clarification": False,
        "requested_skill": None,
    })


def other_frame() -> IntentFrame:
    return IntentFrame.model_validate({
        "schema_version": 2,
        "goals": [{
            "intent": "other",
            "target": {
                "type": "none",
                "explicit_reference_index": None,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": [],
            "knowledge_scope": None,
            "evidence": ["你好"],
        }],
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": [],
        },
        "unresolved_references": [],
        "needs_clarification": False,
        "requested_skill": None,
    })


def request(message="private explanation words") -> AgentRunRequest:
    return AgentRunRequest(
        run_id="run_" + "d" * 32,
        session_id=SESSION_ID,
        request_id=REQUEST_ID,
        message=message,
        current_prediction=PredictionContext(
            prediction_id=PREDICTION_ID, model_version="private-model"),
    )


def final_turn(total_tokens=30) -> LlmTurn:
    return LlmTurn(
        final_response_is_structured=True,
        content="safe answer",
        grounding_prediction_ids=[PREDICTION_ID],
        usage=LlmUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=total_tokens),
    )


class StructuredRoutingObservabilityTest(unittest.TestCase):
    @staticmethod
    def execute(router, turns, *, user_request=None):
        registry = Metrics()
        events = []
        llm = ScriptedLlmClient(turns)
        loop = AgentLoop(
            llm, ToolRegistry(ExplanationBackend()),
            structured_router=router,
            routing_mode="structured_llm")
        with patch("agent.loop.metrics", registry), patch(
                "agent.loop.trace_event",
                side_effect=lambda trace, operation, started, outcome,
                **fields: events.append({
                    "operation": operation,
                    "outcome": outcome,
                    **fields,
                })):
            result = asyncio.run(loop.run(user_request or request()))
        return result, registry.render(), events, llm

    def test_one_attempt_workflow_accounts_for_router_and_final_llm(self):
        _, rendered, _, _ = self.execute(
            MeasuredRouter(), [final_turn()])

        self.assertIn(
            'treesem_agent_intent_router_requests_total{result="success"} 1',
            rendered)
        self.assertIn(
            'treesem_agent_intent_dispatch_total{dispatch="workflow"} 1',
            rendered)
        self.assertIn(
            'treesem_agent_workflow_executions_total{recipe_class="single",result="success"} 1',
            rendered)
        self.assertIn("treesem_agent_llm_calls_per_run_sum 2.0", rendered)
        self.assertIn(
            'treesem_agent_llm_tokens_per_run_sum{kind="total"} 40.0',
            rendered)
        self.assertNotIn('dispatch="open_agent"', rendered)

    def test_repair_and_finalization_count_as_three_llm_calls(self):
        _, rendered, _, _ = self.execute(
            MeasuredRouter(attempts=2, repaired=True), [final_turn()])

        self.assertIn(
            'treesem_agent_intent_router_repairs_total{result="success"} 1',
            rendered)
        self.assertIn("treesem_agent_llm_calls_per_run_sum 3.0", rendered)
        self.assertIn(
            'treesem_agent_llm_tokens_per_run_sum{kind="total"} 50.0',
            rendered)

    def test_safety_refusal_makes_zero_router_and_llm_calls(self):
        router = MeasuredRouter()
        _, rendered, _, llm = self.execute(
            router, [], user_request=request("请帮我窃取其他患者的访问令牌"))

        self.assertEqual(router.calls, 0)
        self.assertEqual(llm.requests, [])
        self.assertNotIn("treesem_agent_intent_router_requests_total", rendered)
        self.assertIn("treesem_agent_llm_calls_per_run_sum 0.0", rendered)

    def test_metrics_and_trace_exclude_text_ids_evidence_and_payloads(self):
        _, rendered, events, _ = self.execute(
            MeasuredRouter(), [final_turn()])
        combined = rendered + repr(events)

        for secret in (
                "private explanation words", "private-model",
                PREDICTION_ID, SESSION_ID, REQUEST_ID, "decision_path"):
            self.assertNotIn(secret, combined)
        router_events = [item for item in events
                         if item["operation"] == "agent.intent_router"]
        dispatch_events = [item for item in events
                           if item["operation"] == "agent.intent_dispatch"]
        self.assertEqual(len(router_events), 1)
        self.assertEqual(router_events[0]["attempt_count"], 1)
        self.assertEqual(router_events[0]["repaired"], False)
        self.assertEqual(len(dispatch_events), 1)
        self.assertEqual(dispatch_events[0]["dispatch"], "workflow")
        self.assertEqual(dispatch_events[0]["goal_count"], 1)

    def test_router_failure_uses_stable_low_cardinality_result(self):
        registry = Metrics()
        events = []
        loop = AgentLoop(
            ScriptedLlmClient([]), ToolRegistry(ExplanationBackend()),
            structured_router=MeasuredRouter(failure=StructuredRouterError(
                "intent_router_unavailable", attempt_count=2,
                repaired=False)),
            routing_mode="structured_llm")
        with patch("agent.loop.metrics", registry), patch(
                "agent.loop.trace_event",
                side_effect=lambda trace, operation, started, outcome,
                **fields: events.append((operation, outcome, fields))):
            with self.assertRaisesRegex(Exception, "intent_router_unavailable"):
                asyncio.run(loop.run(request()))

        rendered = registry.render()
        self.assertIn(
            'treesem_agent_intent_router_requests_total{result="intent_router_unavailable"} 1',
            rendered)
        event = next(item for item in events
                     if item[0] == "agent.intent_router")
        self.assertEqual(event[1], "error")
        self.assertEqual(event[2]["error_code"],
                         "intent_router_unavailable")

    def test_shadow_records_safe_match_without_changing_legacy_response(self):
        registry = Metrics()
        events = []
        router = MeasuredRouter()
        router.result = StructuredRoute(
            other_frame(), usage=None, attempt_count=1, repaired=False)
        loop = AgentLoop(
            ScriptedLlmClient([LlmTurn(content="legacy response")]),
            ToolRegistry(ExplanationBackend()),
            structured_router=router,
            routing_mode="structured_shadow")
        with patch("agent.loop.metrics", registry), patch(
                "agent.loop.trace_event",
                side_effect=lambda trace, operation, started, outcome,
                **fields: events.append({
                    "operation": operation, "outcome": outcome, **fields})):
            response = asyncio.run(loop.run(request("你好")))

        rendered = registry.render()
        self.assertEqual(response.answer, "legacy response")
        self.assertIn(
            'treesem_agent_intent_shadow_total{result="match"} 1',
            rendered)
        shadow = next(item for item in events
                      if item["operation"] == "agent.intent_shadow")
        self.assertEqual(shadow["result"], "match")
        self.assertNotIn("你好", rendered + repr(events))

    def test_shadow_router_failure_is_observed_and_legacy_still_runs(self):
        registry = Metrics()
        events = []
        router = MeasuredRouter(failure=StructuredRouterError(
            "intent_router_unavailable", attempt_count=2))
        loop = AgentLoop(
            ScriptedLlmClient([LlmTurn(content="legacy response")]),
            ToolRegistry(ExplanationBackend()),
            structured_router=router,
            routing_mode="structured_shadow")
        with patch("agent.loop.metrics", registry), patch(
                "agent.loop.trace_event",
                side_effect=lambda trace, operation, started, outcome,
                **fields: events.append({
                    "operation": operation, "outcome": outcome, **fields})):
            response = asyncio.run(loop.run(request("你好")))

        self.assertEqual(response.answer, "legacy response")
        self.assertIn(
            'treesem_agent_intent_shadow_total{result="router_failure"} 1',
            registry.render())
        shadow = next(item for item in events
                      if item["operation"] == "agent.intent_shadow")
        self.assertEqual(shadow["outcome"], "error")


if __name__ == "__main__":
    unittest.main()
