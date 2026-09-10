from __future__ import annotations

import asyncio
import unittest

from agent.intent_frame import IntentFrame
from agent.llm_client import LlmToolPolicy, ScriptedLlmClient
from agent.loop import AgentExecutionError, AgentLoop
from agent.schemas import (
    AgentRunRequest,
    LlmTurn,
    PredictionContext,
)
from agent.structured_router import StructuredRoute, StructuredRouterError
from agent.tool_registry import ToolRegistry


CURRENT = "pred_" + "c" * 32
LATEST = "pred_" + "a" * 32
PREVIOUS = "pred_" + "b" * 32


class FakeBackend:
    def __init__(self):
        self.calls = []

    async def predict_sample(self, context, sample_index):
        self.calls.append(("predict_sample", sample_index))
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
        return {
            "items": [
                {"prediction_id": LATEST},
                {"prediction_id": PREVIOUS},
            ],
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


class FakeStructuredRouter:
    def __init__(self, outcome):
        self.outcome = outcome
        self.contexts = []

    async def route(self, context):
        self.contexts.append(context)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return StructuredRoute(
            self.outcome, usage=None, attempt_count=1, repaired=False)


class RecordingLlm(ScriptedLlmClient):
    def __init__(self, turns):
        super().__init__(turns)
        self.tool_name_sets = []

    async def complete(self, messages, tools, timeout,
                       tool_policy=LlmToolPolicy.auto()):
        self.tool_name_sets.append(frozenset(
            item["function"]["name"] for item in tools))
        return await super().complete(messages, tools, timeout, tool_policy)


def request(message: str, *, current=True):
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message=message,
        current_prediction=(PredictionContext(
            prediction_id=CURRENT, model_version="v1") if current else None),
        capability_token="capability",
    )


def goal(intent, target, evidence, aspects=None):
    return {
        "intent": intent,
        "target": {
            "type": target,
            "explicit_reference_index": None,
            "second_explicit_reference_index": None,
            "sample_reference_index": None,
        },
        "requested_aspects": aspects or [],
        "knowledge_scope": None,
        "evidence": evidence,
    }


def frame(goals, *, excluded_aspects=None, unresolved=None, clarify=False):
    return IntentFrame.model_validate({
        "schema_version": 1,
        "goals": goals,
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": excluded_aspects or [],
        },
        "unresolved_references": unresolved or [],
        "needs_clarification": clarify,
    })


class StructuredAgentLoopTest(unittest.TestCase):
    @staticmethod
    def execute(user_request, router_outcome, final_turns):
        backend = FakeBackend()
        llm = RecordingLlm(final_turns)
        router = FakeStructuredRouter(router_outcome)
        loop = AgentLoop(
            llm, ToolRegistry(backend),
            structured_router=router,
            routing_mode="structured_llm")
        result = asyncio.run(loop.run(user_request))
        return result, backend, llm, router

    def test_browser_current_result_request_executes_explanation_directly(self):
        user_request = request("解释一下刚刚的结果")
        router_frame = frame([
            goal("explanation", "current_prediction", ["刚刚的结果"]),
        ])

        result, backend, llm, router = self.execute(
            user_request, router_frame,
            [LlmTurn(content="这是当前预测的解释。",
                     grounding_prediction_ids=[CURRENT])])

        self.assertEqual(backend.calls, [("get_explanation", CURRENT)])
        self.assertEqual(result.grounding_prediction_ids, [CURRENT])
        self.assertEqual(len(router.contexts), 1)
        self.assertEqual(llm.tool_name_sets, [frozenset()])
        self.assertEqual(llm.tool_policies[0].mode, "none")

    def test_negation_and_previous_reference_use_history_then_explanation(self):
        user_request = request("不要解释结果，看看我们上次的决策树是什么样子的")
        router_frame = frame([
            goal("explanation", "previous_prediction", ["上次的决策树"],
                 ["decision_path"]),
        ], excluded_aspects=["prediction_summary"])

        result, backend, _, _ = self.execute(
            user_request, router_frame,
            [LlmTurn(content="这是上一条记录的决策路径。",
                     grounding_prediction_ids=[PREVIOUS])])

        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None),
            ("get_explanation", PREVIOUS),
        ])
        self.assertEqual(result.grounding_prediction_ids, [PREVIOUS])

    def test_registered_composite_executes_four_direct_tool_calls(self):
        user_request = request("比较最近两次结果并分别解释决策路径")
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"],
                 ["comparison_changes"]),
            goal("explanation", "latest_two_predictions", ["分别解释决策路径"],
                 ["decision_path"]),
        ])

        result, backend, llm, _ = self.execute(
            user_request, router_frame,
            [LlmTurn(content="完成比较并解释两条路径。",
                     grounding_prediction_ids=[LATEST, PREVIOUS])])

        self.assertEqual([item[0] for item in backend.calls], [
            "get_prediction_history", "compare_predictions",
            "get_explanation", "get_explanation",
        ])
        self.assertEqual(llm.tool_name_sets, [frozenset()])
        self.assertEqual(
            set(result.grounding_prediction_ids), {LATEST, PREVIOUS})

    def test_ambiguous_reference_returns_clarification_without_llm_or_tool(self):
        user_request = request("解释那个", current=False)
        router_frame = frame([
            goal("explanation", "none", ["那个"]),
        ], unresolved=["ambiguous_reference"], clarify=True)

        result, backend, llm, _ = self.execute(
            user_request, router_frame, [])

        self.assertIn("哪一条预测", result.answer)
        self.assertEqual(backend.calls, [])
        self.assertEqual(llm.requests, [])

    def test_router_failures_execute_no_tool_and_do_not_open_agent(self):
        for code in ("intent_router_unavailable", "invalid_intent_frame"):
            with self.subTest(code=code):
                backend = FakeBackend()
                llm = RecordingLlm([LlmTurn(content="must not be used")])
                loop = AgentLoop(
                    llm, ToolRegistry(backend),
                    structured_router=FakeStructuredRouter(
                        StructuredRouterError(code)),
                    routing_mode="structured_llm")

                with self.assertRaises(AgentExecutionError) as caught:
                    asyncio.run(loop.run(request("解释当前结果")))

                self.assertEqual(caught.exception.code, code)
                self.assertEqual(backend.calls, [])
                self.assertEqual(llm.requests, [])

    def test_other_request_opens_agent_without_side_effect_tools(self):
        user_request = request("帮我把这句话表达得清楚一点")
        router_frame = frame([
            goal("other", "none", ["表达得清楚"]),
        ])

        _, backend, llm, _ = self.execute(
            user_request, router_frame,
            [LlmTurn(content="可以这样表达。")])

        self.assertEqual(backend.calls, [])
        self.assertEqual(llm.tool_name_sets, [frozenset()])

    def test_unregistered_open_composition_exposes_only_validated_read_tools(self):
        user_request = request("列出历史并解释当前决策路径")
        router_frame = frame([
            goal("history", "session_history", ["列出历史"],
                 ["history_items"]),
            goal("explanation", "current_prediction", ["解释当前决策路径"],
                 ["decision_path"]),
        ])

        _, backend, llm, _ = self.execute(
            user_request, router_frame,
            [LlmTurn(content="你可以选择要查看的记录。")])

        self.assertEqual(backend.calls, [])
        self.assertEqual(llm.tool_name_sets, [frozenset({
            "get_prediction_history", "get_explanation",
        })])

    def test_safety_refusal_happens_before_structured_router(self):
        router_frame = frame([
            goal("knowledge", "general_knowledge", ["出血"]),
        ])
        backend = FakeBackend()
        router = FakeStructuredRouter(router_frame)
        llm = RecordingLlm([])
        loop = AgentLoop(
            llm, ToolRegistry(backend), structured_router=router,
            routing_mode="structured_llm")

        result = asyncio.run(loop.run(request("我现在大量出血并头晕")))

        self.assertIn("急救", result.answer)
        self.assertEqual(router.contexts, [])
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
