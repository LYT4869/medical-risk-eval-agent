from __future__ import annotations

import asyncio
import json
import unittest

from agent.intent_frame import IntentFrame
from agent.llm_client import LlmToolCall, LlmToolPolicy, ScriptedLlmClient
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
            "model_version": "v1",
            "important_features": [{"name": "feature-that-must-be-hidden"}],
            "decision_path": [{"node_id": 7}],
            "explanation_metadata": "must-not-reach-the-final-llm",
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


class FakeKnowledge:
    def __init__(self):
        self.calls = []

    async def search(self, token, query, scope, top_k, trace=None):
        self.calls.append((scope, top_k))
        return {"index_version": "test-index", "retrieval_mode": "hybrid",
                "results": [{
                    "citation_id": "cite_" + "e" * 20,
                    "source_id": "src_test", "title": "Model boundaries",
                    "section": "Metrics", "page": 1,
                    "excerpt": "AUC measures ranking, not individual changes.",
                    "publisher": "Synthetic test", "published_at": "2026-09-13",
                    "url": "https://example.invalid/metrics",
                    "content_sha256": "e" * 64, "score": 1.0,
                }]}


class RecordingLlm(ScriptedLlmClient):
    def __init__(self, turns):
        # This fixture represents an upstream model obeying the final envelope.
        normalized = []
        for turn in turns:
            if (turn.content and not turn.tool_calls and not turn.final_response_error
                    and not turn.content.lstrip().startswith("{")):
                turn = turn.model_copy(update={"content": json.dumps({
                    "answer": turn.content,
                    "grounding_prediction_ids": turn.grounding_prediction_ids,
                    "grounding_source_ids": turn.grounding_source_ids,
                }, ensure_ascii=False)})
            normalized.append(turn)
        super().__init__(normalized)
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


def frame(goals, *, excluded_aspects=None, unresolved=None, clarify=False,
          requested_skill=None):
    return IntentFrame.model_validate({
        "schema_version": 2,
        "goals": goals,
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": excluded_aspects or [],
        },
        "unresolved_references": unresolved or [],
        "needs_clarification": clarify,
        "requested_skill": requested_skill,
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

    def test_deterministic_workflow_projects_only_requested_explanation_data(self):
        user_request = request("只看当前结果的决策路径")
        router_frame = frame([
            goal("explanation", "current_prediction", ["决策路径"],
                 ["decision_path"]),
        ])

        _, _, llm, _ = self.execute(
            user_request, router_frame,
            [LlmTurn(content="这是当前预测的决策路径。",
                     grounding_prediction_ids=[CURRENT])])

        final_context = next(
            item["content"] for item in llm.requests[0]
            if item["role"] == "user" and
            item["content"].startswith("Finalization context:"))
        self.assertIn('"decision_path":[{"node_id":7}]', final_context)
        self.assertIn('"prediction_id":"' + CURRENT + '"', final_context)
        self.assertIn('"model_version":"v1"', final_context)
        self.assertNotIn("important_features", final_context)
        self.assertNotIn("explanation_metadata", final_context)
        self.assertFalse(any("node_id" in item.get("content", "")
                             for item in llm.requests[0]
                             if item["role"] == "system"))

    def test_structured_open_agent_projects_tool_messages_before_reuse(self):
        user_request = request("列出历史并只解释当前决策路径")
        router_frame = frame([
            goal("history", "session_history", ["列出历史"],
                 ["history_items"]),
            goal("explanation", "current_prediction", ["决策路径"],
                 ["decision_path"]),
        ])

        _, _, llm, _ = self.execute(
            user_request, router_frame,
            [
                LlmTurn(tool_calls=[LlmToolCall(
                    id="explain_1", name="get_explanation",
                    arguments={"prediction_id": CURRENT})]),
                LlmTurn(content="这是当前预测的决策路径。",
                        grounding_prediction_ids=[CURRENT]),
            ])

        tool_context = next(
            item["content"] for item in llm.requests[1]
            if item["role"] == "tool")
        self.assertIn('"decision_path": [{"node_id": 7}]', tool_context)
        self.assertIn('"prediction_id": "' + CURRENT + '"', tool_context)
        self.assertNotIn("important_features", tool_context)
        self.assertNotIn("explanation_metadata", tool_context)

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


    def test_finalization_failure_preserves_safe_partial_execution_summary(self):
        user_request = request("读取当前摘要并列出历史；不要泄露 capability")
        router_frame = frame([
            goal("summary", "current_prediction", ["当前摘要"]),
            goal("history", "session_history", ["列出历史"]),
        ])
        backend = FakeBackend()
        llm = RecordingLlm([
            LlmTurn(tool_calls=[LlmToolCall(
                id="first", name="get_prediction",
                arguments={"prediction_id": CURRENT})]),
            LlmTurn(tool_calls=[LlmToolCall(
                id="second", name="get_prediction_history",
                arguments={"limit": 2})]),
            LlmTurn(final_response_error="invalid_final_response"),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend), max_steps=3,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(loop.run(user_request))

        progress = getattr(caught.exception, "execution_progress", None)
        self.assertIsNotNone(progress)
        self.assertEqual(progress["llm_call_count"], 3)
        self.assertEqual(progress["tool_calls"], [
            {"name": "get_prediction", "status": "success"},
            {"name": "get_prediction_history", "status": "success"},
        ])
        self.assertEqual(progress["completed_goal_indexes"], [0, 1])
        self.assertEqual(progress["pending_goal_indexes"], [])
        self.assertEqual(progress["llm_step_kinds"],
                         ["tool_call", "tool_call", "final_answer"])
        self.assertNotIn(CURRENT, str(progress))
        self.assertNotIn("capability", str(progress))
        self.assertNotIn("arguments", str(progress))

    def test_comparison_and_knowledge_reuse_bound_recipes_without_llm_planning(self):
        user_request = request("比较最近两次预测并查询AUC资料")
        user_request.knowledge_capability_token = "knowledge-capability"
        knowledge_goal = goal("knowledge", "general_knowledge", ["AUC资料"],
                              ["knowledge_overview"])
        knowledge_goal["knowledge_scope"] = "model"
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"],
                 ["comparison_changes"]), knowledge_goal,
        ])
        backend, knowledge = FakeBackend(), FakeKnowledge()
        citation = "cite_" + "e" * 20
        llm = RecordingLlm([LlmTurn(
            content="比较与AUC资料说明。" + citation,
            grounding_prediction_ids=[LATEST, PREVIOUS],
            grounding_source_ids=[citation])])
        loop = AgentLoop(llm, ToolRegistry(backend, knowledge),
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        result = asyncio.run(loop.run(user_request))

        self.assertEqual(backend.calls, [
            ("get_prediction_history", 2, None),
            ("compare_predictions", LATEST, PREVIOUS),
        ])
        self.assertEqual(knowledge.calls, [("model", 5)])
        self.assertEqual(llm.tool_name_sets, [frozenset()])
        self.assertEqual(result.knowledge_index_version, "test-index")

    def test_final_protocol_repair_runs_once_without_reexecuting_tools(self):
        user_request = request("只看当前决策路径")
        router_frame = frame([
            goal("explanation", "current_prediction", ["决策路径"],
                 ["decision_path"]),
        ])
        result, backend, llm, _ = self.execute(
            user_request, router_frame, [
                LlmTurn(final_response_error="invalid_final_response"),
                LlmTurn(content="这是当前路径。", grounding_prediction_ids=[CURRENT]),
            ])

        self.assertEqual(backend.calls, [("get_explanation", CURRENT)])
        self.assertEqual(result.answer, "这是当前路径。")
        self.assertEqual([p.mode for p in llm.tool_policies], ["none", "none"])
        self.assertEqual(len(llm.requests), 2)

    def test_failed_final_protocol_repair_does_not_expose_envelope(self):
        user_request = request("只看当前决策路径")
        router_frame = frame([
            goal("explanation", "current_prediction", ["决策路径"],
                 ["decision_path"]),
        ])
        result, backend, llm, _ = self.execute(
            user_request, router_frame, [
                LlmTurn(final_response_error="invalid_final_response"),
                LlmTurn(final_response_error="invalid_final_response"),
                LlmTurn(content="不应该调用第三次"),
            ])

        self.assertIn("自然语言说明暂时不可用", result.answer)
        self.assertEqual(backend.calls, [("get_explanation", CURRENT)])
        self.assertEqual(len(llm.requests), 2)
        self.assertNotIn("grounding_prediction_ids", result.answer)

    def test_completed_open_goals_force_finalization_with_original_request(self):
        user_request = request("先列出历史，再用通俗语言概括当前结果")
        router_frame = frame([
            goal("history", "session_history", ["列出历史"]),
            goal("summary", "current_prediction", ["当前结果"]),
        ])
        result, backend, llm, _ = self.execute(
            user_request, router_frame, [
                LlmTurn(tool_calls=[LlmToolCall(
                    id="h", name="get_prediction_history", arguments={"limit": 2})]),
                LlmTurn(tool_calls=[LlmToolCall(
                    id="p", name="get_prediction", arguments={"prediction_id": CURRENT})]),
                LlmTurn(content="历史与当前结果。", grounding_prediction_ids=[CURRENT]),
            ])

        self.assertEqual(result.answer, "历史与当前结果。")
        self.assertEqual(llm.tool_policies[-1].mode, "none")
        final = next(m["content"] for m in llm.requests[-1]
                     if m["role"] == "user" and
                     m["content"].startswith("Finalization context:"))
        payload = json.loads(final.split("\n", 1)[1])
        self.assertEqual(payload["original_request"], user_request.message)
        self.assertEqual([g["status"] for g in payload["subgoals"]],
                         ["completed", "completed"])
        self.assertEqual([e["tool"] for e in payload["current_evidence"]],
                         ["get_prediction_history", "get_prediction"])

    def test_last_open_step_is_reserved_for_answer_not_another_tool(self):
        user_request = request("读取摘要并列出历史")
        router_frame = frame([
            goal("summary", "current_prediction", ["摘要"]),
            goal("history", "session_history", ["历史"]),
        ])
        backend = FakeBackend()
        llm = RecordingLlm([
            LlmTurn(tool_calls=[LlmToolCall(
                id="p", name="get_prediction", arguments={"prediction_id": CURRENT})]),
            LlmTurn(content="已读取当前结果，但未取得历史。", grounding_prediction_ids=[CURRENT]),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend), max_steps=2,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        result = asyncio.run(loop.run(user_request))

        self.assertEqual(backend.calls, [("get_prediction", CURRENT)])
        self.assertEqual(llm.tool_name_sets[-1], frozenset())
        self.assertEqual(llm.tool_policies[-1].mode, "none")
        self.assertIn("未取得历史", result.answer)

    def test_total_deadline_cancels_deterministic_tool_execution(self):
        class SlowBackend(FakeBackend):
            async def get_explanation(self, context, prediction_id):
                await asyncio.sleep(1)
                return await super().get_explanation(context, prediction_id)

        user_request = request("解释当前决策路径")
        router_frame = frame([
            goal("explanation", "current_prediction", ["决策路径"]),
        ])
        backend, llm = SlowBackend(), RecordingLlm([])
        loop = AgentLoop(llm, ToolRegistry(backend), total_timeout_seconds=0.02,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(asyncio.wait_for(loop.run(user_request), 0.2))

        self.assertEqual(caught.exception.code, "agent_timeout")
        self.assertEqual(backend.calls, [])
        self.assertEqual(llm.requests, [])

    def test_total_deadline_cancels_open_agent_tool_execution(self):
        class SlowBackend(FakeBackend):
            async def get_history(self, context, limit, cursor):
                await asyncio.sleep(1)
                return await super().get_history(context, limit, cursor)

        user_request = request("列出历史并概括当前预测")
        router_frame = frame([
            goal("history", "session_history", ["历史"]),
            goal("summary", "current_prediction", ["当前预测"]),
        ])
        backend = SlowBackend()
        llm = RecordingLlm([LlmTurn(tool_calls=[LlmToolCall(
            id="h", name="get_prediction_history", arguments={"limit": 2})])])
        loop = AgentLoop(llm, ToolRegistry(backend), total_timeout_seconds=0.02,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(asyncio.wait_for(loop.run(user_request), 0.2))

        self.assertEqual(caught.exception.code, "agent_timeout")
        self.assertEqual(caught.exception.execution_progress["llm_call_count"], 1)
        self.assertEqual(caught.exception.execution_progress["tool_calls"], [
            {"name": "get_prediction_history", "status": "error"}])
        self.assertEqual(backend.calls, [])

    def test_unknown_tool_names_are_not_copied_into_execution_logs(self):
        user_request = request("列出历史并概括当前预测")
        router_frame = frame([
            goal("history", "session_history", ["历史"]),
            goal("summary", "current_prediction", ["当前预测"]),
        ])
        secret_name = "sk-sensitive-provider-data"
        llm = RecordingLlm([
            LlmTurn(tool_calls=[LlmToolCall(
                id="unknown", name=secret_name, arguments={})]),
            LlmTurn(final_response_error="invalid_final_response"),
        ])
        loop = AgentLoop(llm, ToolRegistry(FakeBackend()), max_steps=2,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        with self.assertRaises(AgentExecutionError) as caught:
            asyncio.run(loop.run(user_request))

        progress = caught.exception.execution_progress
        self.assertNotIn(secret_name, str(progress))
        self.assertEqual(progress["tool_calls"], [
            {"name": "unknown_tool", "status": "error"}])

    def test_tool_budget_exhaustion_enters_finalizer_without_extra_tool(self):
        user_request = request("读取摘要并列出历史")
        router_frame = frame([
            goal("summary", "current_prediction", ["摘要"]),
            goal("history", "session_history", ["历史"]),
        ])
        backend = FakeBackend()
        llm = RecordingLlm([
            LlmTurn(tool_calls=[LlmToolCall(
                id="p", name="get_prediction", arguments={"prediction_id": CURRENT})]),
            LlmTurn(content="当前结果已读取，历史尚未取得。", grounding_prediction_ids=[CURRENT]),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend), max_tool_calls=1,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")

        asyncio.run(loop.run(user_request))

        self.assertEqual(backend.calls, [("get_prediction", CURRENT)])
        self.assertEqual(llm.tool_policies[-1].mode, "none")
        self.assertEqual(llm.tool_name_sets[-1], frozenset())

    def test_batch_exhausting_tool_budget_finalizes_without_executing_excess(self):
        user_request = request("读取摘要并列出历史")
        router_frame = frame([
            goal("summary", "current_prediction", ["摘要"]),
            goal("history", "session_history", ["历史"]),
        ])
        backend = FakeBackend()
        llm = RecordingLlm([
            LlmTurn(tool_calls=[
                LlmToolCall(id="p", name="get_prediction",
                            arguments={"prediction_id": CURRENT}),
                LlmToolCall(id="h", name="get_prediction_history",
                            arguments={"limit": 2}),
            ]),
            LlmTurn(content="已取得当前摘要，但历史尚未取得。",
                    grounding_prediction_ids=[CURRENT]),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend), max_tool_calls=1,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        result = asyncio.run(loop.run(user_request))
        self.assertEqual(backend.calls, [("get_prediction", CURRENT)])
        self.assertIn("尚未取得", result.answer)
        self.assertEqual(llm.tool_policies[-1].mode, "none")

    def test_composed_workflow_cannot_exceed_domain_tool_budget(self):
        user_request = request("比较最近两次预测并查询AUC资料")
        user_request.knowledge_capability_token = "knowledge-capability"
        knowledge_goal = goal("knowledge", "general_knowledge", ["AUC资料"])
        knowledge_goal["knowledge_scope"] = "model"
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
            knowledge_goal,
        ])
        backend, knowledge = FakeBackend(), FakeKnowledge()
        llm = RecordingLlm([LlmTurn(
            content="已取得历史；比较与资料尚未取得。",
            grounding_prediction_ids=[LATEST, PREVIOUS])])
        loop = AgentLoop(llm, ToolRegistry(backend, knowledge),
                         max_tool_calls=1,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        result = asyncio.run(loop.run(user_request))
        self.assertEqual(backend.calls, [("get_prediction_history", 2, None)])
        self.assertEqual(knowledge.calls, [])
        self.assertEqual(len(result.tools_used), 1)
        self.assertIn("尚未取得", result.answer)
        self.assertEqual(llm.tool_policies[-1].mode, "none")
        payload = json.loads(next(m["content"].split("\n", 1)[1]
            for m in llm.requests[-1] if m["role"] == "user" and
            m["content"].startswith("Finalization context:")))
        self.assertEqual([g["status"] for g in payload["subgoals"]],
                         ["pending", "pending"])

    def test_finalizer_repairs_plain_prose_once_without_rerunning_tools(self):
        backend = FakeBackend()
        llm = ScriptedLlmClient([
            LlmTurn(content="plain text without required envelope",
                    grounding_prediction_ids=[CURRENT]),
            LlmTurn(content=json.dumps({
                "answer": "决策路径已取得。", "grounding_prediction_ids": [CURRENT],
                "grounding_source_ids": []})),
        ])
        router_frame = frame([
            goal("explanation", "current_prediction", ["决策路径"], ["decision_path"]),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend),
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        result = asyncio.run(loop.run(request("只看当前决策路径")))
        self.assertEqual(result.answer, "决策路径已取得。")
        self.assertEqual(len(llm.requests), 2)
        self.assertEqual(backend.calls, [("get_explanation", CURRENT)])

    def test_decoded_valid_envelope_answer_is_not_parsed_as_another_envelope(self):
        loop = AgentLoop(RecordingLlm([]), ToolRegistry(FakeBackend()))
        for answer in ('[1] 已取得路径。', '{"result":"已取得路径"}'):
            with self.subTest(answer=answer):
                turn = LlmTurn(content=answer, final_response_is_structured=True,
                               grounding_prediction_ids=[CURRENT])
                checked, _ = loop._check_final(turn, {CURRENT}, set(),
                                               require_envelope=True)
                self.assertEqual(checked, answer)

    def test_partial_workflow_bad_final_does_not_claim_comparison_was_completed(self):
        user_request = request("比较最近两次预测并查询AUC资料")
        knowledge_goal = goal("knowledge", "general_knowledge", ["AUC资料"])
        knowledge_goal["knowledge_scope"] = "model"
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
            knowledge_goal,
        ])
        llm = RecordingLlm([LlmTurn(final_response_error="invalid_final_response")]*2)
        loop = AgentLoop(llm, ToolRegistry(FakeBackend(), FakeKnowledge()),
                         max_tool_calls=1,
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        result = asyncio.run(loop.run(user_request))
        self.assertNotIn("比较数据已经取得", result.answer)
        self.assertIn("未完整执行", result.answer)

    def test_finalizer_requires_body_citation_even_when_array_is_valid(self):
        loop = AgentLoop(RecordingLlm([]), ToolRegistry(FakeBackend()))
        citation = "cite_" + "e" * 20
        turn = LlmTurn(content="AUC说明。", final_response_is_structured=True,
                       grounding_source_ids=[citation])
        from agent.policy import PolicyViolation
        with self.assertRaises(PolicyViolation) as caught:
            loop._check_final(turn, set(), {citation}, require_envelope=True)
        self.assertEqual(caught.exception.code, "missing_knowledge_citation")

    def test_workflow_final_evidence_has_program_bound_time_direction(self):
        from test_comparison_presentation import comparison

        class RichBackend(FakeBackend):
            async def compare(self, context, prediction_id_a, prediction_id_b):
                self.raw = comparison()
                return self.raw

        backend = RichBackend()
        llm = RecordingLlm([LlmTurn(content="上次0.34，本次0.82，上升48个百分点。",
                                   grounding_prediction_ids=[LATEST, PREVIOUS])])
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
        ])
        loop = AgentLoop(llm, ToolRegistry(backend),
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        asyncio.run(loop.run(request("比较最近两次预测")))
        payload = json.loads(next(m["content"].split("\n", 1)[1]
            for m in llm.requests[-1] if m["role"] == "user" and
            m["content"].startswith("Finalization context:")))
        data = next(e["data"] for e in payload["current_evidence"]
                    if e["tool"] == "compare_predictions")
        self.assertEqual(data["comparison_order"], "previous_to_latest")
        self.assertEqual(data["positive_probability_delta"], 0.48)
        self.assertEqual(backend.raw["positive_probability_delta"], -0.48)

    def test_open_finalizer_refreshes_comparison_after_history_binds_targets(self):
        from test_comparison_presentation import comparison

        class RichBackend(FakeBackend):
            async def compare(self, context, prediction_id_a, prediction_id_b):
                return comparison()

        # Reads are independent; relative binding arrives after comparison facts.
        llm = RecordingLlm([
            LlmTurn(tool_calls=[LlmToolCall(id="c", name="compare_predictions",
                arguments={"prediction_id_a": LATEST, "prediction_id_b": PREVIOUS})]),
            LlmTurn(tool_calls=[LlmToolCall(id="h", name="get_prediction_history",
                arguments={"limit": 2})]),
            LlmTurn(content="上次0.34，本次0.82，上升48个百分点。",
                    grounding_prediction_ids=[LATEST, PREVIOUS]),
        ])
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
            goal("history", "session_history", ["列出历史"]),
        ])
        loop = AgentLoop(llm, ToolRegistry(RichBackend()),
                         structured_router=FakeStructuredRouter(router_frame),
                         routing_mode="structured_llm")
        asyncio.run(loop.run(request("比较最近两次并列出历史")))
        payload = json.loads(next(m["content"].split("\n", 1)[1]
            for m in llm.requests[-1] if m["role"] == "user" and
            m["content"].startswith("Finalization context:")))
        data = next(e["data"] for e in payload["current_evidence"]
                    if e["tool"] == "compare_predictions")
        self.assertEqual(data["comparison_order"], "previous_to_latest")
        self.assertEqual(data["positive_probability_direction"], "increase")

    def test_finalization_context_names_comparison_facts_to_cover(self):
        user_request = request("比较最近两次结果并解释两条路径")
        router_frame = frame([
            goal("comparison", "latest_two_predictions", ["比较最近两次"]),
            goal("explanation", "latest_two_predictions", ["两条路径"],
                 ["decision_path"]),
        ])
        _, _, llm, _ = self.execute(
            user_request, router_frame, [LlmTurn(
                content="比较和路径已取得。", grounding_prediction_ids=[LATEST, PREVIOUS])])
        final = next(m["content"] for m in llm.requests[0]
                     if m["role"] == "user" and
                     m["content"].startswith("Finalization context:"))
        payload = json.loads(final.split("\n", 1)[1])

        self.assertEqual(payload["subgoals"][0].get("evidence_fields_to_cover"), [
            "positive_probability_delta", "label_changed",
            "model_version_changed", "path_changed"])
        self.assertEqual(payload["subgoals"][1].get("evidence_fields_to_cover"),
                         ["decision_path"])


if __name__ == "__main__":
    unittest.main()
