"""Knowledge evidence is not patient evidence, in either routing mode."""
from __future__ import annotations

import asyncio
import json
import unittest

from agent.llm_client import ScriptedLlmClient
from agent.loop import AgentLoop
from agent.policy import PolicyViolation, ResponsePolicy
from agent.routing_types import RequestScope, RoutingDecision, RoutingSource
from agent.schemas import LlmToolCall, LlmTurn, RecentMessage
from agent.tool_registry import ToolRegistry
from test_agent_loop import FakeRouter
from test_structured_agent_loop import (
    CURRENT, FakeBackend, FakeKnowledge, FakeStructuredRouter,
    RecordingLlm, frame, goal, request,
)


CITATION = "cite_" + "e" * 20
FOREIGN_CITATION = "cite_" + "f" * 20


class ModelKnowledge(FakeKnowledge):
    async def search(self, token, query, scope, top_k, trace=None):
        result = await super().search(token, query, scope, top_k, trace)
        result["results"][0].update({
            "title": "Synthetic treeSem output contract",
            "section": "Output definitions and aggregate metrics",
            "excerpt": (
                "positive_probability is the class-1 probability. Confidence is "
                "the probability of the predicted class. Label 0/1 encodes the "
                "negative/positive class, not a clinical diagnosis. Accuracy in "
                "the registered aggregate evaluation is 96.3734%. Tree and neural "
                "network outputs are distinct; a path alone is not importance. "
                "macro-F1 is the unweighted mean of per-class F1 scores."),
        })
        return result


class KnowledgeEvidenceBoundaryTest(unittest.TestCase):
    def execute_knowledge(self, answer, *, current=True, cited=None,
                          prediction_ids=None, legacy=False, recent=None):
        user_request = request(
            "查模型资料说明概率和置信度有什么区别，注明来源。", current=current
        ).model_copy(update={
            "knowledge_capability_token": "synthetic-test-capability",
            "recent_messages": recent or [],
        })
        final = LlmTurn(content=answer,
                        grounding_prediction_ids=prediction_ids or [],
                        grounding_source_ids=[CITATION] if cited is None else cited)
        backend, knowledge = FakeBackend(), ModelKnowledge()
        if legacy:
            llm = ScriptedLlmClient([
                LlmTurn(tool_calls=[LlmToolCall(
                    id="k1", name="search_medical_knowledge",
                    arguments={"query": "model probabilities", "scope": "model",
                               "top_k": 5})]),
                final,
            ])
            loop = AgentLoop(llm, ToolRegistry(backend, knowledge),
                             router=FakeRouter(RoutingDecision(
                                 RequestScope.KNOWLEDGE, RoutingSource.RULE)))
        else:
            knowledge_goal = goal("knowledge", "general_knowledge", ["模型资料"])
            knowledge_goal["knowledge_scope"] = "model"
            llm = RecordingLlm([final])
            loop = AgentLoop(
                llm, ToolRegistry(backend, knowledge), routing_mode="structured_llm",
                structured_router=FakeStructuredRouter(frame([knowledge_goal])))
        return asyncio.run(loop.run(user_request)), backend, llm

    def test_verified_model_terms_do_not_require_a_patient_prediction(self):
        for current in (True, False):
            for answer in (
                    "概率和置信度是两个模型输出术语。来源：" + CITATION,
                    "Label 0/1 is an encoding, not a diagnosis. Source: " + CITATION):
                with self.subTest(current=current, answer=answer):
                    result, backend, _ = self.execute_knowledge(answer, current=current)
                    self.assertEqual(result.answer, answer)
                    self.assertEqual(result.grounding_prediction_ids, [])
                    self.assertEqual(result.grounding_source_ids, [CITATION])
                    self.assertEqual(backend.calls, [])

    def test_legacy_knowledge_answers_get_the_same_evidence_boundary(self):
        answer = "概率和置信度的定义来自模型资料。来源：" + CITATION
        result, backend, _ = self.execute_knowledge(answer, legacy=True)
        self.assertEqual(result.answer, answer)
        self.assertEqual(result.grounding_prediction_ids, [])
        self.assertEqual(result.grounding_source_ids, [CITATION])
        self.assertEqual(backend.calls, [])

    def test_knowledge_context_excludes_unread_patient_record_and_old_clinical_chat(self):
        clinical_message = "患者旧聊天：" + CURRENT + "，上次阳性概率为70%。"
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                result, _, llm = self.execute_knowledge(
                    "模型概率不是临床诊断。来源：" + CITATION, legacy=legacy,
                    recent=[RecentMessage(role="assistant", content=clinical_message)])
                self.assertEqual(result.grounding_prediction_ids, [])
                for messages in llm.requests:
                    serialized = json.dumps(messages, ensure_ascii=False)
                    self.assertNotIn(CURRENT, serialized)
                    self.assertNotIn(clinical_message, serialized)

    def test_model_metric_percentage_can_use_knowledge_evidence(self):
        answer = "已登记的模型评测 Accuracy 为96.3734%，不是你的预测概率。来源：" + CITATION
        result, _, _ = self.execute_knowledge(answer)
        self.assertEqual(result.answer, answer)
        self.assertEqual(result.grounding_prediction_ids, [])

    def test_current_model_encoding_is_not_a_current_patient_result(self):
        answer = "当前模型的标签为1表示正类，并不等同临床诊断。来源：" + CITATION
        result, _, _ = self.execute_knowledge(answer)
        self.assertEqual(result.answer, answer)
        self.assertEqual(result.grounding_prediction_ids, [])

    def test_current_context_id_is_still_not_valid_prediction_grounding(self):
        result, _, _ = self.execute_knowledge(
            "树模型和神经网络是不同的输出链。来源：" + CITATION,
            prediction_ids=[CURRENT])
        self.assertEqual(result.grounding_prediction_ids, [])
        self.assertEqual(result.grounding_source_ids, [])
        self.assertNotIn("不同的输出链", result.answer)

    def test_a_knowledge_citation_does_not_authorize_patient_measurements(self):
        for answer in (
                "你的阳性概率是70%。来源：" + CITATION,
                "当前预测的置信度为0.9。来源：" + CITATION,
                "你的标签是1。来源：" + CITATION,
                "你的阳性概率升至70%。来源：" + CITATION,
                "你上次的阳性概率是70%。来源：" + CITATION,
                "Your positive_probability is 0.7. Source: " + CITATION,
                "For your prediction, the positive probability is 70%. Source: " + CITATION,
                "你的阳性概率，升至70%。来源：" + CITATION,
                "Your predicted probability is 70%. Source: " + CITATION,
                "Your prediction is positive. Source: " + CITATION):
            with self.subTest(answer=answer):
                result, _, _ = self.execute_knowledge(answer)
                self.assertNotEqual(result.answer, answer)
                self.assertEqual(result.grounding_prediction_ids, [])
                self.assertEqual(result.grounding_source_ids, [])

    def test_foreign_knowledge_reference_is_not_accepted(self):
        result, _, _ = self.execute_knowledge(
            "概率定义。来源：" + FOREIGN_CITATION, cited=[FOREIGN_CITATION])
        self.assertEqual(result.grounding_source_ids, [])
        self.assertNotIn(FOREIGN_CITATION, result.answer)

    def test_knowledge_terms_without_any_reference_do_not_bypass_grounding(self):
        answer = "概率和置信度是不同的模型术语。"
        result, _, _ = self.execute_knowledge(answer, cited=[])
        self.assertNotEqual(result.answer, answer)
        self.assertEqual(result.grounding_source_ids, [])

    def test_valid_rag_evidence_does_not_grant_knowledge_only_scope(self):
        with self.assertRaises(PolicyViolation) as caught:
            ResponsePolicy().validate(
                "The probability is 70%. Source: " + CITATION, [], set(),
                cited_sources=[CITATION], available_sources={CITATION},
                knowledge_only=False)
        self.assertEqual(caught.exception.code, "missing_prediction_grounding")

    def test_metric_disclaimer_does_not_depend_on_a_comma(self):
        for answer in (
                "Accuracy is 96.3734% and is not your prediction probability. Source: " + CITATION,
                "模型 Accuracy 为96.3734%，不能作为你的预测概率。来源：" + CITATION):
            with self.subTest(answer=answer):
                result, _, _ = self.execute_knowledge(answer)
                self.assertEqual(result.answer, answer)
                self.assertEqual(result.grounding_prediction_ids, [])

    def test_explicit_english_probability_is_not_an_unresolved_pronoun(self):
        from agent.knowledge_context import knowledge_topic_hint
        self.assertEqual(knowledge_topic_hint(
            request("What is probability and how is it defined?")), (None, False))

    def test_knowledge_followup_preserves_a_public_topic_without_clinical_history(self):
        class TopicKnowledge(ModelKnowledge):
            async def search(self, token, query, scope, top_k, trace=None):
                self.query = query
                return await super().search(token, query, scope, top_k, trace)

        user_request = request("再查资料解释一下它，注明来源。").model_copy(update={
            "knowledge_capability_token": "synthetic-test-capability",
            "recent_messages": [RecentMessage(
                role="user", content="macro-F1 是什么？旧患者记录 " + CURRENT + " 概率70%。")],
        })
        knowledge_goal = goal("knowledge", "general_knowledge", ["查资料"])
        knowledge_goal["knowledge_scope"] = "model"
        knowledge = TopicKnowledge()
        llm = RecordingLlm([LlmTurn(
            content="macro-F1 属于一般模型指标。来源：" + CITATION,
            grounding_source_ids=[CITATION])])
        router = FakeStructuredRouter(frame([knowledge_goal]))
        loop = AgentLoop(llm, ToolRegistry(FakeBackend(), knowledge),
                         routing_mode="structured_llm", structured_router=router)
        result = asyncio.run(loop.run(user_request))
        self.assertEqual(result.grounding_prediction_ids, [])
        self.assertIn("macro-F1", knowledge.query)
        self.assertNotIn(CURRENT, knowledge.query)
        self.assertNotIn("70%", knowledge.query)
        serialized = json.dumps(llm.requests[0], ensure_ascii=False)
        self.assertIn("macro-F1", serialized)
        self.assertNotIn(CURRENT, serialized)
        self.assertNotIn("70%", serialized)
        self.assertEqual(router.contexts[0].recent_messages,
                         tuple(user_request.recent_messages))

    def test_unresolved_knowledge_pronoun_clarifies_without_guessing_or_tools(self):
        user_request = request("再查资料解释一下它，注明来源。").model_copy(update={
            "knowledge_capability_token": "synthetic-test-capability",
            "recent_messages": [RecentMessage(role="assistant", content="已记录你的问题。")],
        })
        knowledge_goal = goal("knowledge", "general_knowledge", ["查资料"])
        knowledge_goal["knowledge_scope"] = "model"
        knowledge, llm = ModelKnowledge(), RecordingLlm([])
        loop = AgentLoop(llm, ToolRegistry(FakeBackend(), knowledge),
                         routing_mode="structured_llm",
                         structured_router=FakeStructuredRouter(frame([knowledge_goal])))
        result = asyncio.run(loop.run(user_request))
        self.assertIn("主题", result.answer)
        self.assertEqual(knowledge.calls, [])
        self.assertEqual(llm.requests, [])
        self.assertEqual(result.tools_used, [])

    def test_business_workflow_keeps_current_binding_and_native_grounding(self):
        backend = FakeBackend()
        answer = "这是当前结果的决策路径。"
        llm = RecordingLlm([LlmTurn(content=answer, grounding_prediction_ids=[CURRENT])])
        loop = AgentLoop(llm, ToolRegistry(backend), routing_mode="structured_llm",
                         structured_router=FakeStructuredRouter(frame([
                             goal("explanation", "current_prediction", ["决策路径"],
                                  ["decision_path"])])))
        result = asyncio.run(loop.run(request("只看当前决策路径")))
        self.assertEqual(result.answer, answer)
        self.assertEqual(backend.calls, [("get_explanation", CURRENT)])
        self.assertEqual(result.grounding_prediction_ids, [CURRENT])


if __name__ == "__main__":
    unittest.main()
