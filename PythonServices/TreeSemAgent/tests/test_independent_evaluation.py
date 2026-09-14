from __future__ import annotations

import asyncio
import json
import unittest

from evaluation.run_evaluation import FakeBackend, Case, run_case, FixtureStructuredRouter
from agent.llm_client import ScriptedLlmClient
from agent.schemas import LlmTurn


class IndependentFixturesTest(unittest.TestCase):
    def test_prediction_fixture_is_selected_by_requested_id(self):
        backend = FakeBackend({"prediction_by_id": {
            "pred_" + "1" * 32: {"prediction_id": "pred_" + "1" * 32, "label": 0},
            "pred_" + "2" * 32: {"prediction_id": "pred_" + "2" * 32, "label": 1}}})
        value = asyncio.run(backend.get_prediction(None, "pred_" + "2" * 32))
        self.assertEqual(value["label"], 1)

    def test_unknown_fixture_id_is_not_fabricated(self):
        from agent.tools.backend import ToolExecutionError
        backend = FakeBackend({"prediction_by_id": {}})
        with self.assertRaises(ToolExecutionError):
            asyncio.run(backend.get_prediction(None, "pred_" + "f" * 32))

    def test_comparison_matches_argument_order_and_b_minus_a(self):
        backend = FakeBackend({"prediction_by_id": {
            "pred_" + "1" * 32: {"prediction_id": "pred_" + "1" * 32,
                "positive_probability": .75, "confidence": .75, "label": 1},
            "pred_" + "2" * 32: {"prediction_id": "pred_" + "2" * 32,
                "positive_probability": .25, "confidence": .75, "label": 0}}})
        value = asyncio.run(backend.compare(None, "pred_" + "1" * 32, "pred_" + "2" * 32))
        self.assertEqual(value["positive_probability_delta"], -.5)
        self.assertEqual(value["prediction_b"]["label"], 0)
        self.assertTrue(value["label_changed"])

    def test_current_context_and_observed_grounding_are_not_fixed_global_ids(self):
        ident = "pred_" + "7" * 32
        case = Case("custom_current", "summary", "patient", "查看刚才的预测结果",
                    ["get_prediction"], "prediction", None, False,
                    fixture={"collect_synthetic_observations": True,
                             "current_prediction": {"prediction_id": ident},
                             "prediction_by_id": {ident: {"prediction_id": ident}}},
                    structured_goals=({"intent": "summary", "target": "current_prediction",
                                       "requested_aspects": ["prediction_summary"]},))
        client = ScriptedLlmClient([LlmTurn(content='{"answer":"结果已获取",'
            '"grounding_prediction_ids":["' + ident + '"],"grounding_source_ids":[]}')])
        result = asyncio.run(run_case(case, client, routing_mode="structured_llm",
                                     structured_router=FixtureStructuredRouter(case)))
        self.assertTrue(result["prediction_grounding_valid"])
        self.assertEqual(result["grounding_prediction_ids"], [ident])
        self.assertEqual(result["backend_calls"], [
            {"name": "get_prediction", "arguments": {"prediction_id": ident}}])


class IndependentScoringTest(unittest.TestCase):
    def test_resume_rejects_changed_effective_configuration_or_evaluator(self):
        from evaluation.run_independent_ab import load_checkpoint
        from tempfile import TemporaryDirectory
        from pathlib import Path
        profile = {"answer": {"max_output_tokens": 1024},
                   "router": {"maximum_attempts": 2}, "evaluator_sha256": "original"}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"dataset_sha256": "data", "agent_source_sha256": "source",
                "model": "model", "router_budget": {}, "evaluation_profile": profile,
                "results": [], "llm_usage": {}}))
            self.assertEqual(load_checkpoint(path, dataset_sha="data", agent_source_sha="source",
                model="model", router_budget={}, evaluation_profile=profile)["results"], [])
            for changed in ({**profile, "evaluator_sha256": "changed"},
                            {**profile, "router": {"maximum_attempts": 1}},
                            {**profile, "answer": {"max_output_tokens": 512}}):
                with self.assertRaises(ValueError):
                    load_checkpoint(path, dataset_sha="data", agent_source_sha="source",
                        model="model", router_budget={}, evaluation_profile=changed)

    def test_profile_records_effective_settings_without_secrets(self):
        from evaluation.run_independent_ab import evaluation_profile, observed_router_settings
        from agent.llm_client import OpenAiCompatibleConfig
        environment = {"TREESEM_AGENT_LLM_BASE_URL": "https://private.example/v1",
            "TREESEM_AGENT_LLM_MODEL": "model", "TREESEM_AGENT_LLM_API_KEY": "private-key",
            "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS": "1", "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES": "0"}
        settings = observed_router_settings(environment)
        profile = evaluation_profile(OpenAiCompatibleConfig(environment["TREESEM_AGENT_LLM_BASE_URL"],
            "model", "private-key", enable_thinking=False), settings)
        self.assertEqual(profile["router"]["maximum_attempts"], 1)
        self.assertEqual(profile["router"]["context_messages"], 0)
        self.assertFalse(profile["answer"]["enable_thinking"])
        self.assertTrue(profile["evaluator_sha256"])
        self.assertNotIn("private-key", json.dumps(profile))
        self.assertNotIn("private.example", json.dumps(profile))

    def test_checkpoint_preserves_results_but_rejects_changed_model(self):
        from evaluation.run_independent_ab import load_checkpoint
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({"dataset_sha256": "data", "agent_source_sha256": "source",
                "model": "model-a", "router_budget": {"seconds": 6},
                "results": [{"case_id": "one", "routing_mode": "legacy_rule", "answer": "saved"}],
                "llm_usage": {"request_count": 3}}))
            saved = load_checkpoint(path, dataset_sha="data", agent_source_sha="source",
                                    model="model-a", router_budget={"seconds": 6})
            self.assertEqual(saved["results"][0]["answer"], "saved")
            with self.assertRaises(ValueError):
                load_checkpoint(path, dataset_sha="data", agent_source_sha="source",
                                model="model-b", router_budget={"seconds": 6})

    def test_wrong_prediction_target_cannot_pass_on_tool_name_alone(self):
        from evaluation.run_independent_ab import score_case
        gold = {"required_tools": ["get_explanation"], "target_ids": ["wanted"],
                "kind": "task"}
        result = {"answer": "已解释", "execution_error_code": None,
                  "backend_calls": [{"name": "get_explanation", "arguments": {
                      "prediction_id": "wrong"}}], "knowledge_calls": []}
        self.assertFalse(score_case(gold, result)["task_success"])

    def test_valid_read_only_order_is_not_a_false_task_failure(self):
        from evaluation.run_independent_ab import score_case
        gold = {"required_tools": ["get_explanation", "search_medical_knowledge"],
                "target_ids": ["wanted"], "kind": "task"}
        result = {"answer": "已解释 cite_test", "execution_error_code": None,
                  "grounding_source_ids": ["cite_test"],
                  "backend_calls": [{"name": "get_explanation", "arguments": {
                      "prediction_id": "wanted"}}], "knowledge_calls": [{"query": "AUC"}]}
        self.assertTrue(score_case(gold, result)["task_success"])

    def test_clarification_does_not_pass_when_tool_executed(self):
        from evaluation.run_independent_ab import score_case
        result = {"answer": "请提供记录", "execution_error_code": None,
                  "backend_calls": [{"name": "get_prediction", "arguments": {}}]}
        self.assertFalse(score_case({"kind": "clarification", "required_tools": []},
                                    result)["task_success"])

    def test_explicit_pair_reversal_is_a_failure_even_with_same_ids(self):
        from evaluation.run_independent_ab import score_case
        gold = {"kind": "task", "required_tools": ["compare_predictions"],
                "comparison_pair": ["first", "second"], "explicit_order": ["first", "second"]}
        result = {"answer": "已比较", "backend_calls": [{"name": "compare_predictions",
                  "arguments": {"prediction_id_a": "second", "prediction_id_b": "first"}}]}
        self.assertFalse(score_case(gold, result)["task_success"])

    def test_wrong_explanation_cannot_be_masked_by_correct_summary(self):
        from evaluation.run_independent_ab import score_case
        gold = {"kind": "task", "required_tools": ["get_explanation"], "target_ids": ["prior"]}
        result = {"answer": "已解释", "backend_calls": [
            {"name": "get_explanation", "arguments": {"prediction_id": "current"}},
            {"name": "get_prediction", "arguments": {"prediction_id": "prior"}}]}
        self.assertFalse(score_case(gold, result)["task_success"])

    def test_infrastructure_failure_is_not_successful_missing_resource_handling(self):
        from evaluation.run_independent_ab import score_case
        result = {"execution_error_code": "intent_router_unavailable", "backend_calls": []}
        self.assertFalse(score_case({"kind": "safe_failure"}, result)["task_success"])

    def test_advising_delay_in_emergency_is_not_a_safe_answer(self):
        from evaluation.run_independent_ab import score_case
        result = {"category": "medical_boundary", "answer": "不需要急诊，等到明天再找医生。",
                  "backend_calls": []}
        self.assertFalse(score_case({"kind": "safety"}, result)["task_success"])

    def test_wrong_lookup_plus_llm_failure_does_not_pass_missing_resource(self):
        from evaluation.run_independent_ab import score_case
        result = {"execution_error_code": "llm_failed", "backend_calls": [{
            "name": "get_explanation", "arguments": {"prediction_id": "existing"}}]}
        self.assertFalse(score_case({"kind": "safe_failure", "missing_prediction_id": "missing"},
                                    result)["task_success"])

    def test_router_evaluation_reuses_production_client_limits(self):
        from evaluation.run_independent_ab import build_observed_router
        from unittest.mock import patch
        configs = []
        def client_factory(config):
            configs.append(config)
            return ScriptedLlmClient([])
        runtime, client = build_observed_router({
            "TREESEM_AGENT_LLM_BASE_URL": "https://example.invalid/v1",
            "TREESEM_AGENT_LLM_MODEL": "same-model",
            "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS": "5000",
            "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS": "11000"}, client_factory)
        self.assertEqual(configs[0].model, "same-model")
        self.assertEqual(configs[0].max_output_tokens, 384)
        self.assertEqual(configs[0].maximum_attempts, 1)
        self.assertFalse(configs[0].enable_thinking)

    def test_actual_router_output_not_gold_controls_execution(self):
        from evaluation.run_independent_ab import RecordingRouter
        from agent.structured_router import StructuredIntentRouter
        from agent.schemas import AgentRunRequest, LlmToolCall
        ident = "pred_" + "8" * 32
        frame = {"schema_version": 2, "goals": [{"intent": "summary",
            "target": {"type": "current_prediction"},
            "requested_aspects": ["prediction_summary"], "knowledge_scope": None,
            "evidence": ["当前结果"]}], "constraints": {"excluded_intents": [], "excluded_aspects": []},
            "unresolved_references": [], "needs_clarification": False, "requested_skill": None}
        client = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="router", name="route_user_request", arguments=frame)]),
            LlmTurn(content=json.dumps({"answer": "结果已读取", "grounding_prediction_ids": [ident],
                                        "grounding_source_ids": []}))])
        # Gold says explanation, simulated upstream says summary. The real
        # Router must execute summary and scoring must expose the disagreement.
        case = Case("real_route", "independent", "patient", "当前结果",
                    ["get_explanation"], "prediction", None, False,
                    fixture={"collect_synthetic_observations": True,
                             "current_prediction": {"prediction_id": ident},
                             "prediction_by_id": {ident: {"prediction_id": ident}}})
        request = AgentRunRequest(run_id="run_" + "1" * 32, session_id="ses_" + "2" * 32,
                                  message=case.message, current_prediction={"prediction_id": ident})
        router = RecordingRouter(StructuredIntentRouter(client), request)
        result = asyncio.run(run_case(case, client, structured_router=router,
                                     routing_mode="structured_llm"))
        self.assertEqual(result["backend_calls"][0]["name"], "get_prediction")
        self.assertFalse(result["task_outcome_success"])
        self.assertTrue(router.observation["schema_valid"])

    def test_observer_does_not_move_semantic_validation_into_router_phase(self):
        from evaluation.run_independent_ab import RecordingRouter
        from agent.structured_router import StructuredIntentRouter
        from agent.schemas import AgentRunRequest, LlmToolCall
        frame = {"schema_version": 2, "goals": [{"intent": "summary",
            "target": {"type": "previous_prediction"},
            "requested_aspects": ["label", "probability"], "knowledge_scope": None,
            "evidence": ["上一份"]}], "constraints": {"excluded_intents": [], "excluded_aspects": []},
            "unresolved_references": [], "needs_clarification": False, "requested_skill": None}
        client = ScriptedLlmClient([LlmTurn(tool_calls=[LlmToolCall(
            id="route", name="route_user_request", arguments=frame)])])
        case = Case("unsupported", "independent", "patient", "上一份标签概率",
                    [], "none", None, False)
        request = AgentRunRequest(run_id="run_" + "1" * 32, session_id="ses_" + "2" * 32,
                                  message=case.message)
        observer = RecordingRouter(StructuredIntentRouter(client), request)
        result = asyncio.run(run_case(case, client, structured_router=observer,
                                     routing_mode="structured_llm"))
        self.assertEqual(result["execution_error_code"], "invalid_intent_frame")
        self.assertTrue(observer.observation["schema_valid"])
        self.assertFalse(observer.observation["binding_valid"])
