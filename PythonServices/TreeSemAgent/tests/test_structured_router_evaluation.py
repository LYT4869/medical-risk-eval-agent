from __future__ import annotations

from dataclasses import replace
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.intent_dispatch import IntentDispatcher
from agent.intent_frame import IntentFrame
from agent.intent_validation import validate_and_bind_intent
from agent.reference_extractor import extract_references
from agent.schemas import AgentRunRequest, PredictionContext


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evaluation" / "structured_router_cases.json"
RUNNER = ROOT / "evaluation" / "run_structured_router_evaluation.py"


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "treesem_structured_router_evaluation", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class StructuredRouterCorpusTest(unittest.TestCase):
    def test_corpus_has_frozen_splits_unique_messages_and_critical_coverage(self):
        module = load_runner()
        cases, _ = module.load_cases(CASES)

        self.assertEqual(len(cases), 120)
        self.assertEqual(
            {split: sum(case.split == split for case in cases)
             for split in ("dev", "validation", "smoke_heldout")},
            {"dev": 60, "validation": 30, "smoke_heldout": 30})
        normalized = [module.normalize_message(case.message)
                      for case in cases]
        self.assertEqual(len(normalized), len(set(normalized)))
        for split in ("validation", "smoke_heldout"):
            present = {item for case in cases if case.split == split
                       for item in case.slice}
            self.assertTrue({
                "security", "generated_id", "unauthorized_tool",
                "urgent_medical",
            } <= present)

    def test_corpus_contains_no_concrete_patient_or_credential_data(self):
        module = load_runner()
        cases, _ = module.load_cases(CASES)
        serialized = "\n".join(case.message for case in cases).lower()
        for forbidden in (
                "authorization:", "bearer eyj", "treeSemRefresh=",
                "patient@example.com", "身份证", "手机号"):
            self.assertNotIn(forbidden.lower(), serialized)

    def test_dev_labels_follow_the_literal_user_request(self):
        module = load_runner()
        loaded, _ = module.load_cases(CASES)
        cases = {case.case_id: case for case in loaded}

        self.assertIn("决策路径", cases["sr_dev_014"].message)
        self.assertEqual(cases["sr_dev_014"].expected_frame.required_aspects,
                         ("decision_path",))
        self.assertIn("结果摘要", cases["sr_dev_036"].message)
        self.assertEqual(cases["sr_dev_036"].expected_frame.excluded_aspects,
                         ("prediction_summary",))
        self.assertIn("摘要", cases["sr_dev_037"].message)
        self.assertEqual(
            cases["sr_dev_037"].expected_frame.intents,
            ("summary",))
        self.assertEqual(
            cases["sr_dev_037"].expected_recipe,
            "read_current_or_explicit_prediction")
        self.assertEqual(
            cases["sr_dev_037"].expected_frame.excluded_intents,
            ("comparison",))
        self.assertIn("检索", cases["sr_dev_057"].message)
        self.assertEqual(
            cases["sr_dev_057"].expected_frame.intents,
            ("knowledge",))
        self.assertEqual(cases["sr_dev_057"].expected_dispatch, "workflow")

    def test_skill_labels_keep_the_business_intent_separate(self):
        module = load_runner()
        loaded, _ = module.load_cases(CASES)
        cases = {case.case_id: case for case in loaded}

        self.assertEqual(
            cases["sr_dev_019"].expected_frame.intents,
            ("explanation",))
        self.assertEqual(
            cases["sr_dev_019"].expected_frame.requested_skill,
            "explain_prediction")
        self.assertEqual(
            cases["sr_dev_021"].expected_frame.intents,
            ("comparison",))
        self.assertEqual(
            cases["sr_dev_023"].expected_frame.intents,
            ("knowledge",))

    def test_knowledge_labels_do_not_make_citation_a_router_aspect(self):
        module = load_runner()
        loaded, _ = module.load_cases(CASES)
        knowledge = [case for case in loaded
                     if "knowledge" in case.expected_frame.intents]

        self.assertTrue(knowledge)
        self.assertTrue(all(
            "citations" not in case.expected_frame.required_aspects
            for case in knowledge))


class StructuredRouterScoringTest(unittest.TestCase):
    def setUp(self):
        self.module = load_runner()
        self.case = self.module.load_cases(CASES)[0][0]

    def expected(self):
        return self.module.LayerOutcome.from_expected(self.case)

    def test_wrong_target_only_lowers_router_target_metric(self):
        outcome = self.expected()
        outcome = self.module.LayerOutcome(
            **{**outcome.__dict__, "target_types": ("none",)})
        report = self.module.score_outcomes([self.case], [outcome])

        self.assertEqual(report["router"]["schema_valid_rate"], 1.0)
        self.assertEqual(report["router"]["intent_accuracy"], 1.0)
        self.assertEqual(report["router"]["target_accuracy"], 0.0)
        self.assertEqual(report["planner"]["dispatch_accuracy"], 1.0)
        self.assertEqual(report["failures"][0]["layers"], ["router_target"])

    def test_planner_mistake_does_not_change_router_metrics(self):
        outcome = self.expected()
        outcome = self.module.LayerOutcome(
            **{**outcome.__dict__, "dispatch": "open_agent",
               "recipe": None})
        report = self.module.score_outcomes([self.case], [outcome])

        self.assertEqual(report["router"]["target_accuracy"], 1.0)
        self.assertEqual(report["planner"]["dispatch_accuracy"], 0.0)
        self.assertEqual(report["planner"]["workflow_mapping_accuracy"], 0.0)

    def test_blocked_tool_attempt_only_affects_end_to_end_safety(self):
        outcome = self.expected()
        outcome = self.module.LayerOutcome(
            **{**outcome.__dict__, "unauthorized_tool_execution": 1,
               "task_success": False})
        report = self.module.score_outcomes([self.case], [outcome])

        self.assertEqual(report["router"]["intent_accuracy"], 1.0)
        self.assertEqual(report["planner"]["dispatch_accuracy"], 1.0)
        self.assertEqual(
            report["end_to_end"]["unauthorized_tool_execution_count"], 1)
        self.assertEqual(report["end_to_end"]["task_success_rate"], 0.0)

    def test_protocol_error_is_reported_without_user_text(self):
        outcome = self.expected()
        outcome = self.module.LayerOutcome(
            **{**outcome.__dict__, "schema_valid": False,
               "error_code": "invalid_intent_frame",
               "task_success": False})
        report = self.module.score_outcomes([self.case], [outcome])

        self.assertEqual(
            report["failures"][0]["error_code"], "invalid_intent_frame")
        self.assertNotIn(self.case.message, repr(report["failures"]))

    def test_constraint_accuracy_includes_excluded_intents(self):
        expected_frame = replace(
            self.case.expected_frame, excluded_intents=("comparison",))
        case = replace(self.case, expected_frame=expected_frame)
        outcome = self.module.LayerOutcome.from_expected(case)
        outcome = replace(outcome, excluded_intents=())

        report = self.module.score_outcomes([case], [outcome])

        self.assertEqual(report["router"]["constraint_accuracy"], 0.0)
        self.assertIn("router_constraint", report["failures"][0]["layers"])

    def test_wrong_skill_preference_only_lowers_router_skill_metric(self):
        expected_frame = replace(
            self.case.expected_frame,
            requested_skill="explain_prediction")
        case = replace(self.case, expected_frame=expected_frame)
        outcome = self.module.LayerOutcome.from_expected(case)
        outcome = replace(outcome, requested_skill=None)

        report = self.module.score_outcomes([case], [outcome])

        self.assertEqual(report["router"]["intent_accuracy"], 1.0)
        self.assertEqual(report["router"]["target_accuracy"], 1.0)
        self.assertEqual(report["router"]["skill_accuracy"], 0.0)
        self.assertIn("router_skill", report["failures"][0]["layers"])

    def test_validator_correction_preserves_end_to_end_task_success(self):
        cases = {case.case_id: case for case in
                 self.module.load_cases(CASES)[0]}
        case = cases["sr_dev_007"]
        value = IntentFrame.model_validate({
            "schema_version": 2,
            "goals": [{
                "intent": "comparison",
                "target": {"type": "latest_two_predictions"},
                "requested_aspects": ["comparison_changes"],
                "knowledge_scope": None,
                "evidence": ["比较"],
            }],
            "constraints": {
                "excluded_intents": [], "excluded_aspects": []},
            "unresolved_references": [],
            "needs_clarification": False,
            "requested_skill": "compare_prediction_history",
        })
        request = AgentRunRequest(
            run_id="run_" + "1" * 32,
            session_id="ses_" + "2" * 32,
            message=case.message,
            current_prediction=PredictionContext(
                prediction_id="pred_" + "a" * 32,
                model_version="evaluation"),
        )
        references = extract_references(case.message)
        validated = validate_and_bind_intent(value, references, request)
        dispatch = IntentDispatcher().dispatch(validated)

        outcome = self.module._frame_outcome(
            case, SimpleNamespace(frame=value, usage=None, attempt_count=1),
            dispatch, 1.0)

        self.assertEqual(outcome.requested_skill,
                         "compare_prediction_history")
        self.assertTrue(outcome.task_success)

    def test_unresolved_empty_goal_counts_as_successful_clarification(self):
        cases = {case.case_id: case for case in
                 self.module.load_cases(CASES)[0]}
        case = cases["sr_dev_031"]
        value = IntentFrame.model_validate({
            "schema_version": 2,
            "goals": [],
            "constraints": {
                "excluded_intents": [], "excluded_aspects": []},
            "unresolved_references": ["missing_prediction_target"],
            "needs_clarification": True,
            "requested_skill": None,
        })
        request = AgentRunRequest(
            run_id="run_" + "1" * 32,
            session_id="ses_" + "2" * 32,
            message=case.message,
        )
        references = extract_references(case.message)
        validated = validate_and_bind_intent(value, references, request)
        dispatch = IntentDispatcher().dispatch(validated)

        outcome = self.module._frame_outcome(
            case, SimpleNamespace(frame=value, usage=None, attempt_count=1),
            dispatch, 1.0)

        self.assertEqual(outcome.dispatch, "clarification")
        self.assertTrue(outcome.task_success)

    def test_reports_clarification_and_critical_safety_gates(self):
        cases, _ = self.module.load_cases(CASES)
        clarification = next(
            case for case in cases
            if case.expected_dispatch == "clarification")
        critical = next(case for case in cases if case.critical)

        report = self.module.score_outcomes(
            [clarification, critical], [
                self.module.LayerOutcome.from_expected(clarification),
                self.module.LayerOutcome.from_expected(critical),
            ])

        self.assertEqual(report["planner"]["clarification_accuracy"], 1.0)
        self.assertEqual(
            report["end_to_end"]["critical_safety_pass_rate"], 1.0)
        self.assertEqual(report["end_to_end"]["workflow_rate"], 0.0)
        self.assertEqual(report["end_to_end"]["open_agent_rate"], 0.0)

    def test_fake_command_gate_rejects_any_deterministic_regression(self):
        cases = [case for case in self.module.load_cases(CASES)[0]
                 if case.split == "dev"]
        passing = self.module.score_outcomes(
            cases, [self.module.LayerOutcome.from_expected(case)
                    for case in cases])
        passing.update({"status": "completed", "mode": "fake"})
        failing = {**passing, "failures": [{"case_id": cases[0].case_id}]}

        self.assertTrue(self.module.report_passes_command_gate(passing))
        self.assertFalse(self.module.report_passes_command_gate(failing))


if __name__ == "__main__":
    unittest.main()
