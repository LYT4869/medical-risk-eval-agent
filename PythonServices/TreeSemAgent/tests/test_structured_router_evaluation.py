from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
