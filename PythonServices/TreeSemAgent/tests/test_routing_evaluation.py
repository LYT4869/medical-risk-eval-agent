import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_routing_evaluation import (
    _rule,
    evaluate_cases,
    load_cases,
    normalized_message,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTING_CASES = ROOT / "evaluation" / "routing_cases.json"


class RoutingCorpusTest(unittest.TestCase):
    def test_rule_adapter_includes_safety_and_rule_miss(self):
        cases = load_cases(ROUTING_CASES)
        by_id = {case.case_id: case for case in cases}
        self.assertEqual(_rule(by_id["safety_01"]), "security_abuse")
        self.assertEqual(_rule(by_id["known_prediction_06"]), "unknown")

    def test_corpus_has_expected_shape_and_disjoint_messages(self):
        cases = load_cases(ROUTING_CASES)

        self.assertEqual(len(cases), 150)
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertEqual(
            len({normalized_message(case.message) for case in cases}),
            len(cases),
        )
        self.assertEqual(
            sum(case.category == "known" for case in cases), 90)
        self.assertEqual(
            sum(case.category == "unknown" for case in cases), 20)
        self.assertEqual(
            sum(case.category == "compositional" for case in cases), 20)
        self.assertEqual(
            sum(case.category == "safety" for case in cases), 20)
        for scope in (
                "prediction", "summary", "explanation", "history",
                "comparison", "knowledge"):
            self.assertEqual(
                sum(case.category == "known" and case.expected_scope == scope
                    for case in cases),
                15,
            )

    def test_loader_rejects_duplicate_normalized_messages(self):
        payload = {
            "schema_version": 1,
            "cases": [
                {"case_id": "a", "split": "held_out", "category": "unknown",
                 "message": "普通 问题", "expected_scope": "unknown"},
                {"case_id": "b", "split": "calibration", "category": "unknown",
                 "message": " 普通   问题 ", "expected_scope": "unknown"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate normalized message"):
                load_cases(path)

    def test_report_separates_safety_from_business_macro_f1(self):
        cases = load_cases(ROUTING_CASES)
        report = evaluate_cases(cases, lambda case: case.expected_scope)

        self.assertEqual(report["case_count"], 150)
        self.assertEqual(report["known_exact_route_accuracy"], 1.0)
        self.assertEqual(report["business_macro_f1"], 1.0)
        self.assertEqual(report["unknown_recall"], 1.0)
        self.assertEqual(report["compositional_fallback_recall"], 1.0)
        self.assertEqual(report["safety_accuracy"], 1.0)
        self.assertIn("p95_route_latency_ms", report)


if __name__ == "__main__":
    unittest.main()
