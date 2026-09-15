import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.run_routing_evaluation import (
    RoutingCase,
    _hybrid_scope,
    _rule,
    create_evaluation_provider,
    evaluate_cases,
    load_cases,
    normalized_message,
)
from agent.routing import RuleRouter, SafetyGate
from agent.routing_types import RequestScope, RoutingDecision, RoutingSource
from agent.semantic_routing import RoutingThresholds, SemanticScores


ROOT = Path(__file__).resolve().parents[1]
ROUTING_CASES = ROOT / "evaluation" / "routing_cases.json"


class RoutingCorpusTest(unittest.TestCase):
    def test_hybrid_adapter_preserves_rules_and_applies_semantic_thresholds(self):
        class Scorer:
            def score(self, message):
                return SemanticScores(RequestScope.PREDICTION, 0.9, 0.4, 0.5)

            def route(self, message):
                return RoutingDecision(
                    RequestScope.PREDICTION, RoutingSource.SEMANTIC)

        cases = load_cases(ROUTING_CASES)
        by_id = {case.case_id: case for case in cases}
        thresholds = RoutingThresholds(0.8, 0.1, 0.7)
        self.assertEqual(_hybrid_scope(
            by_id["known_history_01"], Scorer(),
            RuleRouter(), SafetyGate()), "history")
        self.assertEqual(_hybrid_scope(
            by_id["known_prediction_06"], Scorer(),
            RuleRouter(), SafetyGate()), "prediction")

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

    def test_report_exposes_selective_routing_failure_modes(self):
        cases = [
            RoutingCase("known_correct", "held_out", "known", "a", "prediction"),
            RoutingCase("known_abstain", "held_out", "known", "b", "history"),
            RoutingCase("known_misroute", "held_out", "known", "c", "summary"),
            RoutingCase(
                "known_comparison", "held_out", "known", "h", "comparison"),
            RoutingCase("unknown_forced", "held_out", "unknown", "d", "unknown"),
            RoutingCase("unknown_rejected", "held_out", "unknown", "e", "unknown"),
            RoutingCase(
                "compositional_forced", "held_out", "compositional", "f", "unknown"),
            RoutingCase(
                "compositional_rejected", "held_out", "compositional", "g", "unknown"),
        ]
        actual = {
            "known_correct": "prediction",
            "known_abstain": "unknown",
            "known_misroute": "comparison",
            "known_comparison": "comparison",
            "unknown_forced": "knowledge",
            "unknown_rejected": "unknown",
            "compositional_forced": "comparison",
            "compositional_rejected": "unknown",
        }

        report = evaluate_cases(cases, lambda case: actual[case.case_id])

        self.assertAlmostEqual(report["deterministic_precision"], 0.4)
        self.assertAlmostEqual(report["deterministic_coverage"], 5 / 8)
        self.assertAlmostEqual(report["known_abstention_rate"], 0.25)
        self.assertAlmostEqual(report["known_misroute_rate"], 0.25)
        self.assertAlmostEqual(report["unknown_forced_route_rate"], 0.5)
        self.assertAlmostEqual(report["compositional_forced_route_rate"], 0.5)
        self.assertEqual(
            report["per_scope_precision_recall"]["prediction"],
            {"precision": 1.0, "recall": 1.0},
        )
        self.assertEqual(
            report["per_scope_precision_recall"]["summary"],
            {"precision": 0.0, "recall": 0.0},
        )
        self.assertEqual(
            report["per_scope_precision_recall"]["comparison"],
            {"precision": 1 / 3, "recall": 1.0},
        )
        self.assertEqual(
            report["per_scope_precision_recall"]["knowledge"],
            {"precision": 0.0, "recall": 0.0},
        )
        self.assertEqual(report["rule_precision"], report["deterministic_precision"])
        self.assertEqual(report["rule_coverage"], report["deterministic_coverage"])

    def test_evaluation_provider_selects_sentence_transformer_backend(self):
        marker = object()
        with mock.patch(
                "agent.embedding_provider.SentenceTransformerEmbeddingProvider",
                return_value=marker) as factory:
            actual = create_evaluation_provider(
                "sentence_transformers", model="model", revision="revision",
                artifact_dir=None, tasks_path=Path("tasks"),
                thresholds_path=Path("thresholds"), registry=object())
        self.assertIs(actual, marker)
        factory.assert_called_once_with("model", "revision")

    def test_evaluation_onnx_backend_requires_artifact(self):
        with self.assertRaisesRegex(ValueError, "artifact"):
            create_evaluation_provider(
                "onnx_fp32", model="model", revision="revision",
                artifact_dir=None, tasks_path=Path("tasks"),
                thresholds_path=Path("thresholds"), registry=object())


if __name__ == "__main__":
    unittest.main()
