from __future__ import annotations

import math
import unittest

from evaluation.compare_routing_backends import (
    BackendSnapshot,
    CaseObservation,
    compare_snapshots,
)
from evaluation.run_routing_evaluation import RoutingCase


def case(case_id, category, expected):
    return RoutingCase(
        case_id=case_id,
        split="held_out",
        category=category,
        message=f"message {case_id}",
        expected_scope=expected,
    )


def observation(case_id, scope, embedding=(1.0, 0.0),
                top=0.8, margin=0.2):
    return CaseObservation(
        case_id=case_id,
        route_scope=scope,
        query_embedding=embedding,
        top_similarity=top,
        margin=margin,
    )


class RoutingBackendParityTest(unittest.TestCase):
    def setUp(self):
        self.cases = [
            case("known", "known", "history"),
            case("unknown", "unknown", "unknown"),
            case("composition", "compositional", "unknown"),
            case("safety", "safety", "medical_refusal"),
        ]
        self.golden = BackendSnapshot(
            intent_embeddings=((1.0, 0.0), (0.0, 1.0)),
            cases=(
                observation("known", "history"),
                observation("unknown", "unknown", (0.0, 1.0)),
                observation("composition", "unknown", (0.6, 0.8)),
                observation("safety", "medical_refusal", (0.8, 0.6),
                            top=None, margin=None),
            ),
        )

    def compare(self, candidate, **kwargs):
        return compare_snapshots(
            self.cases, self.golden, candidate,
            embedding_backend="onnx_fp32",
            artifact_version="routing-test",
            required_case_count=4,
            maximum_embedding_delta=1e-4,
            minimum_embedding_cosine=0.9999,
            **kwargs,
        )

    def test_identical_routes_and_vectors_pass(self):
        report = self.compare(self.golden)

        self.assertTrue(report["passed"])
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["route_match_count"], 4)
        self.assertEqual(report["route_mismatch_count"], 0)
        self.assertEqual(report["route_mismatches"], [])
        self.assertEqual(report["maximum_embedding_absolute_delta"], 0.0)
        self.assertAlmostEqual(report["minimum_embedding_cosine_similarity"],
                               1.0)

    def test_any_route_change_fails_and_names_case(self):
        changed = BackendSnapshot(
            intent_embeddings=self.golden.intent_embeddings,
            cases=tuple(
                observation(item.case_id,
                            "history" if item.case_id == "unknown"
                            else item.route_scope,
                            item.query_embedding,
                            item.top_similarity,
                            item.margin)
                for item in self.golden.cases),
        )

        report = self.compare(changed)

        self.assertFalse(report["passed"])
        self.assertEqual(report["route_mismatch_count"], 1)
        self.assertEqual(report["route_mismatches"], [{
            "case_id": "unknown",
            "golden_scope": "unknown",
            "candidate_scope": "history",
        }])
        self.assertEqual(report["quality_metrics"]["unknown_recall"], 0.0)

    def test_vector_delta_gate_is_independent_from_route_gate(self):
        changed = BackendSnapshot(
            intent_embeddings=((0.8, 0.6), (0.0, 1.0)),
            cases=self.golden.cases,
        )

        report = self.compare(changed)

        self.assertFalse(report["passed"])
        self.assertEqual(report["route_mismatch_count"], 0)
        self.assertGreater(report["maximum_embedding_absolute_delta"], 1e-4)

    def test_score_deltas_are_reported_for_semantic_cases_only(self):
        candidate = BackendSnapshot(
            intent_embeddings=self.golden.intent_embeddings,
            cases=tuple(
                observation(item.case_id, item.route_scope,
                            item.query_embedding,
                            None if item.top_similarity is None
                            else item.top_similarity + 0.01,
                            None if item.margin is None
                            else item.margin + 0.02)
                for item in self.golden.cases),
        )

        report = self.compare(candidate)

        self.assertAlmostEqual(report["maximum_top_similarity_delta"], 0.01)
        self.assertAlmostEqual(report["maximum_margin_delta"], 0.02)

    def test_rejects_misaligned_or_nonfinite_snapshots(self):
        missing = BackendSnapshot(
            intent_embeddings=self.golden.intent_embeddings,
            cases=self.golden.cases[:-1])
        with self.assertRaisesRegex(ValueError, "case alignment"):
            self.compare(missing)

        nonfinite = BackendSnapshot(
            intent_embeddings=((math.nan, 0.0), (0.0, 1.0)),
            cases=self.golden.cases)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.compare(nonfinite)

    def test_requires_frozen_case_count(self):
        with self.assertRaisesRegex(ValueError, "frozen parity set"):
            compare_snapshots(
                self.cases, self.golden, self.golden,
                embedding_backend="onnx_fp32",
                artifact_version="routing-test",
                required_case_count=150,
                maximum_embedding_delta=1e-4,
                minimum_embedding_cosine=0.9999)


if __name__ == "__main__":
    unittest.main()
