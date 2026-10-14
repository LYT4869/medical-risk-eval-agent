import unittest
from collections import Counter

from evaluation.freeze_routing_quality_set import freeze_quality_set


def build_candidates():
    result = []
    index = 0
    for scope in (
            "prediction", "summary", "explanation", "history", "comparison",
            "knowledge"):
        for item in range(36):
            index += 1
            result.append({
                "candidate_id": f"rq_{index:04d}",
                "message": f"{scope} message {item}",
                "expected_scope": scope,
                "category": "known",
                "style": "paraphrase",
                "language": "en",
                "rationale": "test",
            })
    for category, scope, count in (
            ("unknown", "unknown", 60),
            ("compositional", "unknown", 60),
            ("safety", "security_abuse", 30),
            ("safety", "medical_refusal", 30)):
        for item in range(count):
            index += 1
            result.append({
                "candidate_id": f"rq_{index:04d}",
                "message": f"{category} {scope} message {item}",
                "expected_scope": scope,
                "category": category,
                "style": "adversarial",
                "language": "en",
                "rationale": "test",
            })
    return result


class RoutingQualityFreezeTest(unittest.TestCase):
    def test_freezes_stratified_360_case_corpus(self):
        candidates = build_candidates()
        excluded = []
        for scope in (
                "prediction", "summary", "explanation", "history",
                "comparison", "knowledge"):
            excluded.extend(list(
                item["candidate_id"] for item in candidates
                if item["expected_scope"] == scope)[:6])

        result, report = freeze_quality_set(candidates, set(excluded))

        self.assertEqual(len(result), 360)
        self.assertEqual(Counter(item["split"] for item in result), {
            "calibration": 120, "held_out": 240})
        for scope in (
                "prediction", "summary", "explanation", "history",
                "comparison", "knowledge"):
            rows = [item for item in result if item["category"] == "known"
                    and item["expected_scope"] == scope]
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(item["split"] == "calibration"
                                 for item in rows), 10)
        self.assertEqual(report["excluded_count"], 36)
        self.assertEqual(set(result[0]), {
            "case_id", "split", "category", "message", "expected_scope"})

    def test_rejects_exclusion_from_hard_gate_categories(self):
        candidates = build_candidates()
        excluded = []
        for scope in (
                "prediction", "summary", "explanation", "history",
                "comparison", "knowledge"):
            excluded.extend(list(
                item["candidate_id"] for item in candidates
                if item["expected_scope"] == scope)[:6])
        excluded[-1] = "rq_0217"
        with self.assertRaisesRegex(ValueError, "known"):
            freeze_quality_set(candidates, set(excluded))


if __name__ == "__main__":
    unittest.main()
