import json
import tempfile
import unittest
from pathlib import Path

from evaluation.prepare_routing_quality_candidates import (
    BATCH_SPECS,
    normalized_message,
    prepare_candidates,
    validate_batch,
)


def case(message, scope="prediction", category="known",
         style="paraphrase", language="zh"):
    return {
        "message": message,
        "expected_scope": scope,
        "category": category,
        "style": style,
        "language": language,
        "rationale": "用于测试的简短审核理由",
    }


class RoutingQualityCandidateTest(unittest.TestCase):
    def test_normalization_handles_unicode_case_spacing_and_punctuation(self):
        self.assertEqual(
            normalized_message(" Ｐredict,  SAMPLE！ "),
            normalized_message("predict sample"),
        )

    def test_batch_plan_locks_total_and_scope_distribution(self):
        self.assertEqual(sum(item.count for item in BATCH_SPECS), 396)
        self.assertEqual(len(BATCH_SPECS), 12)
        known = [item for item in BATCH_SPECS if item.category == "known"]
        self.assertEqual(len(known), 6)
        self.assertTrue(all(item.count == 36 for item in known))

    def test_validate_batch_rejects_wrong_scope_and_extra_fields(self):
        spec = BATCH_SPECS[0]
        styles = [value for value, count in spec.style_counts.items()
                  for _ in range(count)]
        languages = [value for value, count in spec.language_counts.items()
                     for _ in range(count)]
        rows = [case(f"独立消息{i}", style=styles[i], language=languages[i])
                for i in range(spec.count)]
        payload = {"schema_version": 1, "batch_id": spec.batch_id,
                   "cases": rows}
        validate_batch(payload, spec)

        payload["cases"][0]["expected_scope"] = "summary"
        with self.assertRaisesRegex(ValueError, "expected_scope"):
            validate_batch(payload, spec)
        payload["cases"][0]["expected_scope"] = "prediction"
        payload["cases"][0]["split"] = "held_out"
        with self.assertRaisesRegex(ValueError, "fields"):
            validate_batch(payload, spec)

    def test_prepare_rejects_duplicate_or_frozen_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            frozen_cases = []
            for batch_index, spec in enumerate(BATCH_SPECS):
                styles = [
                    value for value, count in spec.style_counts.items()
                    for _ in range(count)]
                languages = [
                    value for value, count in spec.language_counts.items()
                    for _ in range(count)]
                rows = []
                for index in range(spec.count):
                    rows.append(case(
                        f"批次{batch_index}消息{index}", spec.expected_scope,
                        spec.category, styles[index], languages[index]))
                (raw / f"{spec.batch_id}.json").write_text(json.dumps({
                    "schema_version": 1, "batch_id": spec.batch_id,
                    "cases": rows,
                }, ensure_ascii=False), encoding="utf-8")
            frozen_cases.append({"message": "完全不同的旧消息"})
            frozen = root / "frozen.json"
            frozen.write_text(json.dumps({
                "schema_version": 1, "cases": frozen_cases,
            }, ensure_ascii=False), encoding="utf-8")

            result, report = prepare_candidates(raw, frozen)
            self.assertEqual(len(result), 396)
            self.assertEqual(result[0]["candidate_id"], "rq_0001")
            self.assertNotIn("split", result[0])
            self.assertEqual(report["candidate_count"], 396)

            first = raw / f"{BATCH_SPECS[0].batch_id}.json"
            payload = json.loads(first.read_text(encoding="utf-8"))
            payload["cases"][1]["message"] = payload["cases"][0]["message"]
            first.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                prepare_candidates(raw, frozen)


if __name__ == "__main__":
    unittest.main()
