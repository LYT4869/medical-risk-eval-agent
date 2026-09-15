import unittest

from agent.reference_extractor import (
    extract_references,
    sanitize_router_context_text,
)


class ReferenceExtractorTest(unittest.TestCase):
    def test_extracts_and_masks_prediction_ids_in_source_order(self):
        first = "pred_" + "a" * 32
        second = "pred_" + "b" * 32

        result = extract_references(f"比较 {first} 和 {second}")

        self.assertEqual(result.prediction_ids, (first, second))
        self.assertEqual(
            result.router_message,
            "比较 <prediction_ref_0> 和 <prediction_ref_1>")

    def test_duplicate_prediction_id_reuses_the_same_candidate(self):
        prediction_id = "pred_" + "a" * 32

        result = extract_references(
            f"先看 {prediction_id}，再解释 {prediction_id}")

        self.assertEqual(result.prediction_ids, (prediction_id,))
        self.assertEqual(result.router_message.count("<prediction_ref_0>"), 2)

    def test_ignores_uppercase_short_and_overlong_prediction_ids(self):
        values = [
            "pred_" + "A" * 32,
            "pred_" + "a" * 31,
            "pred_" + "b" * 33,
        ]

        result = extract_references(" ".join(values))

        self.assertEqual(result.prediction_ids, ())
        self.assertEqual(result.router_message, " ".join(values))

    def test_extracts_only_explicitly_labelled_sample_indexes(self):
        result = extract_references(
            "运行样本索引 12，再试 sample #7；日期 2026，概率 0.8")

        self.assertEqual(result.sample_indexes, (12, 7))
        self.assertIn("样本索引 <sample_ref_0>", result.router_message)
        self.assertIn("sample <sample_ref_1>", result.router_message)
        self.assertIn("2026", result.router_message)
        self.assertIn("0.8", result.router_message)

    def test_duplicate_sample_index_reuses_the_same_candidate(self):
        result = extract_references("样本 3 和 sample 3")

        self.assertEqual(result.sample_indexes, (3,))
        self.assertEqual(result.router_message.count("<sample_ref_0>"), 2)

    def test_extracts_natural_chinese_and_english_demo_sample_forms(self):
        chinese = extract_references("帮我运行第0号演示样本")
        english = extract_references("Run demonstration sample zero")

        self.assertEqual(chinese.sample_indexes, (0,))
        self.assertIn("<sample_ref_0>", chinese.router_message)
        self.assertEqual(english.sample_indexes, (0,))
        self.assertIn("<sample_ref_0>", english.router_message)

    def test_keeps_at_most_eight_candidates(self):
        message = " ".join(f"样本 {index}" for index in range(10))

        result = extract_references(message)

        self.assertEqual(result.sample_indexes, tuple(range(8)))
        self.assertIn("样本 8", result.router_message)
        self.assertIn("样本 9", result.router_message)

    def test_sanitizes_sensitive_references_in_recent_context(self):
        prediction_id = "pred_" + "a" * 32
        session_id = "ses_" + "b" * 32
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c3IifQ.signature123"
        api_key = "sk-" + "c" * 32

        sanitized = sanitize_router_context_text(
            f"{prediction_id} {session_id} Bearer {jwt} {api_key}")

        self.assertNotIn(prediction_id, sanitized)
        self.assertNotIn(session_id, sanitized)
        self.assertNotIn(jwt, sanitized)
        self.assertNotIn(api_key, sanitized)
        self.assertIn("<redacted_prediction_id>", sanitized)
        self.assertIn("<redacted_session_id>", sanitized)
        self.assertIn("<redacted_credential>", sanitized)


if __name__ == "__main__":
    unittest.main()
