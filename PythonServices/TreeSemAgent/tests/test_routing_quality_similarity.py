import unittest

from evaluation.scan_routing_quality_similarity import (
    cosine_similarity,
    find_similarity_flags,
)


class FakeProvider:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode_query(self, text):
        return self.vectors[text]


class RoutingQualitySimilarityTest(unittest.TestCase):
    def test_cosine_similarity_uses_normalized_dot_product(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.8, 0.6]), 0.8)

    def test_flags_best_candidate_or_frozen_neighbor_without_messages(self):
        candidates = [
            {"candidate_id": "rq_0001", "message": "new one",
             "expected_scope": "prediction"},
            {"candidate_id": "rq_0002", "message": "other one",
             "expected_scope": "history"},
        ]
        frozen = [{"case_id": "old_1", "message": "old one",
                   "expected_scope": "prediction"}]
        provider = FakeProvider({
            "new one": [1.0, 0.0],
            "other one": [0.8, 0.6],
            "old one": [0.99, 0.1410673598],
        })

        flags = find_similarity_flags(
            candidates, frozen, provider, threshold=0.90)

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["candidate_id"], "rq_0001")
        self.assertEqual(flags[0]["neighbor_id"], "old_1")
        self.assertEqual(flags[0]["neighbor_set"], "frozen")
        self.assertTrue(flags[0]["same_scope"])
        self.assertNotIn("message", flags[0])


if __name__ == "__main__":
    unittest.main()
