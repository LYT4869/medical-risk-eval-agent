import math
import unittest

from agent.routing_types import RequestScope, RoutingSource
from agent.semantic_routing import RoutingThresholds, SemanticScorer
from agent.task_registry import load_default_registry


class FakeEmbeddingProvider:
    def __init__(self, query_vectors: dict[str, list[float]]):
        self.registry = load_default_registry()
        self.query_vectors = query_vectors
        self.example_vectors = {}
        self.encode_calls = 0
        for index, definition in enumerate(self.registry.business_definitions):
            vector = [0.0] * 6
            vector[index] = 1.0
            for example in definition.intent_examples:
                self.example_vectors[example] = vector

    def encode(self, texts):
        self.encode_calls += 1
        return [
            self.query_vectors[text]
            if text in self.query_vectors else self.example_vectors[text]
            for text in texts
        ]


class SemanticScorerTest(unittest.TestCase):
    def setUp(self):
        self.registry = load_default_registry()
        self.thresholds = RoutingThresholds(
            min_similarity=0.60,
            min_margin=0.10,
            secondary_intent_similarity=0.65,
        )

    def scorer(self, vectors):
        provider = FakeEmbeddingProvider(vectors)
        return SemanticScorer(self.registry, provider, self.thresholds), provider

    def test_clear_top_one_match_returns_business_scope(self):
        scorer, _ = self.scorer({"clear": [1, 0, 0, 0, 0, 0]})
        decision = scorer.route("clear")
        expected = self.registry.business_definitions[0].scope
        self.assertEqual(decision.scope, expected)
        self.assertEqual(decision.source, RoutingSource.SEMANTIC)
        self.assertAlmostEqual(decision.similarity_score, 1.0)

    def test_low_similarity_falls_back_to_unknown(self):
        scorer, _ = self.scorer({"weak": [1, 1, 1, 1, 1, 1]})
        decision = scorer.route("weak")
        self.assertEqual(decision.scope, RequestScope.UNKNOWN)
        self.assertEqual(decision.reason, "semantic_low_similarity")

    def test_small_margin_falls_back_to_unknown(self):
        scorer, _ = self.scorer({"ambiguous": [1, 0.95, 0, 0, 0, 0]})
        decision = scorer.route("ambiguous")
        self.assertEqual(decision.scope, RequestScope.UNKNOWN)
        self.assertEqual(decision.reason, "semantic_ambiguous_margin")

    def test_high_secondary_intent_falls_back_to_unknown(self):
        thresholds = RoutingThresholds(0.60, 0.05, 0.65)
        provider = FakeEmbeddingProvider({"multi": [1, 0.90, 0, 0, 0, 0]})
        scorer = SemanticScorer(self.registry, provider, thresholds)
        decision = scorer.route("multi")
        self.assertEqual(decision.scope, RequestScope.UNKNOWN)
        self.assertEqual(decision.reason, "semantic_multiple_intents")

    def test_non_finite_and_wrong_dimension_vectors_are_rejected(self):
        scorer, _ = self.scorer({"nan": [math.nan, 0, 0, 0, 0, 0]})
        with self.assertRaisesRegex(ValueError, "finite"):
            scorer.route("nan")
        scorer, _ = self.scorer({"short": [1, 0]})
        with self.assertRaisesRegex(ValueError, "dimension"):
            scorer.route("short")

    def test_examples_are_embedded_once_into_immutable_cache(self):
        scorer, provider = self.scorer({
            "first": [1, 0, 0, 0, 0, 0],
            "second": [1, 0, 0, 0, 0, 0],
        })
        startup_calls = provider.encode_calls
        scorer.route("first")
        scorer.route("second")
        self.assertEqual(startup_calls, 1)
        self.assertEqual(provider.encode_calls, 3)

    def test_thresholds_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            RoutingThresholds(1.1, 0.1, 0.7)
        with self.assertRaises(ValueError):
            RoutingThresholds(0.6, -0.1, 0.7)


if __name__ == "__main__":
    unittest.main()
