import sys
import unittest
from unittest.mock import patch

from agent.embedding_provider import SentenceTransformerEmbeddingProvider


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **options):
        self.calls.append((list(texts), options))
        return [[float(index + 1), 0.0] for index in range(len(texts))]


class EmbeddingProviderTest(unittest.TestCase):
    def test_loads_pinned_model_from_local_files_only(self):
        model = FakeModel()
        factory_calls = []

        def factory(model_id, **options):
            factory_calls.append((model_id, options))
            return model

        provider = SentenceTransformerEmbeddingProvider(
            "intfloat/multilingual-e5-small", "abc123", factory=factory)

        self.assertEqual(factory_calls, [(
            "intfloat/multilingual-e5-small",
            {"revision": "abc123", "local_files_only": True},
        )])
        self.assertEqual(provider.encode_examples(["样例一", "example two"]),
                         [[1.0, 0.0], [2.0, 0.0]])
        self.assertEqual(provider.encode_query("问题"), [1.0, 0.0])
        self.assertEqual(model.calls[0][0],
                         ["passage: 样例一", "passage: example two"])
        self.assertEqual(model.calls[1][0], ["query: 问题"])
        self.assertTrue(model.calls[0][1]["normalize_embeddings"])
        self.assertTrue(model.calls[0][1]["convert_to_numpy"])

    def test_model_and_revision_are_required(self):
        with self.assertRaises(ValueError):
            SentenceTransformerEmbeddingProvider("", "revision", factory=lambda *a, **k: None)
        with self.assertRaises(ValueError):
            SentenceTransformerEmbeddingProvider("model", "", factory=lambda *a, **k: None)

    def test_missing_optional_dependency_has_safe_error(self):
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "sentence-transformers"):
                SentenceTransformerEmbeddingProvider("model", "revision")


if __name__ == "__main__":
    unittest.main()
