import sqlite3
import tempfile
import unittest
from pathlib import Path

from knowledge.ingestion.builder import IndexBuilder
from knowledge.retrieval.index import HybridRetriever, KnowledgeIndex


class ModelEncodingSourceTest(unittest.TestCase):
    def test_registered_encoding_is_retrievable_and_citable_for_both_roles(self):
        corpus = Path(__file__).resolve().parents[1] / "corpus"
        records = IndexBuilder._load_manifest(corpus / "sources.json")
        selected = [r for r in records if r.source_id == "src_treesem_label_coding"]
        self.assertEqual(len(selected), 1, "encoding evidence must be registered")
        builder = IndexBuilder.__new__(IndexBuilder)
        builder.target_tokens, builder.overlap_tokens = 400, 60
        chunks = builder._read(selected[0], corpus)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexical.sqlite"
            builder._write_lexical(path, chunks)
            connection = sqlite3.connect(path)

            class LexicalIndex:
                version = "test-encoding"
                by_id = {c.chunk_id: c for c in chunks}
                _lexical = connection

                def lexical(self, query, limit):
                    return KnowledgeIndex.lexical(self, query, limit)

                def dense(self, vector, limit):
                    raise RuntimeError("dense deliberately unavailable")

            class NoEmbedding:
                def encode_query(self, query):
                    raise RuntimeError("embedding deliberately unavailable")

            try:
                retriever = HybridRetriever(LexicalIndex(), NoEmbedding())
                for role in ("patient", "doctor"):
                    for query in ("标签", "positive_probability", "tree_probability"):
                        with self.subTest(role=role, query=query):
                            result = retriever.search(query, role, {"model_public"}, "model")
                            self.assertTrue(result.hits)
                            self.assertEqual(result.retrieval_mode, "lexical_degraded")
                            hit = result.hits[0].public_json()
                            self.assertEqual(hit["source_id"], "src_treesem_label_coding")
                            self.assertTrue(hit["citation_id"].startswith("cite_"))
                            self.assertEqual(hit["url"], "treesem://project/label-coding")
                    self.assertFalse(retriever.search(
                        "标签", role, {"clinical_patient"}, "clinical").hits)
            finally:
                connection.close()
