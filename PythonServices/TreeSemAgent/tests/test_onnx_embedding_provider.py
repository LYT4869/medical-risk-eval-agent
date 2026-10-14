from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from agent.onnx_embedding_provider import OnnxEmbeddingProvider
from agent.routing_artifact import (
    RoutingArtifact,
    RoutingArtifactManifest,
    intent_examples_sha256,
)


class NamedValue:
    def __init__(self, name):
        self.name = name


class Encoding:
    def __init__(self, ids, attention_mask):
        self.ids = ids
        self.attention_mask = attention_mask


class FakeTokenizer:
    def __init__(self, *, sequence_length=3):
        self.sequence_length = sequence_length
        self.texts = []
        self.truncation = None
        self.padding = None

    def enable_truncation(self, **kwargs):
        self.truncation = kwargs

    def enable_padding(self, **kwargs):
        self.padding = kwargs

    def encode_batch(self, texts, add_special_tokens):
        self.texts.extend(texts)
        assert add_special_tokens is True
        return [Encoding(
            list(range(self.sequence_length)),
            [1] * self.sequence_length) for _ in texts]


class FakeArray:
    def __init__(self, rows, dtype):
        self.rows = rows
        self.dtype = dtype


class FakeSession:
    def __init__(self, output, *, inputs=None, outputs=None):
        self.output = output
        self.inputs = inputs or ["input_ids", "attention_mask"]
        self.outputs = outputs or ["sentence_embedding"]
        self.calls = []

    def get_inputs(self):
        return [NamedValue(name) for name in self.inputs]

    def get_outputs(self):
        return [NamedValue(name) for name in self.outputs]

    def run(self, outputs, feeds):
        self.calls.append((outputs, feeds))
        return [self.output]


def manifest(*, examples, dimension=2):
    digest = hashlib.sha256(b"file").hexdigest()
    return RoutingArtifactManifest(
        schema_version=1,
        artifact_version="routing-test",
        source_model_id="intfloat/multilingual-e5-small",
        source_model_revision=(
            "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        embedding_backend="onnx_fp32",
        onnx_opset=17,
        onnx_input_names=("input_ids", "attention_mask"),
        onnx_output_name="sentence_embedding",
        embedding_dimension=dimension,
        intent_example_count=len(examples),
        intent_examples_sha256=intent_examples_sha256(examples),
        query_prefix="query: ",
        passage_prefix="passage: ",
        max_sequence_length=512,
        pooling="attention_mask_mean",
        normalization="l2",
        pad_token="<pad>",
        pad_token_id=1,
        tokenizer_library_version="0.22.2",
        exporter_version="1",
        task_registry_sha256=digest,
        routing_thresholds_sha256=digest,
        model_sha256=digest,
        tokenizer_sha256=digest,
        tokenizer_config_sha256=digest,
        intent_embeddings_sha256=digest,
        golden_routes_sha256=digest,
    )


class OnnxEmbeddingProviderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.examples = ("第一个示例", "second example")
        self.tokenizer = FakeTokenizer()
        self.session = FakeSession([[3.0, 4.0]])

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, *, embeddings=((1.0, 0.0), (0.0, 1.0)),
                 artifact_manifest=None):
        return RoutingArtifact(
            directory=self.root,
            manifest=artifact_manifest or manifest(examples=self.examples),
            model_path=self.root / "model.onnx",
            tokenizer_path=self.root / "tokenizer.json",
            intent_embeddings=embeddings,
        )

    def provider(self, *, embeddings=((1.0, 0.0), (0.0, 1.0)),
                 artifact_manifest=None, tokenizer=None, session=None):
        return OnnxEmbeddingProvider(
            self.artifact(
                embeddings=embeddings,
                artifact_manifest=artifact_manifest),
            tokenizer_factory=lambda _: tokenizer or self.tokenizer,
            session_factory=lambda _: session or self.session,
            array_factory=lambda rows, dtype: FakeArray(rows, dtype),
        )

    def test_initializes_fixed_tokenizer_contract(self):
        self.provider()

        self.assertEqual(self.tokenizer.truncation, {
            "max_length": 512, "direction": "right"})
        self.assertEqual(self.tokenizer.padding, {
            "direction": "right", "pad_id": 1, "pad_token": "<pad>"})

    def test_query_uses_prefix_int64_inputs_and_normalizes_output(self):
        vector = self.provider().encode_query("解释结果")

        self.assertEqual(self.tokenizer.texts, ["query: 解释结果"])
        self.assertEqual(vector, [0.6, 0.8])
        self.assertEqual(len(self.session.calls), 1)
        outputs, feeds = self.session.calls[0]
        self.assertEqual(outputs, ["sentence_embedding"])
        self.assertEqual(set(feeds), {"input_ids", "attention_mask"})
        self.assertEqual(
            {value.dtype for value in feeds.values()}, {"int64"})

    def test_examples_use_backend_aligned_precomputed_matrix(self):
        provider = self.provider()

        vectors = provider.encode_examples(self.examples)

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(self.session.calls, [])
        self.assertEqual(self.tokenizer.texts, [])

    def test_rejects_unexpected_example_sequence(self):
        provider = self.provider()

        with self.assertRaisesRegex(ValueError, "intent examples"):
            provider.encode_examples(("changed", "second example"))

    def test_rejects_empty_or_non_string_query(self):
        provider = self.provider()
        for value in ("", "   ", True, None):
            with self.subTest(value=value), \
                    self.assertRaisesRegex(ValueError, "query"):
                provider.encode_query(value)

    def test_rejects_tokenizer_output_beyond_manifest_limit(self):
        provider = self.provider(tokenizer=FakeTokenizer(sequence_length=513))

        with self.assertRaisesRegex(ValueError, "maximum length"):
            provider.encode_query("too long")

    def test_rejects_invalid_session_io_contract(self):
        with self.assertRaisesRegex(RuntimeError, "ONNX input"):
            self.provider(session=FakeSession(
                [[1.0, 0.0]], inputs=["features"]))
        with self.assertRaisesRegex(RuntimeError, "ONNX output"):
            self.provider(session=FakeSession(
                [[1.0, 0.0]], outputs=["other"]))

    def test_rejects_output_count_dimension_nonfinite_and_zero_norm(self):
        cases = (
            ([], "batch"),
            ([[1.0]], "dimension"),
            ([[math.inf, 0.0]], "finite"),
            ([[0.0, 0.0]], "positive norm"),
        )
        for output, message in cases:
            with self.subTest(message=message):
                provider = self.provider(session=FakeSession(output))
                with self.assertRaisesRegex(RuntimeError, message):
                    provider.encode_query("query")


if __name__ == "__main__":
    unittest.main()
