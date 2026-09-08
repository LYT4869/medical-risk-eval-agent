from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry
from tools.export_routing_artifact import (
    ExportError,
    artifact_version,
    stable_json_bytes,
    validate_source_model,
    write_intent_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "config" / "tasks.yaml"


class Transformer:
    pass


class Pooling:
    pooling_mode = "mean"
    include_prompt = True


class Normalize:
    pass


class FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 1
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"


class FakeSourceModel:
    max_seq_length = 512
    tokenizer = FakeTokenizer()

    def __init__(self, modules=None):
        self._modules = modules or [Transformer(), Pooling(), Normalize()]

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, index):
        return self._modules[index]

    def get_embedding_dimension(self):
        return 384


class RecordingEncoder:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return self.rows


class RoutingExportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = TaskRegistry.load(TASKS, SUPPORTED_DOMAIN_TOOLS)

    def tearDown(self):
        self.temporary.cleanup()

    def test_stable_json_and_artifact_version_are_deterministic(self):
        left = {"backend": "onnx_fp32", "checksums": {"b": "2", "a": "1"}}
        right = {"checksums": {"a": "1", "b": "2"}, "backend": "onnx_fp32"}

        self.assertEqual(stable_json_bytes(left), stable_json_bytes(right))
        self.assertEqual(artifact_version(left), artifact_version(right))
        self.assertRegex(artifact_version(left), r"^routing-[0-9a-f]{16}$")

    def test_validates_current_sentence_transformer_contract(self):
        contract = validate_source_model(FakeSourceModel())

        self.assertEqual(contract.embedding_dimension, 384)
        self.assertEqual(contract.max_sequence_length, 512)
        self.assertEqual(contract.pad_token_id, 1)
        self.assertEqual(contract.input_names,
                         ("input_ids", "attention_mask"))

    def test_rejects_non_mean_pooling_source(self):
        pooling = Pooling()
        pooling.pooling_mode = "max"

        with self.assertRaisesRegex(ExportError, "mean pooling"):
            validate_source_model(FakeSourceModel(
                [Transformer(), pooling, Normalize()]))

    def test_rejects_changed_module_chain(self):
        with self.assertRaisesRegex(ExportError, "module chain"):
            validate_source_model(FakeSourceModel(
                [Transformer(), Normalize()]))

    def test_rejects_tokenizer_contract_change(self):
        source = FakeSourceModel()
        source.tokenizer = FakeTokenizer()
        source.tokenizer.padding_side = "left"

        with self.assertRaisesRegex(ExportError, "tokenizer"):
            validate_source_model(source)

    def test_uses_current_embedding_dimension_contract(self):
        source = FakeSourceModel()
        source.get_embedding_dimension = lambda: 256

        with self.assertRaisesRegex(ExportError, "embedding dimension"):
            validate_source_model(source)

    def test_selected_backend_writes_its_own_prefixed_intent_embeddings(self):
        examples = tuple(
            example
            for definition in self.registry.business_definitions
            for example in definition.intent_examples)
        rows = [[1.0, 0.0] for _ in examples]
        encoder = RecordingEncoder(rows)
        output = self.root / "intent_embeddings.f32"

        digest = write_intent_embeddings(
            self.registry, encoder, output,
            passage_prefix="passage: ", embedding_dimension=2)

        self.assertEqual(encoder.calls, [[
            "passage: " + example for example in examples]])
        self.assertEqual(len(output.read_bytes()), len(examples) * 2 * 4)
        self.assertEqual(digest,
                         hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(struct.unpack("<2f", output.read_bytes()[:8]),
                         (1.0, 0.0))

    def test_intent_embedding_writer_rejects_count_dimension_and_values(self):
        example_count = sum(
            len(item.intent_examples)
            for item in self.registry.business_definitions)
        cases = (
            ([[1.0, 0.0]] * (example_count - 1), "count"),
            ([[1.0]] * example_count, "dimension"),
            ([[math.nan, 0.0]] * example_count, "finite"),
            ([[0.0, 0.0]] * example_count, "positive norm"),
        )
        for rows, message in cases:
            with self.subTest(message=message), \
                    self.assertRaisesRegex(ExportError, message):
                write_intent_embeddings(
                    self.registry, RecordingEncoder(rows),
                    self.root / f"{message}.f32",
                    passage_prefix="passage: ", embedding_dimension=2)

    def test_stable_json_is_utf8_without_non_finite_values(self):
        rendered = stable_json_bytes({"message": "产后出血", "value": 1.0})
        self.assertEqual(json.loads(rendered.decode("utf-8"))["message"],
                         "产后出血")
        with self.assertRaisesRegex(ExportError, "finite JSON"):
            stable_json_bytes({"value": math.inf})


if __name__ == "__main__":
    unittest.main()
