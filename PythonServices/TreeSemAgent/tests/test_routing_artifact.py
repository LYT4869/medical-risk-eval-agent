from __future__ import annotations

import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from agent.routing_artifact import (
    RoutingArtifactError,
    canonical_intent_examples,
    intent_examples_sha256,
    load_routing_artifact,
)
from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "config" / "tasks.yaml"
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RoutingArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = TaskRegistry.load(TASKS, SUPPORTED_DOMAIN_TOOLS)
        self.examples = canonical_intent_examples(self.registry)

    def tearDown(self):
        self.temporary.cleanup()

    def write_artifact(self, *, backend: str = "onnx_fp32") -> Path:
        directory = self.root / "artifact"
        directory.mkdir()
        (directory / "model.onnx").write_bytes(b"fake-onnx-model")
        (directory / "tokenizer.json").write_text(
            '{"version":"1.0"}\n', encoding="utf-8")
        (directory / "tokenizer_config.json").write_text(
            '{"padding_side":"right","truncation_side":"right"}\n',
            encoding="utf-8")
        (directory / "golden_routes.json").write_text(
            '{"cases":[],"schema_version":1}\n', encoding="utf-8")
        row = struct.pack("<f", 1.0) + struct.pack("<f", 0.0) * 383
        (directory / "intent_embeddings.f32").write_bytes(
            row * len(self.examples))
        manifest = {
            "schema_version": 1,
            "artifact_version": hashlib.sha256(b"artifact").hexdigest()[:16],
            "source_model_id": MODEL,
            "source_model_revision": REVISION,
            "embedding_backend": backend,
            "onnx_opset": 17,
            "onnx_input_names": ["input_ids", "attention_mask"],
            "onnx_output_name": "sentence_embedding",
            "embedding_dimension": 384,
            "intent_example_count": len(self.examples),
            "intent_examples_sha256": intent_examples_sha256(self.examples),
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "max_sequence_length": 512,
            "pooling": "attention_mask_mean",
            "normalization": "l2",
            "pad_token": "<pad>",
            "pad_token_id": 1,
            "tokenizer_library_version": "0.22.2",
            "exporter_version": "1",
            "task_registry_sha256": sha256(TASKS),
            "model_sha256": sha256(directory / "model.onnx"),
            "tokenizer_sha256": sha256(directory / "tokenizer.json"),
            "tokenizer_config_sha256": sha256(
                directory / "tokenizer_config.json"),
            "intent_embeddings_sha256": sha256(
                directory / "intent_embeddings.f32"),
            "golden_routes_sha256": sha256(
                directory / "golden_routes.json"),
        }
        self.write_manifest(directory, manifest)
        return directory

    @staticmethod
    def read_manifest(directory: Path) -> dict[str, object]:
        return json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8"))

    @staticmethod
    def write_manifest(directory: Path, manifest: dict[str, object]) -> None:
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

    def load(self, directory: Path, *, backend: str = "onnx_fp32"):
        return load_routing_artifact(
            directory,
            task_registry_path=TASKS,
            registry=self.registry,
            expected_backend=backend,
            expected_model_id=MODEL,
            expected_revision=REVISION,
        )

    def refresh_checksum(self, directory: Path, field: str,
                         filename: str) -> None:
        manifest = self.read_manifest(directory)
        manifest[field] = sha256(directory / filename)
        self.write_manifest(directory, manifest)

    def test_loads_backend_aligned_artifact(self):
        loaded = self.load(self.write_artifact())

        self.assertEqual(loaded.manifest.embedding_backend, "onnx_fp32")
        self.assertEqual(loaded.manifest.intent_example_count,
                         len(self.examples))
        self.assertEqual(loaded.model_path.name, "model.onnx")
        self.assertEqual(loaded.tokenizer_path.name, "tokenizer.json")
        self.assertEqual(len(loaded.intent_embeddings), len(self.examples))
        self.assertEqual(len(loaded.intent_embeddings[0]), 384)
        self.assertEqual(loaded.intent_embeddings[0][0], 1.0)

    def test_canonical_examples_follow_sorted_scope_and_declared_order(self):
        expected = tuple(
            example
            for definition in self.registry.business_definitions
            for example in definition.intent_examples
        )
        self.assertEqual(self.examples, expected)
        self.assertEqual(intent_examples_sha256(self.examples),
                         intent_examples_sha256(tuple(self.examples)))

    def test_rejects_unknown_manifest_field(self):
        directory = self.write_artifact()
        manifest = self.read_manifest(directory)
        manifest["unexpected"] = True
        self.write_manifest(directory, manifest)

        with self.assertRaisesRegex(RoutingArtifactError,
                                    "manifest fields"):
            self.load(directory)

    def test_rejects_backend_mismatch(self):
        directory = self.write_artifact(backend="onnx_fp32")

        with self.assertRaisesRegex(RoutingArtifactError, "backend"):
            self.load(directory, backend="onnx_int8")

    def test_rejects_model_and_revision_mismatch(self):
        directory = self.write_artifact()
        with self.assertRaisesRegex(RoutingArtifactError, "model"):
            load_routing_artifact(
                directory, task_registry_path=TASKS, registry=self.registry,
                expected_backend="onnx_fp32", expected_model_id="wrong",
                expected_revision=REVISION)
        with self.assertRaisesRegex(RoutingArtifactError, "revision"):
            load_routing_artifact(
                directory, task_registry_path=TASKS, registry=self.registry,
                expected_backend="onnx_fp32", expected_model_id=MODEL,
                expected_revision="wrong")

    def test_rejects_task_registry_or_example_digest_mismatch(self):
        directory = self.write_artifact()
        manifest = self.read_manifest(directory)
        manifest["task_registry_sha256"] = "0" * 64
        self.write_manifest(directory, manifest)
        with self.assertRaisesRegex(RoutingArtifactError, "Task Registry"):
            self.load(directory)

        manifest["task_registry_sha256"] = sha256(TASKS)
        manifest["intent_examples_sha256"] = "1" * 64
        self.write_manifest(directory, manifest)
        with self.assertRaisesRegex(RoutingArtifactError, "intent examples"):
            self.load(directory)

    def test_rejects_modified_business_file(self):
        directory = self.write_artifact()
        (directory / "model.onnx").write_bytes(b"tampered")

        with self.assertRaisesRegex(RoutingArtifactError, "checksum"):
            self.load(directory)

    def test_rejects_truncated_float_matrix(self):
        directory = self.write_artifact()
        (directory / "intent_embeddings.f32").write_bytes(b"\x00" * 7)
        self.refresh_checksum(
            directory, "intent_embeddings_sha256",
            "intent_embeddings.f32")

        with self.assertRaisesRegex(RoutingArtifactError, "matrix size"):
            self.load(directory)

    def test_rejects_non_finite_float_matrix(self):
        directory = self.write_artifact()
        matrix = bytearray((directory / "intent_embeddings.f32").read_bytes())
        matrix[:4] = struct.pack("<f", math.nan)
        (directory / "intent_embeddings.f32").write_bytes(matrix)
        self.refresh_checksum(
            directory, "intent_embeddings_sha256",
            "intent_embeddings.f32")

        with self.assertRaisesRegex(RoutingArtifactError, "finite"):
            self.load(directory)

    def test_rejects_zero_norm_float_matrix_row(self):
        directory = self.write_artifact()
        matrix = bytearray((directory / "intent_embeddings.f32").read_bytes())
        matrix[:384 * 4] = b"\x00" * (384 * 4)
        (directory / "intent_embeddings.f32").write_bytes(matrix)
        self.refresh_checksum(
            directory, "intent_embeddings_sha256",
            "intent_embeddings.f32")

        with self.assertRaisesRegex(RoutingArtifactError, "positive norm"):
            self.load(directory)

    def test_rejects_unsupported_onnx_contract(self):
        directory = self.write_artifact()
        manifest = self.read_manifest(directory)
        manifest["onnx_input_names"] = ["features"]
        self.write_manifest(directory, manifest)

        with self.assertRaisesRegex(RoutingArtifactError, "ONNX input"):
            self.load(directory)


if __name__ == "__main__":
    unittest.main()
