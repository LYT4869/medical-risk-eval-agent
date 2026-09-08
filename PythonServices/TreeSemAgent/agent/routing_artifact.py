from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .task_registry import TaskRegistry


_MODEL_FILE = "model.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_INTENT_EMBEDDINGS_FILE = "intent_embeddings.f32"
_GOLDEN_ROUTES_FILE = "golden_routes.json"
_CHECKSUM = re.compile(r"[0-9a-f]{64}")
_BACKENDS = frozenset({"onnx_fp32", "onnx_int8"})
_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "artifact_version",
    "source_model_id",
    "source_model_revision",
    "embedding_backend",
    "onnx_opset",
    "onnx_input_names",
    "onnx_output_name",
    "embedding_dimension",
    "intent_example_count",
    "intent_examples_sha256",
    "query_prefix",
    "passage_prefix",
    "max_sequence_length",
    "pooling",
    "normalization",
    "pad_token",
    "pad_token_id",
    "tokenizer_library_version",
    "exporter_version",
    "task_registry_sha256",
    "model_sha256",
    "tokenizer_sha256",
    "tokenizer_config_sha256",
    "intent_embeddings_sha256",
    "golden_routes_sha256",
})


class RoutingArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class RoutingArtifactManifest:
    schema_version: int
    artifact_version: str
    source_model_id: str
    source_model_revision: str
    embedding_backend: str
    onnx_opset: int
    onnx_input_names: tuple[str, ...]
    onnx_output_name: str
    embedding_dimension: int
    intent_example_count: int
    intent_examples_sha256: str
    query_prefix: str
    passage_prefix: str
    max_sequence_length: int
    pooling: str
    normalization: str
    pad_token: str
    pad_token_id: int
    tokenizer_library_version: str
    exporter_version: str
    task_registry_sha256: str
    model_sha256: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    intent_embeddings_sha256: str
    golden_routes_sha256: str


@dataclass(frozen=True)
class RoutingArtifact:
    directory: Path
    manifest: RoutingArtifactManifest
    model_path: Path
    tokenizer_path: Path
    intent_embeddings: tuple[tuple[float, ...], ...]


def canonical_intent_examples(registry: TaskRegistry) -> tuple[str, ...]:
    return tuple(
        example
        for definition in registry.business_definitions
        for example in definition.intent_examples
    )


def intent_examples_sha256(examples: Sequence[str]) -> str:
    if any(not isinstance(item, str) or not item for item in examples):
        raise ValueError("intent examples must be non-empty strings")
    payload = json.dumps(
        list(examples), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RoutingArtifactError("cannot read routing artifact file") from exc
    return digest.hexdigest()


def _string(payload: dict[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise RoutingArtifactError(f"invalid routing manifest {name}")
    return value


def _integer(payload: dict[str, object], name: str, *, minimum: int) -> int:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RoutingArtifactError(f"invalid routing manifest {name}")
    return value


def _checksum(payload: dict[str, object], name: str) -> str:
    value = _string(payload, name)
    if _CHECKSUM.fullmatch(value) is None:
        raise RoutingArtifactError(f"invalid routing manifest {name}")
    return value


def _parse_manifest(path: Path) -> RoutingArtifactManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoutingArtifactError("cannot load routing artifact manifest") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise RoutingArtifactError("invalid routing manifest fields")
    if _integer(payload, "schema_version", minimum=1) != 1:
        raise RoutingArtifactError("unsupported routing artifact Schema")
    raw_inputs = payload["onnx_input_names"]
    if (not isinstance(raw_inputs, list) or
            any(not isinstance(item, str) for item in raw_inputs)):
        raise RoutingArtifactError("invalid ONNX input names")
    inputs = tuple(raw_inputs)
    if inputs != ("input_ids", "attention_mask"):
        raise RoutingArtifactError("unsupported ONNX input contract")

    manifest = RoutingArtifactManifest(
        schema_version=1,
        artifact_version=_string(payload, "artifact_version"),
        source_model_id=_string(payload, "source_model_id"),
        source_model_revision=_string(payload, "source_model_revision"),
        embedding_backend=_string(payload, "embedding_backend"),
        onnx_opset=_integer(payload, "onnx_opset", minimum=1),
        onnx_input_names=inputs,
        onnx_output_name=_string(payload, "onnx_output_name"),
        embedding_dimension=_integer(
            payload, "embedding_dimension", minimum=1),
        intent_example_count=_integer(
            payload, "intent_example_count", minimum=1),
        intent_examples_sha256=_checksum(
            payload, "intent_examples_sha256"),
        query_prefix=_string(payload, "query_prefix"),
        passage_prefix=_string(payload, "passage_prefix"),
        max_sequence_length=_integer(
            payload, "max_sequence_length", minimum=1),
        pooling=_string(payload, "pooling"),
        normalization=_string(payload, "normalization"),
        pad_token=_string(payload, "pad_token"),
        pad_token_id=_integer(payload, "pad_token_id", minimum=0),
        tokenizer_library_version=_string(
            payload, "tokenizer_library_version"),
        exporter_version=_string(payload, "exporter_version"),
        task_registry_sha256=_checksum(payload, "task_registry_sha256"),
        model_sha256=_checksum(payload, "model_sha256"),
        tokenizer_sha256=_checksum(payload, "tokenizer_sha256"),
        tokenizer_config_sha256=_checksum(
            payload, "tokenizer_config_sha256"),
        intent_embeddings_sha256=_checksum(
            payload, "intent_embeddings_sha256"),
        golden_routes_sha256=_checksum(payload, "golden_routes_sha256"),
    )
    if manifest.embedding_backend not in _BACKENDS:
        raise RoutingArtifactError("unsupported routing embedding backend")
    if manifest.onnx_opset != 17:
        raise RoutingArtifactError("unsupported ONNX opset")
    if manifest.onnx_output_name != "sentence_embedding":
        raise RoutingArtifactError("unsupported ONNX output contract")
    if manifest.embedding_dimension != 384:
        raise RoutingArtifactError("unsupported embedding dimension")
    if (manifest.query_prefix != "query: " or
            manifest.passage_prefix != "passage: "):
        raise RoutingArtifactError("unsupported E5 prefix contract")
    if manifest.max_sequence_length != 512:
        raise RoutingArtifactError("unsupported tokenizer length")
    if (manifest.pooling != "attention_mask_mean" or
            manifest.normalization != "l2"):
        raise RoutingArtifactError("unsupported embedding postprocessing")
    if manifest.pad_token != "<pad>" or manifest.pad_token_id != 1:
        raise RoutingArtifactError("unsupported tokenizer padding")
    return manifest


def _regular_file(directory: Path, filename: str) -> Path:
    path = directory / filename
    if path.is_symlink() or not path.is_file():
        raise RoutingArtifactError("routing artifact file is missing or unsafe")
    return path


def _load_matrix(path: Path, rows: int,
                 columns: int) -> tuple[tuple[float, ...], ...]:
    expected_bytes = rows * columns * 4
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RoutingArtifactError("cannot read intent embedding matrix") from exc
    if len(raw) != expected_bytes:
        raise RoutingArtifactError("intent embedding matrix size mismatch")
    unpacked = struct.iter_unpack("<f", raw)
    values = [item[0] for item in unpacked]
    matrix = []
    for row_index in range(rows):
        row = tuple(values[
            row_index * columns:(row_index + 1) * columns])
        if any(not math.isfinite(value) for value in row):
            raise RoutingArtifactError("intent embeddings must be finite")
        norm = math.sqrt(sum(value * value for value in row))
        if not math.isfinite(norm) or norm <= 0:
            raise RoutingArtifactError(
                "intent embeddings must have positive norm")
        matrix.append(row)
    return tuple(matrix)


def load_routing_artifact(
        directory: Path, *, task_registry_path: Path,
        registry: TaskRegistry, expected_backend: str,
        expected_model_id: str, expected_revision: str) -> RoutingArtifact:
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise RoutingArtifactError("routing artifact directory is unavailable")
    manifest_path = _regular_file(directory, "manifest.json")
    manifest = _parse_manifest(manifest_path)

    if manifest.embedding_backend != expected_backend:
        raise RoutingArtifactError("routing artifact backend mismatch")
    if manifest.source_model_id != expected_model_id:
        raise RoutingArtifactError("routing artifact model mismatch")
    if manifest.source_model_revision != expected_revision:
        raise RoutingArtifactError("routing artifact revision mismatch")

    if _sha256(task_registry_path) != manifest.task_registry_sha256:
        raise RoutingArtifactError("routing artifact Task Registry mismatch")
    examples = canonical_intent_examples(registry)
    if (len(examples) != manifest.intent_example_count or
            intent_examples_sha256(examples) !=
            manifest.intent_examples_sha256):
        raise RoutingArtifactError("routing artifact intent examples mismatch")

    files = {
        "model_sha256": _regular_file(directory, _MODEL_FILE),
        "tokenizer_sha256": _regular_file(directory, _TOKENIZER_FILE),
        "tokenizer_config_sha256": _regular_file(
            directory, _TOKENIZER_CONFIG_FILE),
        "intent_embeddings_sha256": _regular_file(
            directory, _INTENT_EMBEDDINGS_FILE),
        "golden_routes_sha256": _regular_file(
            directory, _GOLDEN_ROUTES_FILE),
    }
    for field, path in files.items():
        if _sha256(path) != getattr(manifest, field):
            raise RoutingArtifactError("routing artifact checksum mismatch")

    matrix = _load_matrix(
        files["intent_embeddings_sha256"],
        manifest.intent_example_count,
        manifest.embedding_dimension)
    return RoutingArtifact(
        directory=directory.resolve(),
        manifest=manifest,
        model_path=files["model_sha256"].resolve(),
        tokenizer_path=files["tokenizer_sha256"].resolve(),
        intent_embeddings=matrix,
    )
