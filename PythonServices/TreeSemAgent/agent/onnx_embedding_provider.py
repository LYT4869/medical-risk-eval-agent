from __future__ import annotations

import math
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .routing_artifact import (
    RoutingArtifact,
    intent_examples_sha256,
)


def _default_tokenizer_factory(path: Path) -> Any:
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tokenizers is required for ONNX semantic routing") from exc
    return Tokenizer.from_file(str(path))


def _default_session_factory(path: Path) -> Any:
    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "onnxruntime is required for ONNX semantic routing") from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    return onnxruntime.InferenceSession(
        str(path), sess_options=options,
        providers=["CPUExecutionProvider"])


def _default_array_factory(rows: Sequence[Sequence[int]], dtype: str) -> Any:
    try:
        import numpy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required for ONNX semantic routing") from exc
    return numpy.asarray(rows, dtype=dtype)


class OnnxEmbeddingProvider:
    def __init__(
            self, artifact: RoutingArtifact, *,
            tokenizer_factory: Callable[[Path], Any] | None = None,
            session_factory: Callable[[Path], Any] | None = None,
            array_factory: Callable[[Sequence[Sequence[int]], str], Any] |
            None = None):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._artifact = artifact
        self._manifest = artifact.manifest
        self._tokenizer = (tokenizer_factory or _default_tokenizer_factory)(
            artifact.tokenizer_path)
        self._tokenizer.enable_truncation(
            max_length=self._manifest.max_sequence_length,
            direction="right")
        self._tokenizer.enable_padding(
            direction="right", pad_id=self._manifest.pad_token_id,
            pad_token=self._manifest.pad_token)
        self._session = (session_factory or _default_session_factory)(
            artifact.model_path)
        self._array_factory = array_factory or _default_array_factory
        inputs = tuple(item.name for item in self._session.get_inputs())
        outputs = tuple(item.name for item in self._session.get_outputs())
        if inputs != self._manifest.onnx_input_names:
            raise RuntimeError("ONNX input contract does not match artifact")
        if outputs != (self._manifest.onnx_output_name,):
            raise RuntimeError("ONNX output contract does not match artifact")

    def encode_examples(self, texts: Sequence[str]) -> list[list[float]]:
        values = tuple(texts)
        if (len(values) != self._manifest.intent_example_count or
                intent_examples_sha256(values) !=
                self._manifest.intent_examples_sha256):
            raise ValueError("intent examples do not match routing artifact")
        return [list(row) for row in self._artifact.intent_embeddings]

    def encode_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("routing query must be a non-empty string")
        return self._encode([self._manifest.query_prefix + text])[0]

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            encoded = self._tokenizer.encode_batch(
                list(texts), add_special_tokens=True)
        except Exception as exc:
            raise RuntimeError("routing tokenizer inference failed") from exc
        if len(encoded) != len(texts) or not encoded:
            raise RuntimeError("routing tokenizer returned invalid batch")
        input_ids = []
        attention_masks = []
        sequence_length = None
        for item in encoded:
            ids = list(item.ids)
            mask = list(item.attention_mask)
            if not ids or len(ids) != len(mask):
                raise RuntimeError("routing tokenizer returned invalid sequence")
            if len(ids) > self._manifest.max_sequence_length:
                raise ValueError("routing input exceeds tokenizer maximum length")
            if sequence_length is None:
                sequence_length = len(ids)
            elif len(ids) != sequence_length:
                raise RuntimeError("routing tokenizer did not pad the batch")
            input_ids.append(ids)
            attention_masks.append(mask)
        feeds = {
            "input_ids": self._array_factory(input_ids, "int64"),
            "attention_mask": self._array_factory(
                attention_masks, "int64"),
        }
        try:
            result = self._session.run(
                [self._manifest.onnx_output_name], feeds)
        except Exception as exc:
            raise RuntimeError("ONNX routing inference failed") from exc
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError("ONNX routing output batch is invalid")
        raw = result[0].tolist() if hasattr(result[0], "tolist") else result[0]
        if not isinstance(raw, (list, tuple)) or len(raw) != len(texts):
            raise RuntimeError("ONNX routing output batch mismatch")
        return [self._normalize(row) for row in raw]

    def _normalize(self, row: Any) -> list[float]:
        if not isinstance(row, (list, tuple)):
            raise RuntimeError("ONNX routing output row is invalid")
        if len(row) != self._manifest.embedding_dimension:
            raise RuntimeError("ONNX routing output dimension mismatch")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in row):
            raise RuntimeError("ONNX routing output must contain finite values")
        norm = math.sqrt(sum(float(value) * float(value) for value in row))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("ONNX routing output must have positive norm")
        return [float(value) / norm for value in row]
