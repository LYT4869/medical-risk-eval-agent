from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


class SentenceTransformerEmbeddingProvider:
    def __init__(self, model_id: str, revision: str,
                 *, factory: Callable[..., Any] | None = None):
        if not model_id.strip() or not revision.strip():
            raise ValueError("routing embedding model and revision are required")
        if factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for semantic routing") from exc
            factory = SentenceTransformer
        self.model_id = model_id
        self.revision = revision
        self._model = factory(
            model_id, revision=revision, local_files_only=True)

    def encode_examples(self, texts: Sequence[str]) -> list[list[float]]:
        encoded = self._model.encode(
            ["passage: " + text for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._matrix(encoded)

    def encode_query(self, text: str) -> list[float]:
        encoded = self._model.encode(
            ["query: " + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self._matrix(encoded)[0]

    @staticmethod
    def _matrix(value: Any) -> list[list[float]]:
        raw = value.tolist() if hasattr(value, "tolist") else value
        return [[float(item) for item in row] for row in raw]

