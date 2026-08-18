from __future__ import annotations

import argparse
from pathlib import Path

from .builder import IndexBuilder


class SentenceTransformerEmbedding:
    def __init__(self, model_id: str, revision: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is required to build the knowledge index") from exc
        self.model_id = model_id
        self.revision = revision
        self._model = SentenceTransformer(model_id, revision=revision)

    def encode_passages(self, values: list[str]) -> list[list[float]]:
        return self._model.encode(
            ["passage: " + value for value in values], normalize_embeddings=True,
            convert_to_numpy=True).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a versioned treeSem knowledge index")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--embedding-model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--embedding-revision", required=True)
    parser.add_argument("--reranker-model", default=
                        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--reranker-revision", required=True)
    parser.add_argument("--reranker-min-score", required=True, type=float)
    parser.add_argument("--all-scope-reranker-min-score", required=True,
                        type=float)
    parser.add_argument("--rrf-min-score", required=True, type=float)
    args = parser.parse_args()
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sentence-transformers is required to build the knowledge index") from exc
    # Download and validate the exact reranker revision during the offline build.
    # The serving process later uses only the local model cache.
    CrossEncoder(args.reranker_model, revision=args.reranker_revision)
    builder = IndexBuilder(
        SentenceTransformerEmbedding(args.embedding_model, args.embedding_revision),
        reranker_model=args.reranker_model,
        reranker_revision=args.reranker_revision,
        reranker_min_score=args.reranker_min_score,
        all_scope_reranker_min_score=args.all_scope_reranker_min_score,
        rrf_min_score=args.rrf_min_score)
    print(builder.build(args.manifest, args.source_root, args.output_root))


if __name__ == "__main__":
    main()
