from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Protocol

from ..ingestion.chunker import tokens
from ..models import Chunk, SearchHit, SearchResponse


class QueryEmbedding(Protocol):
    def encode_query(self, value: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class KnowledgeIndex:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("unsupported knowledge index schema")
        for name, expected in self.manifest.get("files", {}).items():
            path = self.directory / name
            data = path.read_bytes()
            if len(data) != expected["size"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise ValueError(f"knowledge index checksum mismatch: {name}")
        self.chunks = tuple(Chunk(**json.loads(line)) for line in
                            (self.directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines())
        if len(self.chunks) != self.manifest.get("chunk_count"):
            raise ValueError("knowledge index chunk count mismatch")
        self.by_id = {item.chunk_id: item for item in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise ValueError("duplicate chunk id")
        self._lexical = sqlite3.connect(
            f"file:{self.directory / 'lexical.sqlite'}?mode=ro", uri=True,
            check_same_thread=False)
        try:
            import faiss  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("faiss-cpu is required to load the dense index") from exc
        self._faiss = faiss
        self._dense = faiss.read_index(str(self.directory / "vectors.faiss"))
        if self._dense.ntotal != len(self.chunks):
            raise ValueError("dense index row count mismatch")

    @property
    def version(self) -> str:
        return str(self.manifest["index_version"])

    def close(self) -> None:
        self._lexical.close()

    def lexical(self, query: str, limit: int) -> list[str]:
        parts = [part.lower().replace('"', '') for part in tokens(query) if part.strip()]
        if not parts:
            return []
        expression = " OR ".join(f'"{part}"' for part in parts[:64])
        rows = self._lexical.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
            (expression, limit)).fetchall()
        return [str(row[0]) for row in rows]

    def dense(self, vector: list[float], limit: int) -> list[str]:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("numpy is required for dense retrieval") from exc
        matrix = np.asarray([vector], dtype="float32")
        if matrix.shape[1] != self._dense.d:
            raise ValueError("query embedding dimension mismatch")
        self._faiss.normalize_L2(matrix)
        _, indexes = self._dense.search(matrix, min(limit, len(self.chunks)))
        return [self.chunks[int(index)].chunk_id for index in indexes[0] if index >= 0]


class HybridRetriever:
    def __init__(self, index: KnowledgeIndex, embedding: QueryEmbedding,
                 reranker: Reranker | None = None, rrf_k: int = 60,
                 reranker_min_score: float = float("-inf"),
                 all_scope_reranker_min_score: float | None = None,
                 rrf_min_score: float = float("-inf")):
        self.index = index
        self.embedding = embedding
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.reranker_min_score = reranker_min_score
        self.all_scope_reranker_min_score = (
            reranker_min_score if all_scope_reranker_min_score is None else
            all_scope_reranker_min_score)
        self.rrf_min_score = rrf_min_score

    @staticmethod
    def _must_abstain(query: str, role: str) -> bool:
        normalized = " ".join(query.lower().split())
        always_denied = (
            r"忽略.{0,12}(?:系统|之前).{0,12}指令",
            r"ignore.{0,20}(?:system|previous).{0,20}instruction",
            r"(?:显示|泄露).{0,12}(?:系统提示|专业资料)",
            r"(?:reveal|show).{0,20}system prompt",
            r"(?:具体)?药物.{0,10}剂量|开.{0,6}(?:药|处方)",
            r"(?:specific )?(?:drug )?dosage|prescri(?:be|ption)",
            r"诊断.{0,8}(?:一定|肯定)|definitively diagnose",
            r"数据库.{0,12}患者.{0,8}姓名|patient names?.{0,20}database",
        )
        if any(re.search(pattern, normalized) for pattern in always_denied):
            return True
        patient_denied = (
            r"专业.{0,10}治疗建议",
            r"clinical professional protocol",
        )
        return role == "patient" and any(
            re.search(pattern, normalized) for pattern in patient_denied)

    @staticmethod
    def _allowed(chunk: Chunk, role: str, scopes: set[str], requested: str) -> bool:
        if chunk.knowledge_scope not in scopes:
            return False
        if role == "patient" and chunk.audience not in {"both", "patient"}:
            return False
        if role not in {"patient", "doctor"}:
            return False
        if requested == "model" and not chunk.knowledge_scope.startswith("model_"):
            return False
        if requested == "clinical" and not chunk.knowledge_scope.startswith("clinical_"):
            return False
        return True

    def search(self, query: str, role: str, scopes: set[str], scope: str = "all",
               top_k: int = 5) -> SearchResponse:
        query = query.strip()
        if not query or len(query) > 500 or scope not in {"model", "clinical", "all"}:
            raise ValueError("invalid knowledge search request")
        if top_k < 1 or top_k > 6:
            raise ValueError("invalid top_k")
        if self._must_abstain(query, role):
            return SearchResponse(self.index.version, "hybrid", ())
        lexical: list[str] | None
        dense: list[str] | None
        try:
            lexical = self.index.lexical(query, 20)
        except Exception:
            lexical = None
        try:
            dense = self.index.dense(self.embedding.encode_query(query), 20)
        except Exception:
            dense = None
        if lexical is None and dense is None:
            raise RuntimeError("knowledge retrieval is unavailable")
        if lexical is None:
            mode = "dense_degraded"
        elif dense is None:
            mode = "lexical_degraded"
        else:
            mode = "hybrid"
        scores: dict[str, float] = {}
        for ranking in (lexical or [], dense or []):
            for rank, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        candidates = [self.index.by_id[item[0]] for item in
                      sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                      if self._allowed(self.index.by_id[item[0]], role, scopes, scope)]
        candidates = candidates[:10]
        if self.reranker is not None and candidates:
            try:
                rerank = self.reranker.score(query, [item.text for item in candidates])
                if len(rerank) != len(candidates) or any(not math.isfinite(value) for value in rerank):
                    raise ValueError("invalid reranker output")
                ordered = sorted(zip(candidates, rerank), key=lambda item: (-item[1], item[0].chunk_id))
                minimum = (self.all_scope_reranker_min_score if scope == "all"
                           else self.reranker_min_score)
                hits = [SearchHit(item, float(score)) for item, score in ordered
                        if score >= minimum][:top_k]
            except Exception:
                mode = "rrf_degraded"
                hits = [SearchHit(item, scores[item.chunk_id]) for item in candidates
                        if scores[item.chunk_id] >= self.rrf_min_score][:top_k]
        else:
            hits = [SearchHit(item, scores[item.chunk_id]) for item in candidates
                    if scores[item.chunk_id] >= self.rrf_min_score][:top_k]
        return SearchResponse(self.index.version, mode, tuple(hits))


class SentenceTransformerQueryEmbedding:
    def __init__(self, model_id: str, revision: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is required for knowledge retrieval") from exc
        self._model = SentenceTransformer(model_id, revision=revision, local_files_only=True)

    def encode_query(self, value: str) -> list[float]:
        return self._model.encode(
            ["query: " + value], normalize_embeddings=True,
            convert_to_numpy=True)[0].tolist()


class CrossEncoderReranker:
    def __init__(self, model_id: str, revision: str):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is required for reranking") from exc
        self._model = CrossEncoder(model_id, revision=revision, local_files_only=True)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(item) for item in self._model.predict([[query, value] for value in passages])]
