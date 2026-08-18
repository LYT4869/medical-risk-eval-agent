from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Protocol

from ..models import Chunk, SourceRecord
from .chunker import chunk_markdown, chunk_pages, tokens


class EmbeddingModel(Protocol):
    model_id: str
    revision: str

    def encode_passages(self, values: list[str]) -> list[list[float]]: ...


class _HtmlMarkdownExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._main_depth = 0
        self._skip_depth = 0
        self._tag: str | None = None
        self._all: list[str] = []
        self._main: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "main":
            self._main_depth += 1
        if tag in {"script", "style", "noscript", "nav", "header", "footer"}:
            self._skip_depth += 1
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li"}:
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "main" and self._main_depth:
            self._main_depth -= 1
        if tag in {"script", "style", "noscript", "nav", "header", "footer"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == self._tag:
            self._tag = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._tag is None:
            return
        value = " ".join(data.split())
        if not value:
            return
        if self._tag.startswith("h"):
            value = "#" * int(self._tag[1]) + " " + value
        elif self._tag == "li":
            value = "- " + value
        self._all.append(value)
        if self._main_depth:
            self._main.append(value)

    def markdown(self) -> str:
        return "\n".join(self._main or self._all)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


class IndexBuilder:
    def __init__(self, embedding: EmbeddingModel, target_tokens: int = 400,
                 overlap_tokens: int = 60, *, reranker_model: str,
                 reranker_revision: str, reranker_min_score: float,
                 rrf_min_score: float,
                 all_scope_reranker_min_score: float | None = None):
        self.embedding = embedding
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        if not reranker_model or not reranker_revision:
            raise ValueError("reranker model and revision must be pinned")
        self.reranker_model = reranker_model
        self.reranker_revision = reranker_revision
        all_scope_threshold = (reranker_min_score if
                               all_scope_reranker_min_score is None else
                               all_scope_reranker_min_score)
        if (not math.isfinite(reranker_min_score) or
                not math.isfinite(all_scope_threshold) or
                not math.isfinite(rrf_min_score)):
            raise ValueError("retrieval thresholds must be finite")
        self.reranker_min_score = reranker_min_score
        self.all_scope_reranker_min_score = all_scope_threshold
        self.rrf_min_score = rrf_min_score

    @staticmethod
    def _load_manifest(path: Path) -> list[SourceRecord]:
        raw = path.read_text(encoding="utf-8")
        try:
            root = json.loads(raw)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError("PyYAML is required for non-JSON source manifests") from exc
            root = yaml.safe_load(raw)
        if not isinstance(root, dict) or set(root) != {"schema_version", "sources"} or root["schema_version"] != 1:
            raise ValueError("invalid source manifest")
        if not isinstance(root["sources"], list):
            raise ValueError("sources must be an array")
        records = [SourceRecord.parse(item) for item in root["sources"]]
        if len({item.source_id for item in records}) != len(records):
            raise ValueError("duplicate source_id")
        return sorted(records, key=lambda item: item.source_id)

    def _read(self, record: SourceRecord, root: Path) -> list[Chunk]:
        path = (root / record.local_path).resolve()
        if root.resolve() not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError("source path escapes the trusted root")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record.sha256:
            raise ValueError(f"source checksum mismatch: {record.source_id}")
        if path.suffix.lower() in {".md", ".txt"}:
            return chunk_markdown(record, path.read_text(encoding="utf-8"),
                                  self.target_tokens, self.overlap_tokens)
        if path.suffix.lower() in {".html", ".htm"}:
            parser = _HtmlMarkdownExtractor()
            parser.feed(path.read_text(encoding="utf-8-sig"))
            markdown = parser.markdown()
            if not markdown.strip():
                raise ValueError("HTML source produced no indexable text")
            return chunk_markdown(record, markdown,
                                  self.target_tokens, self.overlap_tokens)
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore[import-not-found]
            except ModuleNotFoundError as exc:
                raise RuntimeError("pypdf is required to ingest PDF sources") from exc
            pages = [(page.extract_text() or "") for page in PdfReader(str(path)).pages]
            return chunk_pages(record, pages, self.target_tokens, self.overlap_tokens)
        raise ValueError("only Markdown, text, HTML, and PDF sources are supported")

    @staticmethod
    def _write_lexical(path: Path, chunks: list[Chunk]) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, tokens)")
            connection.executemany(
                "INSERT INTO chunks_fts(chunk_id,tokens) VALUES (?,?)",
                ((item.chunk_id, " ".join(token.lower() for token in tokens(item.text)))
                 for item in chunks),
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _write_faiss(path: Path, vectors: list[list[float]]) -> int:
        try:
            import faiss  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("faiss-cpu and numpy are required to build the dense index") from exc
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("embedding model returned an invalid matrix")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(path))
        return int(matrix.shape[1])

    def build(self, manifest_path: Path, source_root: Path,
              output_root: Path) -> Path:
        records = self._load_manifest(manifest_path)
        chunks = [chunk for record in records for chunk in self._read(record, source_root)]
        if not chunks:
            raise ValueError("knowledge corpus produced no chunks")
        vectors = self.embedding.encode_passages([item.text for item in chunks])
        identity = {
            "schema_version": 1,
            "sources": [item.__dict__ for item in records],
            "chunker": {"version": 1, "target_tokens": self.target_tokens,
                        "overlap_tokens": self.overlap_tokens},
            "embedding": {"model": self.embedding.model_id,
                          "revision": self.embedding.revision},
            "reranker": {"model": self.reranker_model,
                         "revision": self.reranker_revision},
            "thresholds": {"reranker_min_score": self.reranker_min_score,
                           "all_scope_reranker_min_score":
                               self.all_scope_reranker_min_score,
                           "rrf_min_score": self.rrf_min_score},
        }
        version = "knowledge-" + hashlib.sha256(
            canonical_json(identity).encode("utf-8")).hexdigest()[:16]
        output = output_root / version
        if output.exists():
            manifest_file = output / "manifest.json"
            if not manifest_file.is_file():
                raise ValueError("existing knowledge index is incomplete")
            existing = json.loads(manifest_file.read_text(encoding="utf-8"))
            if existing.get("index_version") != version:
                raise ValueError("existing knowledge index version mismatch")
            if any(existing.get(name) != value for name, value in identity.items()):
                raise ValueError("existing knowledge index identity mismatch")
            for name, metadata in existing.get("files", {}).items():
                path = output / name
                if (not path.is_file() or path.stat().st_size != metadata["size"] or
                        hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]):
                    raise ValueError("existing knowledge index checksum mismatch")
            return output
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=output_root))
        try:
            (temporary / "sources.json").write_text(
                canonical_json([record.__dict__ for record in records]) + "\n",
                encoding="utf-8")
            (temporary / "chunks.jsonl").write_text(
                "".join(canonical_json(item.json()) + "\n" for item in chunks),
                encoding="utf-8")
            self._write_lexical(temporary / "lexical.sqlite", chunks)
            dimension = self._write_faiss(temporary / "vectors.faiss", vectors)
            manifest = {
                **identity,
                "index_version": version,
                "created_at": max(item.ingested_at for item in records),
                "chunk_count": len(chunks),
                "embedding_dimension": dimension,
                "files": {},
            }
            for name in ("sources.json", "chunks.jsonl", "lexical.sqlite",
                         "vectors.faiss"):
                data = (temporary / name).read_bytes()
                manifest["files"][name] = {
                    "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            (temporary / "manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8")
            os.replace(temporary, output)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return output
