from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
import re
from typing import Any


ALLOWED_AUDIENCES = {"patient", "doctor", "both"}
ALLOWED_SCOPES = {
    "model_public",
    "model_technical",
    "clinical_patient",
    "clinical_professional",
}
_SOURCE_ID = re.compile(r"^src_[A-Za-z0-9_-]{1,76}$")


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    publisher: str
    source_type: str
    audience: str
    knowledge_scope: str
    version: str
    published_at: str
    original_url: str
    local_path: str
    license_note: str
    sha256: str
    ingested_at: str

    @classmethod
    def parse(cls, value: dict[str, Any]) -> "SourceRecord":
        if set(value) != set(cls.__annotations__):
            raise ValueError("source manifest has unknown or missing fields")
        record = cls(**{key: str(item) for key, item in value.items()})
        if not _SOURCE_ID.fullmatch(record.source_id):
            raise ValueError("invalid source_id")
        if record.audience not in ALLOWED_AUDIENCES:
            raise ValueError("invalid source audience")
        if record.knowledge_scope not in ALLOWED_SCOPES:
            raise ValueError("invalid knowledge scope")
        if len(record.sha256) != 64 or any(c not in "0123456789abcdef" for c in record.sha256):
            raise ValueError("invalid source sha256")
        limits = {"title": 300, "publisher": 200, "source_type": 64,
                  "version": 128, "published_at": 64,
                  "original_url": 2048, "local_path": 512,
                  "license_note": 1000, "ingested_at": 64}
        for name, maximum in limits.items():
            text = getattr(record, name)
            if not text.strip() or len(text) > maximum:
                raise ValueError(f"invalid source {name}")
        relative = PurePosixPath(record.local_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid source local_path")
        if not (record.original_url.startswith("https://") or
                record.original_url.startswith("treesem://")):
            raise ValueError("invalid source URL")
        return record


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    citation_id: str
    source_id: str
    title: str
    section: str
    page: int | None
    text: str
    audience: str
    knowledge_scope: str
    publisher: str
    published_at: str
    url: str
    content_sha256: str

    def json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float

    def public_json(self, excerpt_limit: int = 800) -> dict[str, Any]:
        value = self.chunk.json()
        value.pop("chunk_id")
        value.pop("audience")
        value.pop("knowledge_scope")
        value["excerpt"] = value.pop("text")[:excerpt_limit]
        value["score"] = self.score
        return value


@dataclass(frozen=True)
class SearchResponse:
    index_version: str
    retrieval_mode: str
    hits: tuple[SearchHit, ...]

    def json(self) -> dict[str, Any]:
        return {
            "index_version": self.index_version,
            "retrieval_mode": self.retrieval_mode,
            "results": [hit.public_json() for hit in self.hits],
        }
