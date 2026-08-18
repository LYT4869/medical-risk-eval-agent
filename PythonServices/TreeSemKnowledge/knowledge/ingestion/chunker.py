from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..models import Chunk, SourceRecord


_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]|[^\s]")


@dataclass(frozen=True)
class Section:
    heading: str
    page: int | None
    text: str


def tokens(value: str) -> list[str]:
    return _TOKEN.findall(value)


def _windows(items: list[str], target: int, overlap: int) -> list[str]:
    if target <= 0 or overlap < 0 or overlap >= target:
        raise ValueError("invalid chunk parameters")
    if not items:
        return []
    result: list[str] = []
    offset = 0
    while offset < len(items):
        result.append(" ".join(items[offset:offset + target]))
        if offset + target >= len(items):
            break
        offset += target - overlap
    return result


def _chunks(source: SourceRecord, sections: list[Section], target: int,
            overlap: int) -> list[Chunk]:
    result: list[Chunk] = []
    ordinal = 0
    for section in sections:
        for text in _windows(tokens(section.text), target, overlap):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            identity = f"{source.source_id}\0{section.heading}\0{section.page}\0{ordinal}\0{content_hash}"
            chunk_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            result.append(Chunk(
                chunk_id=f"chk_{chunk_hash}",
                citation_id=f"cite_{chunk_hash}",
                source_id=source.source_id,
                title=source.title,
                section=section.heading or source.title,
                page=section.page,
                text=text,
                audience=source.audience,
                knowledge_scope=source.knowledge_scope,
                publisher=source.publisher,
                published_at=source.published_at,
                url=source.original_url,
                content_sha256=content_hash,
            ))
            ordinal += 1
    return result


def chunk_markdown(source: SourceRecord, text: str, target: int = 400,
                   overlap: int = 60) -> list[Chunk]:
    sections: list[Section] = []
    heading = source.title
    body: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("#"):
            if " ".join(body).strip():
                sections.append(Section(heading, None, " ".join(body).strip()))
            heading = line.lstrip("#").strip() or source.title
            body = []
        else:
            body.append(line)
    if " ".join(body).strip():
        sections.append(Section(heading, None, " ".join(body).strip()))
    return _chunks(source, sections, target, overlap)

def chunk_pages(source: SourceRecord, pages: list[str], target: int = 400,
                overlap: int = 60) -> list[Chunk]:
    sections = [Section(f"Page {index}", index, text.strip())
                for index, text in enumerate(pages, 1) if text.strip()]
    return _chunks(source, sections, target, overlap)
