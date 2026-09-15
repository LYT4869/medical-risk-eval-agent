from __future__ import annotations

import re
from dataclasses import dataclass


_PREDICTION_ID = re.compile(
    r"(?<![A-Za-z0-9_])pred_[0-9a-f]{32}(?![A-Za-z0-9_])")
_SESSION_ID = re.compile(
    r"(?<![A-Za-z0-9_])ses_[0-9a-f]{32}(?![A-Za-z0-9_])")
_SAMPLE_INDEX = re.compile(
    r"(?P<label>样本(?:索引)?|sample(?:\s+index)?)"
    r"(?P<separator>\s*(?:#\s*)?)(?P<value>[0-9]+)",
    re.IGNORECASE)
_CHINESE_NUMBERED_SAMPLE = re.compile(
    r"第\s*(?P<value>[0-9]+)\s*号?(?:演示)?样本")
_ENGLISH_WORD_SAMPLE = re.compile(
    r"(?P<label>(?:demonstration\s+)?sample(?:\s+index)?)\s+"
    r"(?P<word>zero|one|two|three|four|five|six|seven|eight|nine)",
    re.IGNORECASE)
_ENGLISH_DIGITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}")
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])")
_API_KEY = re.compile(
    r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])",
    re.IGNORECASE)
_MAX_CANDIDATES = 8


@dataclass(frozen=True)
class ReferenceExtraction:
    prediction_ids: tuple[str, ...]
    sample_indexes: tuple[int, ...]
    router_message: str


def _replace_prediction_ids(message: str) -> tuple[str, tuple[str, ...]]:
    candidates: list[str] = []

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if value in candidates:
            index = candidates.index(value)
        elif len(candidates) < _MAX_CANDIDATES:
            candidates.append(value)
            index = len(candidates) - 1
        else:
            return value
        return f"<prediction_ref_{index}>"

    return _PREDICTION_ID.sub(replace, message), tuple(candidates)


def _replace_sample_indexes(message: str) -> tuple[str, tuple[int, ...]]:
    candidates: list[int] = []

    def placeholder(value: int) -> str | None:
        if value in candidates:
            index = candidates.index(value)
        elif len(candidates) < _MAX_CANDIDATES:
            candidates.append(value)
            index = len(candidates) - 1
        else:
            return None
        return f"<sample_ref_{index}>"

    def replace_chinese(match: re.Match[str]) -> str:
        replacement = placeholder(int(match.group("value")))
        return (f"演示样本 {replacement}" if replacement is not None
                else match.group(0))

    def replace_digits(match: re.Match[str]) -> str:
        replacement = placeholder(int(match.group("value")))
        return (f"{match.group('label')} {replacement}"
                if replacement is not None else match.group(0))

    def replace_word(match: re.Match[str]) -> str:
        replacement = placeholder(_ENGLISH_DIGITS[match.group("word").lower()])
        return (f"{match.group('label')} {replacement}"
                if replacement is not None else match.group(0))

    result = _CHINESE_NUMBERED_SAMPLE.sub(replace_chinese, message)
    result = _SAMPLE_INDEX.sub(replace_digits, result)
    result = _ENGLISH_WORD_SAMPLE.sub(replace_word, result)
    return result, tuple(candidates)


def sanitize_router_context_text(text: str) -> str:
    sanitized = _BEARER_TOKEN.sub("<redacted_credential>", text)
    sanitized = _JWT.sub("<redacted_credential>", sanitized)
    sanitized = _API_KEY.sub("<redacted_credential>", sanitized)
    sanitized = _PREDICTION_ID.sub("<redacted_prediction_id>", sanitized)
    return _SESSION_ID.sub("<redacted_session_id>", sanitized)


def extract_references(message: str) -> ReferenceExtraction:
    masked, prediction_ids = _replace_prediction_ids(message)
    masked, sample_indexes = _replace_sample_indexes(masked)
    return ReferenceExtraction(
        prediction_ids=prediction_ids,
        sample_indexes=sample_indexes,
        router_message=sanitize_router_context_text(masked),
    )
