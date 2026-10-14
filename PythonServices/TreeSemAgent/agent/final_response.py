"""Decode final answers without treating malformed envelopes as prose."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedFinalResponse:
    answer: str | None
    prediction_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    error: str | None = None


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate final field")
        result[key] = value
    return result


def parse_final_response(content: str | None) -> ParsedFinalResponse:
    if not content:
        return ParsedFinalResponse(content)
    candidate = content.strip()
    if candidate.startswith("```json") and candidate.endswith("```"):
        candidate = candidate[7:-3].strip()
    envelope_marker = re.search(
        r'"(?:answer|grounding_prediction_ids|grounding_source_ids)"\s*:',
        candidate) is not None
    if not candidate.startswith(("{", "[")) and not envelope_marker:
        # Compatibility only: ordinary prose is still checked by ResponsePolicy.
        return ParsedFinalResponse(content)
    try:
        value = json.loads(candidate, object_pairs_hook=_unique_object)
        if not isinstance(value, dict) or set(value) != {
                "answer", "grounding_prediction_ids", "grounding_source_ids"}:
            raise ValueError("invalid final fields")
        answer = value["answer"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("invalid final answer")
        for key in ("grounding_prediction_ids", "grounding_source_ids"):
            ids = value[key]
            if not isinstance(ids, list) or any(
                    not isinstance(item, str) for item in ids):
                raise ValueError("invalid grounding array")
        return ParsedFinalResponse(answer, tuple(value["grounding_prediction_ids"]),
                                   tuple(value["grounding_source_ids"]))
    except (ValueError, TypeError):
        # Never pass the malformed upstream body to the caller or repair prompt.
        return ParsedFinalResponse(None, error="invalid_final_response")
