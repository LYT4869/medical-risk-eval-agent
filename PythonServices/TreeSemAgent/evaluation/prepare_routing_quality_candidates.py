from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


CASE_FIELDS = {
    "message", "expected_scope", "category", "style", "language",
    "rationale",
}
KNOWN_STYLE_COUNTS = {
    "paraphrase": 8,
    "colloquial": 8,
    "elliptical": 6,
    "noisy": 6,
    "boundary": 8,
}
KNOWN_LANGUAGE_COUNTS = {"zh": 24, "en": 6, "mixed": 6}
THIRTY_LANGUAGE_COUNTS = {"zh": 18, "en": 6, "mixed": 6}
UNKNOWN_STYLE_COUNTS = {
    "ood": 9,
    "colloquial": 6,
    "elliptical": 5,
    "noisy": 5,
    "boundary": 5,
}


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    category: str
    expected_scope: str
    count: int
    style_counts: dict[str, int]
    language_counts: dict[str, int]


BATCH_SPECS = tuple([
    BatchSpec(
        f"known_{scope}", "known", scope, 36,
        KNOWN_STYLE_COUNTS, KNOWN_LANGUAGE_COUNTS)
    for scope in (
        "prediction", "summary", "explanation", "history", "comparison",
        "knowledge")
] + [
    BatchSpec(
        "unknown_a", "unknown", "unknown", 30,
        UNKNOWN_STYLE_COUNTS, THIRTY_LANGUAGE_COUNTS),
    BatchSpec(
        "unknown_b", "unknown", "unknown", 30,
        UNKNOWN_STYLE_COUNTS, THIRTY_LANGUAGE_COUNTS),
    BatchSpec(
        "compositional_a", "compositional", "unknown", 30,
        {"multi_intent": 30}, THIRTY_LANGUAGE_COUNTS),
    BatchSpec(
        "compositional_b", "compositional", "unknown", 30,
        {"multi_intent": 30}, THIRTY_LANGUAGE_COUNTS),
    BatchSpec(
        "safety_security", "safety", "security_abuse", 30,
        {"adversarial": 30}, THIRTY_LANGUAGE_COUNTS),
    BatchSpec(
        "safety_medical", "safety", "medical_refusal", 30,
        {"adversarial": 30}, THIRTY_LANGUAGE_COUNTS),
])


def normalized_message(message: str) -> str:
    normalized = unicodedata.normalize("NFKC", message).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized)
    return " ".join(without_punctuation.split())


def _text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    result = value.strip()
    if len(result) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    return result


def validate_batch(payload: object, spec: BatchSpec) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or set(payload) != {
            "schema_version", "batch_id", "cases"}:
        raise ValueError(f"invalid envelope fields for {spec.batch_id}")
    if payload["schema_version"] != 1 or payload["batch_id"] != spec.batch_id:
        raise ValueError(f"invalid schema or batch id for {spec.batch_id}")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or len(raw_cases) != spec.count:
        raise ValueError(
            f"{spec.batch_id} requires exactly {spec.count} cases")
    result: list[dict[str, str]] = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict) or set(raw) != CASE_FIELDS:
            raise ValueError(
                f"invalid case fields in {spec.batch_id} at index {index}")
        if raw["category"] != spec.category:
            raise ValueError(f"category mismatch in {spec.batch_id}")
        if raw["expected_scope"] != spec.expected_scope:
            raise ValueError(f"expected_scope mismatch in {spec.batch_id}")
        result.append({
            "message": _text(raw["message"], "message", 500),
            "expected_scope": raw["expected_scope"],
            "category": raw["category"],
            "style": _text(raw["style"], "style", 32),
            "language": _text(raw["language"], "language", 16),
            "rationale": _text(raw["rationale"], "rationale", 80),
        })
    styles = Counter(item["style"] for item in result)
    languages = Counter(item["language"] for item in result)
    if styles != Counter(spec.style_counts):
        raise ValueError(
            f"style quota mismatch in {spec.batch_id}: {dict(styles)}")
    if languages != Counter(spec.language_counts):
        raise ValueError(
            f"language quota mismatch in {spec.batch_id}: {dict(languages)}")
    return result


def _frozen_messages(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
            payload.get("cases"), list):
        raise ValueError("invalid frozen routing corpus")
    return {
        normalized_message(item["message"])
        for item in payload["cases"]
    }


SENSITIVE_PATTERNS = {
    "email": re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),
    "mainland_phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "mainland_identity": re.compile(
        r"(?<!\d)\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"),
}


def _sensitive_findings(
        cases: list[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for case in cases:
        for name, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(case["message"]):
                result.append({
                    "candidate_id": case["candidate_id"],
                    "finding": name,
                })
    return result


def prepare_candidates(
        raw_directory: Path,
        frozen_corpus: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    expected_files = {f"{spec.batch_id}.json" for spec in BATCH_SPECS}
    actual_files = {path.name for path in raw_directory.glob("*.json")}
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        extra = sorted(actual_files - expected_files)
        raise ValueError(f"batch file mismatch: missing={missing}, extra={extra}")

    result: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    frozen = _frozen_messages(frozen_corpus)
    for spec in BATCH_SPECS:
        path = raw_directory / f"{spec.batch_id}.json"
        rows = validate_batch(
            json.loads(path.read_text(encoding="utf-8")), spec)
        for row in rows:
            normalized = normalized_message(row["message"])
            if normalized in seen:
                raise ValueError(
                    f"duplicate candidate messages in {seen[normalized]} "
                    f"and {spec.batch_id}")
            if normalized in frozen:
                raise ValueError(
                    f"candidate in {spec.batch_id} overlaps frozen corpus")
            seen[normalized] = spec.batch_id
            result.append({
                "candidate_id": f"rq_{len(result) + 1:04d}",
                **row,
            })

    findings = _sensitive_findings(result)
    if findings:
        raise ValueError(
            f"potential sensitive data found in {len(findings)} candidates")
    rendered = json.dumps(
        {"schema_version": 1, "cases": result},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report: dict[str, object] = {
        "schema_version": 1,
        "candidate_count": len(result),
        "candidate_sha256": hashlib.sha256(
            rendered.encode("utf-8")).hexdigest(),
        "batch_count": len(BATCH_SPECS),
        "category_counts": dict(sorted(Counter(
            item["category"] for item in result).items())),
        "scope_counts": dict(sorted(Counter(
            item["expected_scope"] for item in result).items())),
        "language_counts": dict(sorted(Counter(
            item["language"] for item in result).items())),
        "style_counts": dict(sorted(Counter(
            item["style"] for item in result).items())),
        "exact_duplicate_count": 0,
        "frozen_overlap_count": 0,
        "sensitive_finding_count": 0,
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and combine treeSem routing quality candidates")
    parser.add_argument("--raw-directory", required=True, type=Path)
    parser.add_argument("--frozen-corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    cases, report = prepare_candidates(
        args.raw_directory, args.frozen_corpus)
    output = {"schema_version": 1, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
