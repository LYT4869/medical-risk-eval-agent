from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


KNOWN_SCOPES = (
    "prediction", "summary", "explanation", "history", "comparison",
    "knowledge",
)
FREEZE_SEED = "treesem-routing-quality-v1"


def _split_group(case: dict[str, object]) -> tuple[str, str]:
    category = str(case["category"])
    scope = str(case["expected_scope"])
    if category == "known":
        return category, scope
    if category == "safety":
        return category, scope
    return category, category


def _calibration_count(group: tuple[str, str]) -> int:
    if group[0] == "known":
        return 10
    if group[0] == "safety":
        return 10
    return 20


def _rank(candidate_id: str) -> str:
    return hashlib.sha256(
        f"{FREEZE_SEED}:{candidate_id}".encode("utf-8")).hexdigest()


def freeze_quality_set(
        candidates: list[dict[str, object]],
        excluded_ids: set[str]) -> tuple[list[dict[str, str]], dict[str, object]]:
    identifiers = [str(item["candidate_id"]) for item in candidates]
    if len(candidates) != 396 or len(set(identifiers)) != 396:
        raise ValueError("quality candidate set must contain 396 unique cases")
    if len(excluded_ids) != 36 or not excluded_ids.issubset(set(identifiers)):
        raise ValueError("review must exclude exactly 36 existing candidates")
    excluded = [item for item in candidates
                if str(item["candidate_id"]) in excluded_ids]
    if any(item["category"] != "known" for item in excluded):
        raise ValueError("only known-intent redundancy candidates may be excluded")
    excluded_by_scope = Counter(str(item["expected_scope"]) for item in excluded)
    if excluded_by_scope != Counter({scope: 6 for scope in KNOWN_SCOPES}):
        raise ValueError("review must exclude six known candidates per scope")

    kept = [item for item in candidates
            if str(item["candidate_id"]) not in excluded_ids]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for item in kept:
        grouped[_split_group(item)].append(item)
    expected_sizes = {
        **{("known", scope): 30 for scope in KNOWN_SCOPES},
        ("unknown", "unknown"): 60,
        ("compositional", "compositional"): 60,
        ("safety", "security_abuse"): 30,
        ("safety", "medical_refusal"): 30,
    }
    if {group: len(rows) for group, rows in grouped.items()} != expected_sizes:
        raise ValueError("reviewed quality set has an invalid group distribution")

    calibration: set[str] = set()
    for group, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: (
            _rank(str(item["candidate_id"])), str(item["candidate_id"])))
        calibration.update(
            str(item["candidate_id"])
            for item in ordered[:_calibration_count(group)])

    counters: Counter[tuple[str, str]] = Counter()
    result = []
    for item in kept:
        group = _split_group(item)
        counters[group] += 1
        group_name = "_".join(group)
        result.append({
            "case_id": f"quality_{group_name}_{counters[group]:03d}",
            "split": (
                "calibration" if str(item["candidate_id"]) in calibration
                else "held_out"),
            "category": str(item["category"]),
            "message": str(item["message"]),
            "expected_scope": str(item["expected_scope"]),
        })

    rendered = json.dumps(
        {"schema_version": 1, "cases": result},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    report: dict[str, object] = {
        "schema_version": 1,
        "freeze_seed": FREEZE_SEED,
        "candidate_count": len(candidates),
        "excluded_count": len(excluded_ids),
        "frozen_count": len(result),
        "frozen_sha256": hashlib.sha256(
            rendered.encode("utf-8")).hexdigest(),
        "split_counts": dict(sorted(Counter(
            item["split"] for item in result).items())),
        "category_counts": dict(sorted(Counter(
            item["category"] for item in result).items())),
        "scope_counts": dict(sorted(Counter(
            item["expected_scope"] for item in result).items())),
        "excluded_by_scope": dict(sorted(excluded_by_scope.items())),
    }
    return result, report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a reviewed treeSem routing quality set")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--review-decisions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    candidate_payload = json.loads(
        args.candidates.read_text(encoding="utf-8"))
    decision_payload = json.loads(
        args.review_decisions.read_text(encoding="utf-8"))
    if (candidate_payload.get("schema_version") != 1 or
            decision_payload.get("schema_version") != 1):
        raise ValueError("unsupported quality set schema")
    excluded = decision_payload.get("excluded")
    if not isinstance(excluded, list) or any(
            not isinstance(item, dict) or set(item) != {"candidate_id", "reason"}
            or item["reason"] not in {"near_duplicate", "ambiguous", "unnatural"}
            for item in excluded):
        raise ValueError("invalid review decisions")
    cases, report = freeze_quality_set(
        candidate_payload["cases"],
        {str(item["candidate_id"]) for item in excluded})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        {"schema_version": 1, "cases": cases},
        ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
