from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

from agent.embedding_provider import SentenceTransformerEmbeddingProvider
from agent.routing import RuleRouter, SafetyGate
from agent.routing_types import RequestScope
from agent.semantic_routing import (
    RoutingThresholds,
    SemanticScorer,
    SemanticScores,
)
from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry
from evaluation.run_routing_evaluation import RoutingCase, load_cases


@dataclass(frozen=True)
class RoutingObservation:
    category: str
    expected_scope: RequestScope
    scores: SemanticScores | None
    fixed_scope: RequestScope | None = None


def apply_thresholds(observation: RoutingObservation,
                     thresholds: RoutingThresholds) -> RequestScope:
    if observation.fixed_scope is not None:
        return observation.fixed_scope
    if observation.scores is None:
        return RequestScope.UNKNOWN
    scores = observation.scores
    if scores.top_similarity < thresholds.min_similarity:
        return RequestScope.UNKNOWN
    if scores.margin < thresholds.min_margin:
        return RequestScope.UNKNOWN
    if scores.secondary_similarity >= thresholds.secondary_intent_similarity:
        return RequestScope.UNKNOWN
    return scores.top_scope


def _candidate_boundaries(values: list[float], lower: float,
                          upper: float) -> list[float]:
    unique = sorted(set(values))
    candidates = {lower, upper}
    for left, right in zip(unique, unique[1:]):
        candidates.add((left + right) / 2.0)
    for value in unique:
        candidates.add(min(upper, max(lower, math.nextafter(value, upper))))
    return sorted(candidates)


def _quality(observations: list[RoutingObservation],
             thresholds: RoutingThresholds) -> dict[str, float | int]:
    actual = [apply_thresholds(item, thresholds) for item in observations]
    known = [i for i, item in enumerate(observations)
             if item.category == "known"]
    fallback = [i for i, item in enumerate(observations)
                if item.category in {"unknown", "compositional"}]
    false_deterministic = sum(
        actual[i] != RequestScope.UNKNOWN for i in fallback)
    return {
        "known_accuracy": (
            sum(actual[i] == observations[i].expected_scope for i in known) /
            len(known) if known else 1.0),
        "fallback_recall": (
            sum(actual[i] == RequestScope.UNKNOWN for i in fallback) /
            len(fallback) if fallback else 1.0),
        "false_deterministic": false_deterministic,
    }


def _stable_threshold(value: float, *, upper_boundary: bool) -> float:
    """Move a learned boundary to a reproducible six-decimal grid.

    Minimum similarity and margin are lower bounds, so rounding them down
    preserves accepted calibration examples.  Secondary similarity is an
    upper rejection boundary, so rounding it up provides the same protection.
    This also leaves enough numeric clearance for normal PyTorch/ONNX drift.
    """
    rounding = ROUND_CEILING if upper_boundary else ROUND_FLOOR
    return float(Decimal(str(value)).quantize(
        Decimal("0.000001"), rounding=rounding))


def _stabilize_thresholds(thresholds: RoutingThresholds) -> RoutingThresholds:
    return RoutingThresholds(
        min_similarity=_stable_threshold(
            thresholds.min_similarity, upper_boundary=False),
        min_margin=_stable_threshold(
            thresholds.min_margin, upper_boundary=False),
        secondary_intent_similarity=_stable_threshold(
            thresholds.secondary_intent_similarity, upper_boundary=True),
    )


def choose_thresholds(
        observations: list[RoutingObservation], *,
        minimum_known_accuracy: float,
        minimum_fallback_recall: float) -> RoutingThresholds:
    scored = [item.scores for item in observations if item.scores is not None]
    similarity_candidates = _candidate_boundaries(
        [item.top_similarity for item in scored], -1.0, 1.0)
    margin_candidates = _candidate_boundaries(
        [item.margin for item in scored], 0.0, 2.0)
    secondary_candidates = _candidate_boundaries(
        [item.secondary_similarity for item in scored], -1.0, 1.0)

    best: tuple[tuple[float, ...], RoutingThresholds] | None = None
    for minimum, margin, secondary in itertools.product(
            similarity_candidates, margin_candidates, secondary_candidates):
        thresholds = _stabilize_thresholds(
            RoutingThresholds(minimum, margin, secondary))
        quality = _quality(observations, thresholds)
        known_accuracy = float(quality["known_accuracy"])
        fallback_recall = float(quality["fallback_recall"])
        if (known_accuracy + 1e-12 < minimum_known_accuracy or
                fallback_recall + 1e-12 < minimum_fallback_recall):
            continue
        rank = (
            -float(quality["false_deterministic"]),
            known_accuracy,
            fallback_recall,
            minimum,
            margin,
            -secondary,
        )
        if best is None or rank > best[0]:
            best = (rank, thresholds)
    if best is None:
        raise ValueError("no thresholds satisfy calibration constraints")
    return best[1]


def build_observations(cases: list[RoutingCase], scorer: SemanticScorer,
                       rules: RuleRouter,
                       safety: SafetyGate) -> list[RoutingObservation]:
    result = []
    for case in cases:
        expected = RequestScope(case.expected_scope)
        safety_decision = safety.evaluate(case.message)
        if not safety_decision.allowed:
            result.append(RoutingObservation(
                case.category, expected, None,
                safety_decision.refusal_scope or RequestScope.UNKNOWN))
            continue
        rule = rules.route(case.message)
        if rule is not None:
            result.append(RoutingObservation(
                case.category, expected, None, rule.scope))
            continue
        result.append(RoutingObservation(
            case.category, expected, scorer.score(case.message)))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Agent semantic routing")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-fallback-recall", type=float, default=0.95)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    calibration = [case for case in cases if case.split == "calibration"]
    registry = TaskRegistry.load(args.tasks, SUPPORTED_DOMAIN_TOOLS)
    provider = SentenceTransformerEmbeddingProvider(args.model, args.revision)
    permissive = RoutingThresholds(-1.0, 0.0, 1.0)
    scorer = SemanticScorer(registry, provider, permissive)
    observations = build_observations(
        calibration, scorer, RuleRouter(registry), SafetyGate())

    rule_known_accuracy = float(_quality(
        [RoutingObservation(item.category, item.expected_scope, None,
                            item.fixed_scope)
         for item in observations], permissive)["known_accuracy"])
    target_met = True
    try:
        thresholds = choose_thresholds(
            observations,
            minimum_known_accuracy=rule_known_accuracy,
            minimum_fallback_recall=args.minimum_fallback_recall)
    except ValueError:
        target_met = False
        thresholds = choose_thresholds(
            observations,
            minimum_known_accuracy=rule_known_accuracy,
            minimum_fallback_recall=0.0)

    quality = _quality(observations, thresholds)
    raw = args.cases.read_bytes()
    result = {
        "schema_version": 1,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "model": args.model,
        "revision": args.revision,
        "thresholds": asdict(thresholds),
        "calibration": quality,
        "fallback_target": args.minimum_fallback_recall,
        "fallback_target_met": target_met,
        "rule_known_accuracy": rule_known_accuracy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
