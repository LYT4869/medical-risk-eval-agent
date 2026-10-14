from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agent.embedding_provider import SentenceTransformerEmbeddingProvider
from agent.onnx_embedding_provider import OnnxEmbeddingProvider
from agent.routing import RuleRouter, SafetyGate
from agent.routing_artifact import (
    canonical_intent_examples,
    load_routing_artifact,
)
from agent.semantic_routing import RoutingThresholds, SemanticScorer
from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry
from evaluation.run_routing_evaluation import (
    RoutingCase,
    evaluate_cases,
    load_cases,
)


@dataclass(frozen=True)
class CaseObservation:
    case_id: str
    route_scope: str
    query_embedding: tuple[float, ...]
    top_similarity: float | None
    margin: float | None


@dataclass(frozen=True)
class BackendSnapshot:
    intent_embeddings: tuple[tuple[float, ...], ...]
    cases: tuple[CaseObservation, ...]


def _matrix(value: object) -> tuple[tuple[float, ...], ...]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    try:
        matrix = tuple(tuple(float(item) for item in row) for row in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("backend embedding matrix is invalid") from exc
    if (not matrix or not matrix[0] or
            any(len(row) != len(matrix[0]) for row in matrix)):
        raise ValueError("backend embedding matrix shape is invalid")
    if any(not math.isfinite(item) for row in matrix for item in row):
        raise ValueError("backend embeddings must be finite")
    if any(sum(item * item for item in row) <= 0 for row in matrix):
        raise ValueError("backend embeddings must have positive norm")
    return matrix


def _route_observation(
        case: RoutingCase, scorer: SemanticScorer,
        rules: RuleRouter, safety: SafetyGate,
        query_embedding: tuple[float, ...]) -> CaseObservation:
    safety_decision = safety.evaluate(case.message)
    if not safety_decision.allowed:
        scope = (safety_decision.refusal_scope.value
                 if safety_decision.refusal_scope is not None else "unknown")
        return CaseObservation(case.case_id, scope, query_embedding, None, None)
    rule = rules.route(case.message)
    if rule is not None:
        return CaseObservation(
            case.case_id, rule.scope.value, query_embedding, None, None)
    scores = scorer.score(case.message)
    decision = scorer.route(case.message)
    return CaseObservation(
        case.case_id, decision.scope.value, query_embedding,
        scores.top_similarity, scores.margin)


def collect_snapshot(
        cases: Sequence[RoutingCase], registry: TaskRegistry,
        provider: object, thresholds: RoutingThresholds) -> BackendSnapshot:
    examples = canonical_intent_examples(registry)
    intent_embeddings = _matrix(provider.encode_examples(examples))
    scorer = SemanticScorer(registry, provider, thresholds)
    rules = RuleRouter(registry)
    safety = SafetyGate()
    observations = []
    for case in cases:
        query = _matrix([provider.encode_query(case.message)])[0]
        observations.append(_route_observation(
            case, scorer, rules, safety, query))
    return BackendSnapshot(intent_embeddings, tuple(observations))


def _paired_vectors(
        golden: BackendSnapshot,
        candidate: BackendSnapshot) -> list[
            tuple[tuple[float, ...], tuple[float, ...]]]:
    pairs = list(zip(golden.intent_embeddings, candidate.intent_embeddings))
    pairs.extend((left.query_embedding, right.query_embedding)
                 for left, right in zip(golden.cases, candidate.cases))
    for left, right in pairs:
        if len(left) != len(right):
            raise ValueError("backend embedding dimensions are not aligned")
        if any(not math.isfinite(item) for item in left + right):
            raise ValueError("backend embeddings must be finite")
        if (sum(item * item for item in left) <= 0 or
                sum(item * item for item in right) <= 0):
            raise ValueError("backend embeddings must have positive norm")
    return pairs


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    return numerator / (left_norm * right_norm)


def compare_snapshots(
        cases: Sequence[RoutingCase], golden: BackendSnapshot,
        candidate: BackendSnapshot, *, embedding_backend: str,
        artifact_version: str, required_case_count: int,
        maximum_embedding_delta: float,
        minimum_embedding_cosine: float) -> dict[str, object]:
    if len(cases) != required_case_count:
        raise ValueError("frozen parity set case count mismatch")
    case_ids = [case.case_id for case in cases]
    if ([item.case_id for item in golden.cases] != case_ids or
            [item.case_id for item in candidate.cases] != case_ids):
        raise ValueError("backend case alignment mismatch")
    if len(golden.intent_embeddings) != len(candidate.intent_embeddings):
        raise ValueError("backend intent embedding count mismatch")
    pairs = _paired_vectors(golden, candidate)
    absolute_delta = max(
        abs(left_item - right_item)
        for left, right in pairs
        for left_item, right_item in zip(left, right))
    minimum_cosine = min(_cosine(left, right) for left, right in pairs)

    mismatches = []
    top_deltas = []
    margin_deltas = []
    candidate_routes = {}
    for left, right in zip(golden.cases, candidate.cases):
        candidate_routes[right.case_id] = right.route_scope
        if left.route_scope != right.route_scope:
            mismatches.append({
                "case_id": left.case_id,
                "golden_scope": left.route_scope,
                "candidate_scope": right.route_scope,
            })
        if ((left.top_similarity is None) !=
                (right.top_similarity is None) or
                (left.margin is None) != (right.margin is None)):
            raise ValueError("backend semantic score alignment mismatch")
        if left.top_similarity is not None:
            if (not math.isfinite(left.top_similarity) or
                    not math.isfinite(right.top_similarity)):
                raise ValueError("backend semantic scores must be finite")
            top_deltas.append(abs(
                left.top_similarity - right.top_similarity))
        if left.margin is not None:
            if (not math.isfinite(left.margin) or
                    not math.isfinite(right.margin)):
                raise ValueError("backend semantic scores must be finite")
            margin_deltas.append(abs(left.margin - right.margin))

    quality = evaluate_cases(
        list(cases), lambda case: candidate_routes[case.case_id])
    hard_quality = (
        quality["unknown_recall"] == 1.0 and
        quality["compositional_fallback_recall"] == 1.0 and
        quality["safety_accuracy"] == 1.0)
    passed = (
        not mismatches and hard_quality and
        absolute_delta <= maximum_embedding_delta and
        minimum_cosine >= minimum_embedding_cosine)
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "embedding_backend": embedding_backend,
        "artifact_version": artifact_version,
        "maximum_embedding_absolute_delta": absolute_delta,
        "minimum_embedding_cosine_similarity": minimum_cosine,
        "maximum_top_similarity_delta": max(top_deltas, default=0.0),
        "maximum_margin_delta": max(margin_deltas, default=0.0),
        "route_match_count": len(cases) - len(mismatches),
        "route_mismatch_count": len(mismatches),
        "route_mismatches": mismatches,
        "quality_metrics": quality,
        "gates": {
            "maximum_embedding_delta": maximum_embedding_delta,
            "minimum_embedding_cosine": minimum_embedding_cosine,
            "hard_quality_passed": hard_quality,
        },
        "passed": passed,
    }


def _load_thresholds(path: Path, model: str,
                     revision: str) -> RoutingThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != 1 or
            payload.get("model") != model or
            payload.get("revision") != revision):
        raise ValueError("routing thresholds do not match model")
    return RoutingThresholds(**payload["thresholds"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare treeSem Agent routing embedding backends")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--embedding-backend",
                        choices=("onnx_fp32", "onnx_int8"), required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    cases = load_cases(arguments.cases)
    registry = TaskRegistry.load(arguments.tasks, SUPPORTED_DOMAIN_TOOLS)
    thresholds = _load_thresholds(
        arguments.thresholds, arguments.model, arguments.revision)
    artifact = load_routing_artifact(
        arguments.artifact_dir,
        task_registry_path=arguments.tasks,
        thresholds_path=arguments.thresholds,
        registry=registry,
        expected_backend=arguments.embedding_backend,
        expected_model_id=arguments.model,
        expected_revision=arguments.revision)
    golden = collect_snapshot(
        cases, registry,
        SentenceTransformerEmbeddingProvider(
            arguments.model, arguments.revision), thresholds)
    candidate = collect_snapshot(
        cases, registry, OnnxEmbeddingProvider(artifact), thresholds)
    if arguments.embedding_backend == "onnx_fp32":
        maximum_delta, minimum_cosine = 1e-4, 0.9999
    else:
        maximum_delta, minimum_cosine = 0.05, 0.98
    report = compare_snapshots(
        cases, golden, candidate,
        embedding_backend=arguments.embedding_backend,
        artifact_version=artifact.manifest.artifact_version,
        required_case_count=150,
        maximum_embedding_delta=maximum_delta,
        minimum_embedding_cosine=minimum_cosine)
    rendered = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
