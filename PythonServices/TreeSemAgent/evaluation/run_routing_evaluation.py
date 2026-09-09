from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


BUSINESS_SCOPES = (
    "prediction", "summary", "explanation", "history", "comparison",
    "knowledge",
)
ALL_SCOPES = set(BUSINESS_SCOPES) | {
    "unknown", "security_abuse", "medical_refusal",
}
SPLITS = {"registry", "calibration", "held_out"}
CATEGORIES = {"known", "unknown", "compositional", "safety"}


@dataclass(frozen=True)
class RoutingCase:
    case_id: str
    split: str
    category: str
    message: str
    expected_scope: str


def normalized_message(message: str) -> str:
    return " ".join(message.casefold().split())


def load_cases(path: Path) -> list[RoutingCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or set(payload) != {
            "schema_version", "cases"}:
        raise ValueError("invalid routing corpus schema")
    result: list[RoutingCase] = []
    identifiers: set[str] = set()
    messages: set[str] = set()
    required = {"case_id", "split", "category", "message", "expected_scope"}
    for raw in payload["cases"]:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("invalid routing case fields")
        case = RoutingCase(**raw)
        if not case.case_id or case.case_id in identifiers:
            raise ValueError("duplicate or empty case id")
        if case.split not in SPLITS or case.category not in CATEGORIES:
            raise ValueError("invalid routing case split or category")
        if case.expected_scope not in ALL_SCOPES:
            raise ValueError("invalid expected scope")
        normalized = normalized_message(case.message)
        if not normalized or normalized in messages:
            raise ValueError("duplicate normalized message")
        identifiers.add(case.case_id)
        messages.add(normalized)
        result.append(case)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(_expected: Iterable[str], _actual: Iterable[str], label: str) -> float:
    expected = list(_expected)
    actual = list(_actual)
    true_positive = sum(e == label and a == label for e, a in zip(expected, actual))
    false_positive = sum(e != label and a == label for e, a in zip(expected, actual))
    false_negative = sum(e == label and a != label for e, a in zip(expected, actual))
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return _ratio(2 * precision * recall, precision + recall)


def _precision_recall(
        expected: list[str], actual: list[str], indexes: list[int],
        label: str) -> dict[str, float]:
    true_positive = sum(
        expected[i] == label and actual[i] == label for i in indexes)
    false_positive = sum(
        expected[i] != label and actual[i] == label for i in indexes)
    false_negative = sum(
        expected[i] == label and actual[i] != label for i in indexes)
    return {
        "precision": _ratio(true_positive, true_positive + false_positive),
        "recall": _ratio(true_positive, true_positive + false_negative),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def evaluate_cases(
        cases: list[RoutingCase],
        router: Callable[[RoutingCase], str]) -> dict[str, object]:
    actual: list[str] = []
    durations: list[float] = []
    for case in cases:
        started = time.perf_counter()
        scope = router(case)
        durations.append((time.perf_counter() - started) * 1000.0)
        if scope not in ALL_SCOPES:
            raise ValueError(f"router returned invalid scope: {scope}")
        actual.append(scope)

    expected = [case.expected_scope for case in cases]
    known_indexes = [i for i, case in enumerate(cases) if case.category == "known"]
    non_safety_indexes = [
        i for i, case in enumerate(cases) if case.category != "safety"]
    routed_indexes = [
        i for i in non_safety_indexes if actual[i] in BUSINESS_SCOPES]
    unknown_indexes = [
        i for i, case in enumerate(cases) if case.category == "unknown"]
    compositional_indexes = [
        i for i, case in enumerate(cases)
        if case.category == "compositional"]
    correctly_routed = sum(
        expected[i] == actual[i] for i in routed_indexes)
    deterministic_precision = _ratio(correctly_routed, len(routed_indexes))
    deterministic_coverage = _ratio(
        len(routed_indexes), len(non_safety_indexes))
    confusion: dict[str, dict[str, int]] = {}
    for wanted, observed in zip(expected, actual):
        confusion.setdefault(wanted, {})[observed] = (
            confusion.setdefault(wanted, {}).get(observed, 0) + 1)

    return {
        "case_count": len(cases),
        "known_exact_route_accuracy": _ratio(
            sum(expected[i] == actual[i] for i in known_indexes),
            len(known_indexes)),
        "business_macro_f1": sum(
            _f1((expected[i] for i in known_indexes),
                (actual[i] for i in known_indexes), scope)
            for scope in BUSINESS_SCOPES) / len(BUSINESS_SCOPES),
        "deterministic_precision": deterministic_precision,
        "deterministic_coverage": deterministic_coverage,
        "known_abstention_rate": _ratio(
            sum(actual[i] == "unknown" for i in known_indexes),
            len(known_indexes)),
        "known_misroute_rate": _ratio(
            sum(actual[i] in BUSINESS_SCOPES and actual[i] != expected[i]
                for i in known_indexes),
            len(known_indexes)),
        "unknown_forced_route_rate": _ratio(
            sum(actual[i] in BUSINESS_SCOPES for i in unknown_indexes),
            len(unknown_indexes)),
        "compositional_forced_route_rate": _ratio(
            sum(actual[i] in BUSINESS_SCOPES for i in compositional_indexes),
            len(compositional_indexes)),
        "per_scope_precision_recall": {
            scope: _precision_recall(expected, actual, known_indexes, scope)
            for scope in BUSINESS_SCOPES
        },
        # Retained for report compatibility. "Rule" here means the
        # deterministic routing layer, regardless of its implementation.
        "rule_precision": deterministic_precision,
        "rule_coverage": deterministic_coverage,
        "unknown_recall": _ratio(
            sum(actual[i] == "unknown" for i in unknown_indexes),
            len(unknown_indexes)),
        "compositional_fallback_recall": _ratio(
            sum(actual[i] == "unknown" for i in compositional_indexes),
            len(compositional_indexes)),
        "safety_accuracy": _ratio(
            sum(expected[i] == actual[i] for i, case in enumerate(cases)
                if case.category == "safety"),
            sum(case.category == "safety" for case in cases)),
        "p50_route_latency_ms": _percentile(durations, 0.50),
        "p95_route_latency_ms": _percentile(durations, 0.95),
        "confusion_matrix": confusion,
    }


def _current_rule(case: RoutingCase) -> str:
    from agent.run_guard import AgentRunGuard
    return AgentRunGuard.for_request(case.message).scope.value


def _rule(case: RoutingCase) -> str:
    from agent.routing import RuleRouter, SafetyGate

    safety = SafetyGate().evaluate(case.message)
    if not safety.allowed:
        return (safety.refusal_scope.value if safety.refusal_scope is not None
                else "unknown")
    decision = RuleRouter().route(case.message)
    return decision.scope.value if decision is not None else "unknown"


def _hybrid_scope(case: RoutingCase, scorer, rules, safety) -> str:
    from agent.routing_types import RequestScope

    safety_decision = safety.evaluate(case.message)
    if not safety_decision.allowed:
        return (safety_decision.refusal_scope or RequestScope.UNKNOWN).value
    rule = rules.route(case.message)
    if rule is not None:
        return rule.scope.value
    return scorer.route(case.message).scope.value


def create_evaluation_provider(
        backend: str, *, model: str, revision: str,
        artifact_dir: Path | None, tasks_path: Path,
        thresholds_path: Path, registry: object):
    if backend == "sentence_transformers":
        from agent.embedding_provider import \
            SentenceTransformerEmbeddingProvider
        return SentenceTransformerEmbeddingProvider(model, revision)
    if backend not in {"onnx_fp32", "onnx_int8"}:
        raise ValueError("unsupported routing embedding backend")
    if artifact_dir is None:
        raise ValueError("ONNX evaluation requires an artifact directory")
    from agent.onnx_embedding_provider import OnnxEmbeddingProvider
    from agent.routing_artifact import load_routing_artifact

    artifact = load_routing_artifact(
        artifact_dir,
        task_registry_path=tasks_path,
        thresholds_path=thresholds_path,
        registry=registry,
        expected_backend=backend,
        expected_model_id=model,
        expected_revision=revision,
    )
    return OnnxEmbeddingProvider(artifact)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate treeSem Agent routing")
    parser.add_argument(
        "--cases", type=Path,
        default=Path(__file__).with_name("routing_cases.json"))
    parser.add_argument("--router", choices=("current-rule", "rule", "hybrid"),
                        default="current-rule")
    parser.add_argument("--split", choices=tuple(sorted(SPLITS)) + ("all",),
                        default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument(
        "--embedding-backend",
        choices=("sentence_transformers", "onnx_fp32", "onnx_int8"),
        default="sentence_transformers")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.cases)
    if args.split != "all":
        cases = [case for case in cases if case.split == args.split]
    if args.router == "current-rule":
        router = _current_rule
    elif args.router == "rule":
        router = _rule
    else:
        if not all((args.tasks, args.thresholds, args.model, args.revision)):
            parser.error(
                "hybrid router requires --tasks, --thresholds, --model and --revision")
        from agent.routing import RuleRouter, SafetyGate
        from agent.semantic_routing import RoutingThresholds, SemanticScorer
        from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry

        threshold_payload = json.loads(args.thresholds.read_text(encoding="utf-8"))
        if (threshold_payload.get("schema_version") != 1 or
                threshold_payload.get("model") != args.model or
                threshold_payload.get("revision") != args.revision):
            raise ValueError("routing thresholds do not match model configuration")
        thresholds = RoutingThresholds(**threshold_payload["thresholds"])
        registry = TaskRegistry.load(args.tasks, SUPPORTED_DOMAIN_TOOLS)
        scorer = SemanticScorer(
            registry,
            create_evaluation_provider(
                args.embedding_backend,
                model=args.model,
                revision=args.revision,
                artifact_dir=args.artifact_dir,
                tasks_path=args.tasks,
                thresholds_path=args.thresholds,
                registry=registry),
            thresholds)
        rules = RuleRouter(registry)
        safety = SafetyGate()
        router = lambda case: _hybrid_scope(
            case, scorer, rules, safety)
    report = evaluate_cases(cases, router)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
