from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Sequence

from agent.onnx_embedding_provider import OnnxEmbeddingProvider
from agent.routing_artifact import load_routing_artifact
from agent.semantic_routing import RoutingThresholds, SemanticScorer
from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry


BENCHMARK_QUERIES = (
    "把我之前做过的都列一下",
    "请说明当前预测的主要影响因素",
    "比较最近两次模型结果有什么不同",
    "产后出血风险模型的概率代表什么",
    "请概括当前这次模型输出",
    "用演示样本执行一次风险预测",
)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile values cannot be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("percentile quantile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def validate_counts(warmup: int, iterations: int) -> None:
    if isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup count must be non-negative")
    if isinstance(iterations, bool) or iterations <= 0:
        raise ValueError("iteration count must be positive")


def recursive_file_size(directory: Path) -> int:
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("artifact directory is unavailable or unsafe")
    total = 0
    for path in directory.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            total += path.stat().st_size
    return total


def maximum_rss_kib(raw_maximum_rss: int | float,
                    platform_name: str | None = None) -> int:
    value = float(raw_maximum_rss)
    if not math.isfinite(value) or value < 0:
        raise ValueError("maximum RSS must be finite and non-negative")
    current = (platform_name or sys.platform).lower()
    if current.startswith("darwin"):
        value /= 1024.0
    return int(math.ceil(value))


def deployment_footprint(
        *, container_image_bytes: int, artifact_bytes: int,
        baseline_deployment_bytes: int | None) -> dict[str, int | float | None]:
    if container_image_bytes <= 0:
        raise ValueError("container image size must be positive")
    if artifact_bytes <= 0:
        raise ValueError("artifact size must be positive")
    total = container_image_bytes + artifact_bytes
    if baseline_deployment_bytes is None:
        relative = None
    else:
        if baseline_deployment_bytes <= 0:
            raise ValueError("baseline deployment size must be positive")
        relative = ((total - baseline_deployment_bytes) /
                    baseline_deployment_bytes)
    return {
        "container_image_bytes": container_image_bytes,
        "artifact_size_bytes": artifact_bytes,
        "total_deployment_bytes": total,
        "baseline_deployment_bytes": baseline_deployment_bytes,
        "relative_change": relative,
    }


def _thresholds(path: Path, model: str,
                revision: str) -> RoutingThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema_version") != 1 or
            payload.get("model") != model or
            payload.get("revision") != revision):
        raise ValueError("routing thresholds do not match model")
    return RoutingThresholds(**payload["thresholds"])


def _elapsed_ms(operation) -> tuple[object, float]:
    started = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - started) / 1_000_000.0


def benchmark(
        *, artifact_dir: Path, tasks_path: Path, thresholds_path: Path,
        backend: str, model: str, revision: str,
        warmup: int, iterations: int, container_image_bytes: int,
        baseline_deployment_bytes: int | None = None) -> dict[str, object]:
    validate_counts(warmup, iterations)
    if backend not in {"onnx_fp32", "onnx_int8"}:
        raise ValueError("unsupported routing embedding backend")

    def initialize():
        registry = TaskRegistry.load(tasks_path, SUPPORTED_DOMAIN_TOOLS)
        thresholds = _thresholds(thresholds_path, model, revision)
        artifact = load_routing_artifact(
            artifact_dir,
            task_registry_path=tasks_path,
            thresholds_path=thresholds_path,
            registry=registry,
            expected_backend=backend,
            expected_model_id=model,
            expected_revision=revision,
        )
        provider = OnnxEmbeddingProvider(artifact)
        return artifact, SemanticScorer(registry, provider, thresholds)

    initialized, cold_initialization_ms = _elapsed_ms(initialize)
    artifact, scorer = initialized
    _, first_route_ms = _elapsed_ms(
        lambda: scorer.route(BENCHMARK_QUERIES[0]))
    for index in range(warmup):
        scorer.route(BENCHMARK_QUERIES[index % len(BENCHMARK_QUERIES)])

    durations = []
    for index in range(iterations):
        _, duration = _elapsed_ms(
            lambda index=index: scorer.route(
                BENCHMARK_QUERIES[index % len(BENCHMARK_QUERIES)]))
        durations.append(duration)

    try:
        import onnxruntime
    except ModuleNotFoundError as exc:
        raise RuntimeError("onnxruntime is unavailable") from exc
    footprint = deployment_footprint(
        container_image_bytes=container_image_bytes,
        artifact_bytes=recursive_file_size(artifact_dir),
        baseline_deployment_bytes=baseline_deployment_bytes)
    return {
        "schema_version": 1,
        "artifact_version": artifact.manifest.artifact_version,
        "embedding_backend": backend,
        **footprint,
        "cold_initialization_ms": cold_initialization_ms,
        "first_route_ms": first_route_ms,
        "warmed_p50_route_ms": percentile(durations, 0.50),
        "warmed_p95_route_ms": percentile(durations, 0.95),
        "warmed_p99_route_ms": percentile(durations, 0.99),
        "maximum_rss_kib": maximum_rss_kib(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "warmup_count": warmup,
        "iteration_count": iterations,
        "benchmark_query_count": len(BENCHMARK_QUERIES),
        "runtime": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "onnxruntime": onnxruntime.__version__,
            "cpu_count": os.cpu_count(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark treeSem Agent ONNX routing runtime")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--backend", choices=("onnx_fp32", "onnx_int8"),
                        required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument(
        "--revision",
        default="614241f622f53c4eeff9890bdc4f31cfecc418b3")
    parser.add_argument("--warmup", required=True, type=int)
    parser.add_argument("--iterations", required=True, type=int)
    parser.add_argument("--container-image-bytes", required=True, type=int)
    parser.add_argument("--baseline-deployment-bytes", type=int)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    report = benchmark(
        artifact_dir=arguments.artifact_dir,
        tasks_path=arguments.tasks,
        thresholds_path=arguments.thresholds,
        backend=arguments.backend,
        model=arguments.model,
        revision=arguments.revision,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
        container_image_bytes=arguments.container_image_bytes,
        baseline_deployment_bytes=arguments.baseline_deployment_bytes)
    rendered = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
