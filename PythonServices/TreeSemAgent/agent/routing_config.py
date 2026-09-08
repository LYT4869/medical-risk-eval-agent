from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

from .embedding_provider import SentenceTransformerEmbeddingProvider
from .routing import HybridRouter, RuleRouter
from .routing_executor import SemanticRoutingExecutor
from .semantic_routing import RoutingThresholds, SemanticRouter, SemanticScorer
from .task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry


DEFAULT_ROUTING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_ROUTING_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class RoutingMode(str, Enum):
    RULE = "rule"
    HYBRID_OPTIONAL = "hybrid_optional"
    HYBRID_REQUIRED = "hybrid_required"


@dataclass(frozen=True)
class RoutingSettings:
    mode: RoutingMode
    task_registry: Path
    thresholds: Path
    model: str
    revision: str
    workers: int
    queue_capacity: int
    admission_timeout_seconds: float
    route_timeout_seconds: float

    @classmethod
    def from_environment(
            cls, environment: Mapping[str, str] | None = None,
            *, root: Path | None = None) -> "RoutingSettings":
        values = os.environ if environment is None else environment
        root = root or Path(__file__).resolve().parents[1]
        try:
            mode = RoutingMode(values.get(
                "TREESEM_AGENT_ROUTING_MODE", RoutingMode.RULE.value))
        except ValueError as exc:
            raise RuntimeError(
                "TREESEM_AGENT_ROUTING_MODE must be rule, hybrid_optional or hybrid_required") from exc

        workers = _integer(values, "TREESEM_AGENT_ROUTING_WORKERS", 1,
                           allow_zero=False)
        queue_capacity = _integer(
            values, "TREESEM_AGENT_ROUTING_QUEUE_CAPACITY", 8,
            allow_zero=True)
        admission_ms = _integer(
            values, "TREESEM_AGENT_ROUTING_ADMISSION_TIMEOUT_MS", 5,
            allow_zero=False)
        route_ms = _integer(
            values, "TREESEM_AGENT_ROUTING_TIMEOUT_MS", 150,
            allow_zero=False)
        return cls(
            mode=mode,
            task_registry=Path(values.get(
                "TREESEM_AGENT_TASK_REGISTRY",
                str(root / "config" / "tasks.yaml"))),
            thresholds=Path(values.get(
                "TREESEM_AGENT_ROUTING_THRESHOLDS",
                str(root / "config" / "routing_thresholds.json"))),
            model=values.get(
                "TREESEM_AGENT_ROUTING_MODEL", DEFAULT_ROUTING_MODEL).strip(),
            revision=values.get(
                "TREESEM_AGENT_ROUTING_MODEL_REVISION",
                DEFAULT_ROUTING_REVISION).strip(),
            workers=workers,
            queue_capacity=queue_capacity,
            admission_timeout_seconds=admission_ms / 1000.0,
            route_timeout_seconds=route_ms / 1000.0,
        )


def _integer(values: Mapping[str, str], name: str, default: int,
             *, allow_zero: bool) -> int:
    try:
        value = int(values.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 0 or (value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise RuntimeError(f"{name} must be {requirement}")
    return value


@dataclass
class RoutingRuntime:
    mode: RoutingMode
    router: HybridRouter
    registry: TaskRegistry
    semantic_router: SemanticRouter | None = None
    degradation_reason: str | None = None

    @property
    def semantic_available(self) -> bool:
        return self.semantic_router is not None

    async def close(self) -> None:
        if self.semantic_router is not None:
            await self.semantic_router.close()
            self.semantic_router = None


def build_routing_runtime(
        settings: RoutingSettings,
        *, provider_factory: Callable[[str, str], object] =
        SentenceTransformerEmbeddingProvider) -> RoutingRuntime:
    registry = TaskRegistry.load(
        settings.task_registry, SUPPORTED_DOMAIN_TOOLS)
    rules = RuleRouter(registry)
    if settings.mode == RoutingMode.RULE:
        return RoutingRuntime(
            settings.mode, HybridRouter(rules, None), registry)

    try:
        payload = json.loads(settings.thresholds.read_text(encoding="utf-8"))
        if (set(payload) != {
                "schema_version", "corpus_sha256", "model", "revision",
                "thresholds", "calibration", "fallback_target",
                "fallback_target_met", "rule_known_accuracy"} or
                payload["schema_version"] != 1 or
                payload["model"] != settings.model or
                payload["revision"] != settings.revision):
            raise RuntimeError(
                "routing thresholds do not match configured model")
        thresholds = RoutingThresholds(**payload["thresholds"])
        if not settings.model or not settings.revision:
            raise RuntimeError("routing model and revision are required")
        provider = provider_factory(settings.model, settings.revision)
        scorer = SemanticScorer(registry, provider, thresholds)
        executor = SemanticRoutingExecutor(
            workers=settings.workers,
            queue_capacity=settings.queue_capacity,
            admission_timeout_seconds=settings.admission_timeout_seconds,
            route_timeout_seconds=settings.route_timeout_seconds)
        semantic = SemanticRouter(
            scorer, executor,
            optional=settings.mode == RoutingMode.HYBRID_OPTIONAL)
        return RoutingRuntime(
            settings.mode, HybridRouter(rules, semantic), registry, semantic)
    except Exception:
        if settings.mode == RoutingMode.HYBRID_REQUIRED:
            raise
        return RoutingRuntime(
            settings.mode, HybridRouter(rules, None), registry,
            degradation_reason="semantic_initialization_failed")
