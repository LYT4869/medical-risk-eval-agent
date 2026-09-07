from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from .routing_types import RequestScope, RoutingDecision, RoutingSource
from .task_registry import TaskRegistry


class EmbeddingProvider(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@dataclass(frozen=True)
class RoutingThresholds:
    min_similarity: float
    min_margin: float
    secondary_intent_similarity: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        if not 0.0 <= self.min_margin <= 2.0:
            raise ValueError("min_margin must be between 0 and 2")
        if not -1.0 <= self.secondary_intent_similarity <= 1.0:
            raise ValueError(
                "secondary_intent_similarity must be between -1 and 1")


class SemanticScorer:
    def __init__(self, registry: TaskRegistry, provider: EmbeddingProvider,
                 thresholds: RoutingThresholds):
        self._provider = provider
        self._thresholds = thresholds
        scopes: list[RequestScope] = []
        examples: list[str] = []
        for definition in registry.business_definitions:
            for example in definition.intent_examples:
                scopes.append(definition.scope)
                examples.append(example)
        vectors = provider.encode(examples)
        if len(vectors) != len(examples):
            raise ValueError("embedding count mismatch")
        normalized = tuple(self._normalize(vector) for vector in vectors)
        if not normalized:
            raise ValueError("semantic examples cannot be empty")
        dimension = len(normalized[0])
        if any(len(vector) != dimension for vector in normalized):
            raise ValueError("embedding dimension mismatch")
        self._dimension = dimension
        self._examples = tuple(zip(scopes, normalized))

    def route(self, message: str) -> RoutingDecision:
        vectors = self._provider.encode([message])
        if len(vectors) != 1:
            raise ValueError("query embedding count mismatch")
        query = self._normalize(vectors[0])
        if len(query) != self._dimension:
            raise ValueError("query embedding dimension mismatch")
        scores: dict[RequestScope, float] = {}
        for scope, example in self._examples:
            similarity = sum(left * right for left, right in zip(query, example))
            scores[scope] = max(scores.get(scope, -1.0), similarity)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        (top_scope, top_score), (_, second_score) = ranked[:2]
        margin = top_score - second_score
        common = {
            "similarity_score": top_score,
            "margin": margin,
            "secondary_score": second_score,
        }
        if top_score < self._thresholds.min_similarity:
            return self._unknown("semantic_low_similarity", **common)
        if margin < self._thresholds.min_margin:
            return self._unknown("semantic_ambiguous_margin", **common)
        if second_score >= self._thresholds.secondary_intent_similarity:
            return self._unknown("semantic_multiple_intents", **common)
        return RoutingDecision(
            top_scope, RoutingSource.SEMANTIC, **common)

    @staticmethod
    def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
        if (not vector or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in vector)):
            raise ValueError("embedding values must be finite numbers")
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("embedding norm must be finite and positive")
        return tuple(float(value) / norm for value in vector)

    @staticmethod
    def _unknown(reason: str, **scores: float) -> RoutingDecision:
        return RoutingDecision(
            RequestScope.UNKNOWN, RoutingSource.UNKNOWN,
            reason=reason, **scores)
