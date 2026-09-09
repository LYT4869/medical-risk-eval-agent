from __future__ import annotations

from typing import Protocol

from .rule_evidence import RuleEvidenceExtractor
from .rule_resolver import RuleIntentResolver
from .routing_types import RequestScope, RoutingDecision, RoutingSource
from .safety_policy import SafetyPolicy
from .task_registry import TaskRegistry, load_default_registry

class AgentRouter(Protocol):
    async def route(self, message: str) -> RoutingDecision: ...


SafetyGate = SafetyPolicy


class RuleRouter:
    def __init__(self, registry: TaskRegistry | None = None):
        resolved_registry = registry or load_default_registry()
        self._extractor = RuleEvidenceExtractor(resolved_registry)
        self._resolver = RuleIntentResolver()

    def route(self, message: str) -> RoutingDecision | None:
        return self._resolver.resolve(self._extractor.extract(message))


class RuleOnlyRouter:
    def __init__(self, rules: RuleRouter | None = None):
        self._rules = rules or RuleRouter()

    async def route(self, message: str) -> RoutingDecision:
        decision = self._rules.route(message)
        if decision is not None:
            return decision
        return RoutingDecision(
            RequestScope.UNKNOWN, RoutingSource.UNKNOWN,
            reason="rule_miss")


class HybridRouter:
    def __init__(self, rules: RuleRouter, semantic):
        self._rules = rules
        self._semantic = semantic

    async def route(self, message: str) -> RoutingDecision:
        decision = self._rules.route(message)
        if decision is not None:
            return decision
        if self._semantic is None:
            return RoutingDecision(
                RequestScope.UNKNOWN, RoutingSource.UNKNOWN,
                reason="semantic_disabled")
        return await self._semantic.route(message)
