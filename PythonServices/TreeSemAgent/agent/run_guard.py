from __future__ import annotations

from dataclasses import dataclass

from .routing_types import RequestScope
from .task_registry import BUSINESS_SCOPES, TaskRegistry, load_default_registry


@dataclass(frozen=True)
class GuardRejection:
    code: str
    tool_error: str


_READ_ONLY_NATIVE_TOOLS = {
    "get_prediction",
    "get_explanation",
    "get_prediction_history",
    "compare_predictions",
}
_REGISTERED_DOMAIN_TOOLS = _READ_ONLY_NATIVE_TOOLS | {
    "predict_sample", "search_medical_knowledge",
}

_SPECIAL_INITIAL_TOOLS = {
    RequestScope.SKILL: {"activate_skill"},
    RequestScope.SECURITY_ABUSE: set(),
    RequestScope.MEDICAL_REFUSAL: set(),
    RequestScope.UNKNOWN: _READ_ONLY_NATIVE_TOOLS,
}


class AgentRunGuard:
    def __init__(self, scope: RequestScope, *,
                 registry: TaskRegistry | None = None,
                 include_summary: bool = False,
                 initial_tools: set[str] | None = None):
        self.scope = scope
        self.security_refusal = (
            "explicit_security_abuse"
            if scope == RequestScope.SECURITY_ABUSE else None)
        self.medical_refusal = (
            "unsafe_individual_medical_request"
            if scope == RequestScope.MEDICAL_REFUSAL else None)
        self._initial_tools = (set(initial_tools) if initial_tools is not None else
            set((registry or load_default_registry()).definition(
                scope).allowed_tools)
            if scope in BUSINESS_SCOPES
            else set(_SPECIAL_INITIAL_TOOLS[scope]))
        if scope == RequestScope.EXPLANATION and include_summary:
            self._initial_tools.add("get_prediction")
        self._active_skill_tools: set[str] | None = None
        self._attempts: dict[str, int] = {}
        self._successful: set[str] = set()
        self._knowledge_satisfied = False

    @classmethod
    def for_request(cls, message: str,
                    registry: TaskRegistry | None = None) -> AgentRunGuard:
        from .routing import RuleRouter, SafetyGate

        safety = SafetyGate().evaluate(message)
        if not safety.allowed:
            return cls(safety.refusal_scope or RequestScope.UNKNOWN)
        decision = RuleRouter(registry).route(message)
        if decision is None:
            return cls(RequestScope.UNKNOWN, registry=registry)
        return cls(decision.scope, registry=registry,
                   include_summary=decision.include_summary)

    @classmethod
    def for_scope(cls, scope: RequestScope, *,
                  registry: TaskRegistry | None = None,
                  include_summary: bool = False) -> AgentRunGuard:
        return cls(scope, registry=registry, include_summary=include_summary)

    @classmethod
    def for_allowed_tools(cls, allowed_tools: set[str]) -> AgentRunGuard:
        permitted = _REGISTERED_DOMAIN_TOOLS | {"activate_skill"}
        if not allowed_tools <= permitted:
            raise ValueError("structured routing contains an unknown Tool")
        return cls(
            RequestScope.UNKNOWN, initial_tools=set(allowed_tools))

    def allowed_tools(self) -> set[str]:
        allowed = set(
            self._active_skill_tools
            if self._active_skill_tools is not None
            else self._initial_tools)
        allowed.difference_update(self._successful)
        if self._attempts.get("predict_sample", 0) > 0:
            allowed.discard("predict_sample")
        if (self._knowledge_satisfied or
                self._attempts.get("search_medical_knowledge", 0) >= 2):
            allowed.discard("search_medical_knowledge")
        return allowed

    def before_tool(self, name: str) -> GuardRejection | None:
        if (name == "search_medical_knowledge" and
                self._attempts.get(name, 0) >= 2):
            return GuardRejection(
                "knowledge_attempt_limit", "knowledge attempt limit reached")
        if name not in self.allowed_tools():
            return GuardRejection("tool_not_allowed", "tool is not allowed")
        return None

    def record_tool(self, name: str, status: str,
                    citation_count: int = 0) -> None:
        self._attempts[name] = self._attempts.get(name, 0) + 1
        if name == "search_medical_knowledge":
            self._knowledge_satisfied = (
                self._knowledge_satisfied or
                (status == "success" and citation_count > 0))
        if status == "success" and name != "search_medical_knowledge":
            self._successful.add(name)

    def record_skill_activation(self, required_tools: set[str]) -> None:
        self._active_skill_tools = (
            set(required_tools) & _REGISTERED_DOMAIN_TOOLS)
