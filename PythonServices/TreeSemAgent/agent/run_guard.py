from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .routing_types import RequestScope
from .schemas import CompareArgs, HistoryArgs, PredictionIdArgs
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
_READ_ACTION_SCHEMAS = {
    "get_prediction": PredictionIdArgs,
    "get_explanation": PredictionIdArgs,
    "get_prediction_history": HistoryArgs,
    "compare_predictions": CompareArgs,
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
        self._successful_tools: set[str] = set()
        self._successful_read_actions: set[tuple[str, str]] = set()
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
        allowed.difference_update(self._successful_tools)
        if self._attempts.get("predict_sample", 0) > 0:
            allowed.discard("predict_sample")
        if (self._knowledge_satisfied or
                self._attempts.get("search_medical_knowledge", 0) >= 2):
            allowed.discard("search_medical_knowledge")
        return allowed

    @staticmethod
    def _action_key(
            name: str, arguments: dict[str, Any] | None
            ) -> tuple[str, str] | None:
        try:
            schema = _READ_ACTION_SCHEMAS[name]
            canonical_arguments = schema.model_validate(
                arguments or {}).model_dump(mode="json")
            normalized = json.dumps(
                canonical_arguments, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False)
        except (KeyError, TypeError, ValueError):
            return None
        return name, normalized

    def before_tool(
            self, name: str,
            arguments: dict[str, Any] | None = None) -> GuardRejection | None:
        if (name == "search_medical_knowledge" and
                self._attempts.get(name, 0) >= 2):
            return GuardRejection(
                "knowledge_attempt_limit", "knowledge attempt limit reached")
        if name not in self.allowed_tools():
            return GuardRejection("tool_not_allowed", "tool is not allowed")
        if name in _READ_ONLY_NATIVE_TOOLS:
            action_key = self._action_key(name, arguments)
            if action_key is None:
                return GuardRejection(
                    "invalid_tool_arguments", "tool arguments are invalid")
            if action_key in self._successful_read_actions:
                return GuardRejection(
                    "repeated_tool_call", "successful action already completed")
        return None

    def record_tool(self, name: str, status: str,
                    citation_count: int = 0,
                    arguments: dict[str, Any] | None = None) -> None:
        self._attempts[name] = self._attempts.get(name, 0) + 1
        if name == "search_medical_knowledge":
            self._knowledge_satisfied = (
                self._knowledge_satisfied or
                (status == "success" and citation_count > 0))
        if status != "success" or name == "search_medical_knowledge":
            return
        action_key = self._action_key(name, arguments)
        if name in _READ_ONLY_NATIVE_TOOLS and action_key is not None:
            self._successful_read_actions.add(action_key)
        elif name not in _READ_ONLY_NATIVE_TOOLS:
            self._successful_tools.add(name)

    def record_skill_activation(self, required_tools: set[str]) -> None:
        self._active_skill_tools = (
            set(required_tools) & _REGISTERED_DOMAIN_TOOLS)
