from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .routing_types import RequestScope


BUSINESS_SCOPES = frozenset({
    RequestScope.PREDICTION,
    RequestScope.SUMMARY,
    RequestScope.EXPLANATION,
    RequestScope.HISTORY,
    RequestScope.COMPARISON,
    RequestScope.KNOWLEDGE,
})
SUPPORTED_DOMAIN_TOOLS = {
    "predict_sample", "get_prediction", "get_explanation",
    "get_prediction_history", "compare_predictions",
    "search_medical_knowledge",
}
DEFAULT_TASKS_PATH = Path(__file__).resolve().parents[1] / "config" / "tasks.yaml"


class TaskRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class TaskDefinition:
    scope: RequestScope
    mode: str
    rule_terms: tuple[str, ...]
    intent_examples: tuple[str, ...]
    workflow_stages: tuple[str, ...]
    allowed_tools: frozenset[str]


class TaskRegistry:
    _ROOT_FIELDS = frozenset({"schema_version", "tasks"})
    _TASK_FIELDS = frozenset({
        "scope", "mode", "rule_terms", "intent_examples",
        "workflow_stages", "allowed_tools",
    })

    def __init__(self, definitions: dict[RequestScope, TaskDefinition]):
        self._definitions = dict(definitions)

    @classmethod
    def load(cls, path: Path, available_tools: set[str]) -> "TaskRegistry":
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise TaskRegistryError("cannot load task registry") from exc
        if (not isinstance(payload, dict) or
                set(payload) != cls._ROOT_FIELDS or
                payload.get("schema_version") != 1 or
                not isinstance(payload.get("tasks"), list)):
            raise TaskRegistryError("invalid task registry schema")

        definitions: dict[RequestScope, TaskDefinition] = {}
        for raw in payload["tasks"]:
            if not isinstance(raw, dict):
                raise TaskRegistryError("task definition must be an object")
            unknown = set(raw) - cls._TASK_FIELDS
            missing = cls._TASK_FIELDS - set(raw)
            if unknown:
                raise TaskRegistryError("unknown task fields")
            if missing:
                raise TaskRegistryError("missing task fields")
            try:
                scope = RequestScope(raw["scope"])
            except (TypeError, ValueError) as exc:
                raise TaskRegistryError("invalid business scope") from exc
            if scope not in BUSINESS_SCOPES:
                raise TaskRegistryError("task must use a business scope")
            if scope in definitions:
                raise TaskRegistryError("duplicate scope")
            if raw["mode"] != "deterministic":
                raise TaskRegistryError("unsupported mode")
            rule_terms = cls._string_tuple(raw["rule_terms"], "rule_terms")
            intent_examples = cls._string_tuple(
                raw["intent_examples"], "intent_examples")
            allowed_tools = frozenset(cls._string_tuple(
                raw["allowed_tools"], "allowed_tools"))
            stages = cls._string_tuple(
                raw["workflow_stages"], "workflow_stages")
            if not intent_examples:
                raise TaskRegistryError("intent_examples cannot be empty")
            if not allowed_tools <= available_tools:
                raise TaskRegistryError("task contains unknown tool")
            if not set(stages) <= allowed_tools:
                raise TaskRegistryError("workflow stage outside allowed_tools")
            definitions[scope] = TaskDefinition(
                scope, raw["mode"], rule_terms, intent_examples, stages,
                allowed_tools)
        if set(definitions) != BUSINESS_SCOPES:
            raise TaskRegistryError("registry must define all business scopes")
        return cls(definitions)

    @staticmethod
    def _string_tuple(value: object, field: str) -> tuple[str, ...]:
        if (not isinstance(value, list) or
                any(not isinstance(item, str) or not item.strip()
                    for item in value)):
            raise TaskRegistryError(f"{field} must be a string list")
        normalized = tuple(item.strip() for item in value)
        if len(set(normalized)) != len(normalized):
            raise TaskRegistryError(f"{field} contains duplicates")
        return normalized

    def definition(self, scope: RequestScope) -> TaskDefinition:
        try:
            return self._definitions[scope]
        except KeyError as exc:
            raise TaskRegistryError("scope is not registered") from exc

    @property
    def business_definitions(self) -> tuple[TaskDefinition, ...]:
        return tuple(self._definitions[scope] for scope in sorted(
            self._definitions, key=lambda item: item.value))


@lru_cache(maxsize=1)
def load_default_registry() -> TaskRegistry:
    return TaskRegistry.load(DEFAULT_TASKS_PATH, SUPPORTED_DOMAIN_TOOLS)
