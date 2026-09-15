from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .routing_types import RequestScope
from .run_guard import AgentRunGuard
from .task_registry import BUSINESS_SCOPES, TaskRegistry, load_default_registry


class WorkflowMode(str, Enum):
    DETERMINISTIC = "deterministic"
    OPEN_AGENT = "open_agent"


@dataclass(frozen=True)
class WorkflowPlan:
    mode: WorkflowMode
    stages: tuple[str, ...] = ()
    expected_skill_id: str | None = None

    @classmethod
    def deterministic(
            cls, *stages: str,
            expected_skill_id: str | None = None) -> WorkflowPlan:
        return cls(WorkflowMode.DETERMINISTIC, tuple(stages),
                   expected_skill_id)

    @classmethod
    def open_agent(cls) -> WorkflowPlan:
        return cls(WorkflowMode.OPEN_AGENT)


_SKILL_COMPARISON = (
    "比较", "差异", "变化", "历史", "compare", "comparison", "history",
)
_SKILL_EXPLANATION = (
    "解释", "特征", "决策路径", "explain", "explanation", "feature",
    "decision path",
)
_SKILL_EDUCATION = (
    "循证", "教育", "指南", "知识", "产后出血", "evidence", "education",
    "guideline", "knowledge", "pph", "postpartum haemorrhage",
    "postpartum hemorrhage",
)
_EXPLICIT_EDUCATION = (
    "循证教育", "教育技能", "evidence-education", "evidence education",
    "education skill",
)


class LegacyWorkflowPlanner:
    @classmethod
    def for_request(cls, message: str,
                    guard: AgentRunGuard,
                    registry: TaskRegistry | None = None) -> WorkflowPlan:
        scope = guard.scope
        if scope == RequestScope.SKILL:
            return cls._skill_plan(message)
        if scope in BUSINESS_SCOPES:
            definition = (registry or load_default_registry()).definition(scope)
            stages = list(definition.workflow_stages)
            if (scope == RequestScope.EXPLANATION and
                    "get_prediction" in guard.allowed_tools()):
                stages.insert(0, "get_prediction")
            return WorkflowPlan.deterministic(*stages)
        return WorkflowPlan.open_agent()

    @classmethod
    def _skill_plan(cls, message: str) -> WorkflowPlan:
        normalized = " ".join(message.lower().split())
        if cls._contains(normalized, _EXPLICIT_EDUCATION):
            return WorkflowPlan.deterministic(
                "activate_skill", "search_medical_knowledge",
                expected_skill_id="pph_evidence_education")
        if cls._contains(normalized, _SKILL_COMPARISON):
            return WorkflowPlan.deterministic(
                "activate_skill", "get_prediction_history",
                "compare_predictions",
                expected_skill_id="compare_prediction_history")
        if cls._contains(normalized, _SKILL_EXPLANATION):
            return WorkflowPlan.deterministic(
                "activate_skill", "get_prediction", "get_explanation",
                "search_medical_knowledge",
                expected_skill_id="explain_prediction")
        if cls._contains(normalized, _SKILL_EDUCATION):
            return WorkflowPlan.deterministic(
                "activate_skill", "search_medical_knowledge",
                expected_skill_id="pph_evidence_education")
        return WorkflowPlan.open_agent()

    @staticmethod
    def _contains(message: str, markers: tuple[str, ...]) -> bool:
        return any(marker in message for marker in markers)


# Explicit compatibility alias for the legacy Rule/E5 rollback path.
WorkflowPlanner = LegacyWorkflowPlanner
