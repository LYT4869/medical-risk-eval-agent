from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .run_guard import AgentRunGuard, RequestScope


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


class WorkflowPlanner:
    @classmethod
    def for_request(cls, message: str,
                    guard: AgentRunGuard) -> WorkflowPlan:
        scope = guard.scope
        if scope == RequestScope.PREDICTION:
            return WorkflowPlan.deterministic("predict_sample")
        if scope == RequestScope.SUMMARY:
            return WorkflowPlan.deterministic("get_prediction")
        if scope == RequestScope.EXPLANATION:
            stages = []
            if "get_prediction" in guard.allowed_tools():
                stages.append("get_prediction")
            stages.append("get_explanation")
            return WorkflowPlan.deterministic(*stages)
        if scope == RequestScope.HISTORY:
            return WorkflowPlan.deterministic("get_prediction_history")
        if scope == RequestScope.COMPARISON:
            return WorkflowPlan.deterministic(
                "get_prediction_history", "compare_predictions")
        if scope == RequestScope.KNOWLEDGE:
            return WorkflowPlan.deterministic("search_medical_knowledge")
        if scope == RequestScope.SKILL:
            return cls._skill_plan(message)
        return WorkflowPlan.open_agent()

    @classmethod
    def _skill_plan(cls, message: str) -> WorkflowPlan:
        normalized = " ".join(message.lower().split())
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
