from __future__ import annotations

import re
from dataclasses import dataclass

from .routing_types import RequestScope


@dataclass(frozen=True)
class GuardRejection:
    code: str
    tool_error: str


_SKILL = (
    "技能", "流程", "workflow", "skill", "stable process",
    "trusted workflow",
)
_PREDICTION = (
    "演示样本", "预测演示样本", "demo sample", "demonstration sample",
    "predict sample", "运行第",
)
_COMPARISON = ("比较", "compare", "difference", "差异", "变化")
_HISTORY = ("历史", "history", "recent prediction", "最近预测")
_EXPLANATION = (
    "解释", "explain", "important feature", "重要特征", "决策路径",
    "decision path",
)
_SUMMARY = (
    "标签", "概率", "置信度", "模型版本", "label", "probability",
    "confidence", "model version",
)
_STORED_READ = (
    "读取", "查看", "当前预测", "当前结果", "刚才", "记录",
    "read", "retrieve", "stored", "current prediction", "current result",
)
_STORED_PREDICTION = _SUMMARY + (
    "当前预测", "当前结果", "刚才的结果", "上一次",
    "stored prediction", "current prediction", "current result",
)
_KNOWLEDGE = (
    "资料", "指南", "知识", "引用", "什么是", "介绍", "evidence",
    "guideline", "documentation", "documented", "cite", "material",
    "source", "what is", "overview",
)

_ABUSE = re.compile(
    r"(?:伪造|编造|绕过|忽略.*规则|ignore.*instruction|fabricate|invent|bypass)",
    re.IGNORECASE,
)
_PROTECTED = re.compile(
    r"(?:预测|概率|引用|权限|其他患者|另一名患者|prediction|probability|citation|authorization|other patient)",
    re.IGNORECASE,
)
_DEFENSIVE_ABUSE = re.compile(
    r"(?:不要|不得|不能|避免|拒绝|do not|don't|must not|never)\s*"
    r"(?:伪造|编造|fabricate|invent)",
    re.IGNORECASE,
)

_PERSONALIZED_REQUEST = (
    "为我", "给我", "个体化", "具体药物", "确定诊断",
    "prescribe", "individualized", "personalized",
)
_CLINICAL_ACTION = (
    "处方", "药物", "剂量", "治疗方案", "诊断",
    "prescription", "medication", "dosage", "dose", "regimen",
    "diagnosis", "diagnose",
)
_UNAVAILABLE_EVIDENCE = (
    "不存在", "没有也要", "未收录", "缺失", "absent", "unindexed",
    "does not contain", "not contain", "unavailable",
)
_SOURCE_REQUEST = (
    "资料", "索引", "引用", "方案", "证据", "quote", "source",
    "protocol", "evidence",
)

_READ_ONLY_NATIVE_TOOLS = {
    "get_prediction",
    "get_explanation",
    "get_prediction_history",
    "compare_predictions",
}
_REGISTERED_DOMAIN_TOOLS = _READ_ONLY_NATIVE_TOOLS | {
    "predict_sample", "search_medical_knowledge",
}

_INITIAL_TOOLS = {
    RequestScope.SKILL: {"activate_skill"},
    RequestScope.PREDICTION: {"predict_sample"},
    RequestScope.SUMMARY: {"get_prediction"},
    RequestScope.EXPLANATION: {"get_explanation"},
    RequestScope.HISTORY: {"get_prediction_history"},
    RequestScope.COMPARISON: {
        "get_prediction_history", "compare_predictions"},
    RequestScope.KNOWLEDGE: {"search_medical_knowledge"},
    RequestScope.SECURITY_ABUSE: set(),
    RequestScope.MEDICAL_REFUSAL: set(),
    RequestScope.UNKNOWN: _READ_ONLY_NATIVE_TOOLS,
}


class AgentRunGuard:
    def __init__(self, scope: RequestScope, *, include_summary: bool = False):
        self.scope = scope
        self.security_refusal = (
            "explicit_security_abuse"
            if scope == RequestScope.SECURITY_ABUSE else None)
        self.medical_refusal = (
            "unsafe_individual_medical_request"
            if scope == RequestScope.MEDICAL_REFUSAL else None)
        self._initial_tools = set(_INITIAL_TOOLS[scope])
        if scope == RequestScope.EXPLANATION and include_summary:
            self._initial_tools.add("get_prediction")
        self._active_skill_tools: set[str] | None = None
        self._attempts: dict[str, int] = {}
        self._successful: set[str] = set()
        self._knowledge_satisfied = False

    @classmethod
    def for_request(cls, message: str) -> AgentRunGuard:
        normalized = " ".join(message.lower().split())
        abuse_candidate = _DEFENSIVE_ABUSE.sub("", normalized)
        if (_ABUSE.search(abuse_candidate) and
                _PROTECTED.search(abuse_candidate)):
            return cls(RequestScope.SECURITY_ABUSE)
        personalized_medical = (
            cls._contains(normalized, _PERSONALIZED_REQUEST) and
            cls._contains(normalized, _CLINICAL_ACTION))
        fabricated_source = (
            cls._contains(normalized, _UNAVAILABLE_EVIDENCE) and
            cls._contains(normalized, _SOURCE_REQUEST))
        if personalized_medical or fabricated_source:
            return cls(RequestScope.MEDICAL_REFUSAL)
        if cls._contains(normalized, _SKILL):
            return cls(RequestScope.SKILL)
        if cls._contains(normalized, _PREDICTION):
            return cls(RequestScope.PREDICTION)
        if cls._contains(normalized, _COMPARISON):
            return cls(RequestScope.COMPARISON)
        if cls._contains(normalized, _HISTORY):
            return cls(RequestScope.HISTORY)
        has_explanation = cls._contains(normalized, _EXPLANATION)
        has_knowledge = cls._contains(normalized, _KNOWLEDGE)
        if (has_explanation and
                (not has_knowledge or
                 cls._contains(normalized, _STORED_PREDICTION))):
            return cls(
                RequestScope.EXPLANATION,
                include_summary=cls._contains(normalized, _SUMMARY),
            )
        if (cls._contains(normalized, _SUMMARY) and
                cls._contains(normalized, _STORED_READ)):
            return cls(RequestScope.SUMMARY)
        if has_knowledge:
            return cls(RequestScope.KNOWLEDGE)
        return cls(RequestScope.UNKNOWN)

    @staticmethod
    def _contains(message: str, markers: tuple[str, ...]) -> bool:
        return any(marker in message for marker in markers)

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
