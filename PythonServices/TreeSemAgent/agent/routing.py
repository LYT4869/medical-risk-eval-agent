from __future__ import annotations

import re
from dataclasses import dataclass

from .routing_types import RequestScope, RoutingDecision, RoutingSource
from .task_registry import TaskRegistry, load_default_registry


_SKILL = (
    "技能", "流程", "workflow", "skill", "stable process",
    "trusted workflow",
)
_STORED_READ = (
    "读取", "查看", "当前预测", "当前结果", "刚才", "记录",
    "read", "retrieve", "stored", "current prediction", "current result",
)
_STORED_PREDICTION = (
    "标签", "概率", "置信度", "模型版本", "label", "probability",
    "confidence", "model version", "当前预测", "当前结果", "刚才的结果",
    "上一次", "stored prediction", "current prediction", "current result",
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


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    refusal_scope: RequestScope | None = None
    reason: str | None = None


class SafetyGate:
    def evaluate(self, message: str) -> SafetyDecision:
        normalized = normalize(message)
        abuse_candidate = _DEFENSIVE_ABUSE.sub("", normalized)
        if (_ABUSE.search(abuse_candidate) and
                _PROTECTED.search(abuse_candidate)):
            return SafetyDecision(
                False, RequestScope.SECURITY_ABUSE,
                "explicit_security_abuse")
        personalized_medical = (
            contains(normalized, _PERSONALIZED_REQUEST) and
            contains(normalized, _CLINICAL_ACTION))
        fabricated_source = (
            contains(normalized, _UNAVAILABLE_EVIDENCE) and
            contains(normalized, _SOURCE_REQUEST))
        if personalized_medical or fabricated_source:
            return SafetyDecision(
                False, RequestScope.MEDICAL_REFUSAL,
                "unsafe_individual_medical_request")
        return SafetyDecision(True)


class RuleRouter:
    def __init__(self, registry: TaskRegistry | None = None):
        self._registry = registry or load_default_registry()

    def route(self, message: str) -> RoutingDecision | None:
        normalized = normalize(message)
        if contains(normalized, _SKILL):
            return self._decision(RequestScope.SKILL)
        if self._matches(RequestScope.PREDICTION, normalized):
            return self._decision(RequestScope.PREDICTION)
        if self._matches(RequestScope.COMPARISON, normalized):
            return self._decision(RequestScope.COMPARISON)
        if self._matches(RequestScope.HISTORY, normalized):
            return self._decision(RequestScope.HISTORY)
        has_explanation = self._matches(RequestScope.EXPLANATION, normalized)
        has_knowledge = self._matches(RequestScope.KNOWLEDGE, normalized)
        if (has_explanation and
                (not has_knowledge or contains(normalized, _STORED_PREDICTION))):
            include_summary = (
                self._matches(RequestScope.SUMMARY, normalized) and
                contains(normalized, _STORED_READ))
            return self._decision(
                RequestScope.EXPLANATION, include_summary=include_summary)
        if (self._matches(RequestScope.SUMMARY, normalized) and
                contains(normalized, _STORED_READ)):
            return self._decision(RequestScope.SUMMARY)
        if has_knowledge:
            return self._decision(RequestScope.KNOWLEDGE)
        return None

    def _matches(self, scope: RequestScope, message: str) -> bool:
        return contains(message, self._registry.definition(scope).rule_terms)

    @staticmethod
    def _decision(scope: RequestScope, *,
                  include_summary: bool = False) -> RoutingDecision:
        return RoutingDecision(
            scope=scope, source=RoutingSource.RULE,
            include_summary=include_summary)


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


def normalize(message: str) -> str:
    return " ".join(message.casefold().split())


def contains(message: str, markers: tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)

