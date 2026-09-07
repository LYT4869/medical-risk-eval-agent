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
_STRONG_KNOWLEDGE = (
    "资料", "指南", "引用", "evidence", "guideline", "documentation",
    "documented", "cite", "material", "source", "模型限制",
    "model limitation", "model limit", "guidance",
)
_KNOWLEDGE_DOMAIN = (
    "产后出血", "pph", "postpartum", "模型", "treesem", "风险",
    "医学", "临床", "特征", "概率", "指标", "medical", "clinical",
)
_KNOWLEDGE_RETRIEVAL_ACTION = (
    "查找", "检索", "搜索", "查询资料", "find", "retrieve", "search",
    "provide general medical evidence",
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
    "prescribe", "individualized", "personalized", "for me",
)
_CLINICAL_ACTION = (
    "处方", "药物", "剂量", "治疗方案", "用药方案", "诊断",
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
        has_prediction = self._matches(RequestScope.PREDICTION, normalized)
        has_comparison = self._matches(RequestScope.COMPARISON, normalized)
        has_history = self._matches(RequestScope.HISTORY, normalized)
        has_explanation = self._matches(RequestScope.EXPLANATION, normalized)
        has_summary = self._matches(RequestScope.SUMMARY, normalized)
        has_stored_read = contains(normalized, _STORED_READ)
        generic_knowledge = self._matches(RequestScope.KNOWLEDGE, normalized)
        strong_knowledge = contains(normalized, _STRONG_KNOWLEDGE)
        has_knowledge = (
            strong_knowledge or
            (generic_knowledge and contains(normalized, _KNOWLEDGE_DOMAIN)))
        explanation_position = first_marker_position(
            normalized,
            self._registry.definition(RequestScope.EXPLANATION).rule_terms)
        retrieval_position = first_marker_position(
            normalized, _KNOWLEDGE_RETRIEVAL_ACTION)
        explanation_then_retrieval = (
            explanation_position is not None and
            retrieval_position is not None and
            explanation_position < retrieval_position)

        if (has_prediction and
                (has_comparison or has_history or has_explanation or
                 has_knowledge)):
            return self._compositional()
        if has_comparison and (has_explanation or has_knowledge):
            return self._compositional()
        if has_history and has_knowledge and not has_comparison:
            return self._compositional()
        if has_explanation and has_knowledge and explanation_then_retrieval:
            return self._compositional()
        if (has_summary and has_knowledge and not has_explanation and
                (has_stored_read or contains(
                    normalized, ("同时", "并且", "以及", " and ")))):
            return self._compositional()

        if has_prediction:
            return self._decision(RequestScope.PREDICTION)
        if has_comparison:
            return self._decision(RequestScope.COMPARISON)
        if has_history:
            return self._decision(RequestScope.HISTORY)
        if (has_explanation and
                (not has_knowledge or contains(normalized, _STORED_PREDICTION))):
            include_summary = (
                has_summary and has_stored_read)
            return self._decision(
                RequestScope.EXPLANATION, include_summary=include_summary)
        if has_summary and has_stored_read:
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

    @staticmethod
    def _compositional() -> RoutingDecision:
        return RoutingDecision(
            RequestScope.UNKNOWN, RoutingSource.RULE,
            reason="rule_compositional")


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


def normalize(message: str) -> str:
    return " ".join(message.casefold().split())


def contains(message: str, markers: tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)


def first_marker_position(message: str,
                          markers: tuple[str, ...]) -> int | None:
    positions = [message.find(marker) for marker in markers if marker in message]
    return min(positions) if positions else None
