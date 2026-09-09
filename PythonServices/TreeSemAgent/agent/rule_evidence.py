from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .routing_types import RequestScope
from .task_registry import TaskRegistry, load_default_registry


class IntentAction(str, Enum):
    PREDICT = "predict"
    READ = "read"
    EXPLAIN = "explain"
    LIST = "list"
    COMPARE = "compare"
    RETRIEVE = "retrieve"


class IntentObject(str, Enum):
    DEMO_SAMPLE = "demo_sample"
    PREDICTION_RECORD = "prediction_record"
    PREDICTION_FACT = "prediction_fact"
    EXPLANATION_DETAIL = "explanation_detail"
    HISTORY = "history"
    KNOWLEDGE = "knowledge"


class IntentReference(str, Enum):
    EXPLICIT_SAMPLE = "explicit_sample"
    CURRENT_PREDICTION = "current_prediction"
    PRIOR_PREDICTION = "prior_prediction"
    MULTIPLE_PREDICTIONS = "multiple_predictions"


@dataclass(frozen=True)
class RuleEvidence:
    actions: frozenset[IntentAction]
    objects: frozenset[IntentObject]
    references: frozenset[IntentReference]
    registry_scopes: frozenset[RequestScope]
    explicit_skill: bool = False
    vague_reference: bool = False


_ACTION_MARKERS = {
    IntentAction.PREDICT: (
        "预测样本", "预测演示", "运行样本", "运行第", "做预测", "新预测",
        "predict sample", "run sample", "run demo", "evaluate sample",
        "evaluate synthetic", "predict a demo case", "new prediction",
        "execute a new prediction",
    ),
    IntentAction.READ: (
        "读取", "查看", "展示", "给出", "告诉我", "read", "show",
        "display", "retrieve stored", "give me",
    ),
    IntentAction.EXPLAIN: (
        "解释", "为什么", "为啥", "怎么判", "explain", "why",
    ),
    IntentAction.LIST: (
        "列出", "历史", "最近记录", "list", "history", "recent records",
        "recent predictions",
    ),
    IntentAction.COMPARE: (
        "比较", "对比", "差异", "变化", "compare", "comparison",
        "comparing", "difference", "changed",
    ),
    IntentAction.RETRIEVE: (
        "查找", "检索", "搜索", "查询", "科普", "介绍", "什么是",
        "引用", "find", "retrieve", "search", "look up", "overview",
        "describe", "cite", "指南来源", "guideline source",
    ),
}

_OBJECT_MARKERS = {
    IntentObject.DEMO_SAMPLE: (
        "演示样本", "样本", "示例", "demo sample", "demonstration sample",
        "synthetic sample", "demo case", "sample",
    ),
    IntentObject.PREDICTION_RECORD: (
        "预测结果", "当前预测", "当前结果", "刚才结果", "刚才的结果",
        "结果", "prediction result", "current prediction", "current result",
        "result",
    ),
    IntentObject.PREDICTION_FACT: (
        "标签", "概率", "置信度", "模型版本", "label", "probability",
        "confidence", "model version", "score",
    ),
    IntentObject.EXPLANATION_DETAIL: (
        "重要特征", "主要特征", "关键因素", "决策路径", "树路径",
        "important feature", "decision path", "tree path",
    ),
    IntentObject.HISTORY: (
        "历史", "历史记录", "最近记录", "旧记录", "保存的记录", "history",
        "recent records", "recent predictions", "saved runs", "previous runs",
    ),
    IntentObject.KNOWLEDGE: (
        "产后出血", "pph", "postpartum", "指南", "资料", "证据", "引用",
        "模型限制", "医学", "临床", "知识", "evidence", "guideline",
        "guidance", "citation", "material", "source", "model limitation",
        "model limitations", "general limitations", "model material",
        "medical", "clinical",
    ),
}

_CURRENT_REFERENCE = (
    "当前预测", "当前结果", "当前概率", "当前标签", "当前置信度",
    "刚才", "这次", "最新", "预测概率", "current prediction",
    "current result", "latest result", "this result",
)
_PRIOR_REFERENCE = (
    "上一次", "上次", "上一条", "之前一条", "之前的", "旧记录",
    "prior prediction",
    "previous prediction", "previous result", "earlier run",
    "earlier",
)
_MULTIPLE_REFERENCE = (
    "最近两次", "两条", "两个结果", "前后", "multiple predictions",
    "two predictions", "two results", "two saved runs", "latest two",
)
_SKILL_MARKERS = (
    "技能", "流程", "workflow", "skill", "stable process",
    "trusted workflow",
)
_VAGUE_REFERENCE = (
    "这个情况", "当前情况", "处理一下这个", "处理一下当前情况",
    "看看这个", "help me with this", "help with this",
    "with this situation",
)
_EXPLICIT_SAMPLE_PATTERN = re.compile(
    r"(?:第\s*\d+|\d+\s*号|(?:sample|case)\s*\d+|样本\s*\d+)")


def normalize(message: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", message).casefold().split())


def _contains(message: str, markers: tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)


class RuleEvidenceExtractor:
    def __init__(self, registry: TaskRegistry | None = None):
        self._registry = registry or load_default_registry()

    def extract(self, message: str) -> RuleEvidence:
        normalized = normalize(message)
        actions = frozenset(
            action for action, markers in _ACTION_MARKERS.items()
            if _contains(normalized, markers))
        objects = frozenset(
            intent_object for intent_object, markers in _OBJECT_MARKERS.items()
            if _contains(normalized, markers))

        references: set[IntentReference] = set()
        if _EXPLICIT_SAMPLE_PATTERN.search(normalized):
            references.add(IntentReference.EXPLICIT_SAMPLE)
        if _contains(normalized, _CURRENT_REFERENCE):
            references.add(IntentReference.CURRENT_PREDICTION)
        if _contains(normalized, _PRIOR_REFERENCE):
            references.add(IntentReference.PRIOR_PREDICTION)
        if (_contains(normalized, _MULTIPLE_REFERENCE) or
                (IntentAction.COMPARE in actions and
                 IntentReference.CURRENT_PREDICTION in references and
                 IntentReference.PRIOR_PREDICTION in references)):
            references.add(IntentReference.MULTIPLE_PREDICTIONS)

        registry_scopes = frozenset(
            definition.scope
            for definition in self._registry.business_definitions
            if _contains(normalized, definition.rule_terms))
        return RuleEvidence(
            actions=actions,
            objects=objects,
            references=frozenset(references),
            registry_scopes=registry_scopes,
            explicit_skill=_contains(normalized, _SKILL_MARKERS),
            vague_reference=_contains(normalized, _VAGUE_REFERENCE),
        )
