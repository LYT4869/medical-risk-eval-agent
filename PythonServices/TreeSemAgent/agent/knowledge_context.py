"""Bounded public topic hints, never historical patient messages or facts."""
from __future__ import annotations

import re

from .schemas import AgentRunRequest


# Closed public terminology, not an intent router or clinical classifier. Only
# canonical names can leave this extractor; no values, IDs or arbitrary text.
_TOPICS = {
    "macro-f1": "macro-F1", "f1": "F1", "auc": "AUC", "auprc": "AUPRC",
    "accuracy": "Accuracy", "positive_probability": "positive_probability",
    "confidence": "confidence", "置信度": "confidence",
    "probability": "probability", "概率": "probability",
    "label": "label", "标签": "label", "standardscaler": "StandardScaler",
    "标准化": "standardization", "反标准化": "inverse standardization",
    "决策路径": "decision path", "decision path": "decision path",
    "重要特征": "important features", "important features": "important features",
    "pph": "PPH", "产后出血": "PPH",
}
_TERM = re.compile(
    r"(?<![A-Za-z0-9_])(?:" + "|".join(
        re.escape(term) for term in sorted(_TOPICS, key=len, reverse=True)) +
    r")(?![A-Za-z0-9_])", re.I)
_REFERENCE = re.compile(
    r"它|该(?:指标|术语|概念)|这个(?:指标|术语|概念)|"
    r"\bit\b|\bthat (?:metric|term|concept)\b", re.I)


def knowledge_topic_hint(request: AgentRunRequest) -> tuple[str | None, bool]:
    """Return one public topic or ask for clarification on unresolved reference."""
    if _TERM.search(request.message) or not _REFERENCE.search(request.message):
        return None, False
    for item in reversed(request.recent_messages[-4:]):
        topics = list(dict.fromkeys(_TOPICS[match.group().lower()]
                                   for match in _TERM.finditer(item.content[-1000:])))
        if topics:
            return (topics[0], False) if len(topics) == 1 else (None, True)
    return None, True
