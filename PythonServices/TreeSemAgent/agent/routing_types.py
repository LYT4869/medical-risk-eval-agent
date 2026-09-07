from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RequestScope(str, Enum):
    SKILL = "skill"
    PREDICTION = "prediction"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    HISTORY = "history"
    COMPARISON = "comparison"
    KNOWLEDGE = "knowledge"
    SECURITY_ABUSE = "security_abuse"
    MEDICAL_REFUSAL = "medical_refusal"
    UNKNOWN = "unknown"


class RoutingSource(str, Enum):
    RULE = "rule"
    SEMANTIC = "semantic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RoutingDecision:
    scope: RequestScope
    source: RoutingSource
    similarity_score: float | None = None
    margin: float | None = None
    secondary_score: float | None = None
    reason: str | None = None
    include_summary: bool = False

