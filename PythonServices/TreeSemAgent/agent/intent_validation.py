from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .intent_frame import (
    IntentFrame,
    IntentKind,
    KnowledgeScope,
    RequestedSkill,
    RequestedAspect,
    TargetKind,
    UnresolvedReference,
)
from .reference_extractor import ReferenceExtraction
from .schemas import AgentRunRequest


class IntentFrameViolation(ValueError):
    _CODES = {
        "invalid_candidate_index",
        "fabricated_evidence",
        "incompatible_target",
        "incompatible_aspect",
        "invalid_clarification_state",
        "invalid_knowledge_scope",
        "incompatible_skill",
    }

    def __init__(self, code: str):
        if code not in self._CODES:
            raise ValueError("unknown IntentFrame violation code")
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class BoundTarget:
    kind: TargetKind
    prediction_ids: tuple[str, ...] = ()
    sample_index: int | None = None


@dataclass(frozen=True)
class BoundGoal:
    intent: IntentKind
    target: BoundTarget
    requested_aspects: tuple[RequestedAspect, ...]
    knowledge_scope: KnowledgeScope | None


@dataclass(frozen=True)
class ValidatedIntent:
    frame: IntentFrame
    goals: tuple[BoundGoal, ...]
    excluded_intents: frozenset[IntentKind]
    excluded_aspects: frozenset[RequestedAspect]
    requested_skill: RequestedSkill | None


@dataclass(frozen=True)
class IntentValidationResult:
    validated: ValidatedIntent | None = None
    clarification_code: str | None = None

    def __post_init__(self) -> None:
        if (self.validated is None) == (self.clarification_code is None):
            raise ValueError(
                "validation result requires exactly one outcome")


_TARGETS = {
    IntentKind.PREDICTION: {TargetKind.DEMO_SAMPLE},
    IntentKind.SUMMARY: {
        TargetKind.CURRENT_PREDICTION,
        TargetKind.PREVIOUS_PREDICTION,
        TargetKind.EXPLICIT_PREDICTION,
    },
    IntentKind.EXPLANATION: {
        TargetKind.CURRENT_PREDICTION,
        TargetKind.PREVIOUS_PREDICTION,
        TargetKind.LATEST_TWO_PREDICTIONS,
        TargetKind.EXPLICIT_PREDICTION,
        TargetKind.EXPLICIT_PREDICTION_PAIR,
    },
    IntentKind.HISTORY: {TargetKind.SESSION_HISTORY},
    IntentKind.COMPARISON: {
        TargetKind.LATEST_TWO_PREDICTIONS,
        TargetKind.EXPLICIT_PREDICTION_PAIR,
    },
    IntentKind.KNOWLEDGE: {TargetKind.GENERAL_KNOWLEDGE},
    IntentKind.OTHER: set(TargetKind),
}

_ASPECTS = {
    IntentKind.PREDICTION: {
        RequestedAspect.PREDICTION_SUMMARY,
        RequestedAspect.LABEL,
        RequestedAspect.PROBABILITY,
        RequestedAspect.CONFIDENCE,
        RequestedAspect.MODEL_VERSION,
    },
    IntentKind.SUMMARY: {
        RequestedAspect.PREDICTION_SUMMARY,
        RequestedAspect.LABEL,
        RequestedAspect.PROBABILITY,
        RequestedAspect.CONFIDENCE,
        RequestedAspect.MODEL_VERSION,
    },
    IntentKind.EXPLANATION: {
        RequestedAspect.IMPORTANT_FEATURES,
        RequestedAspect.DECISION_PATH,
    },
    IntentKind.HISTORY: {
        RequestedAspect.HISTORY_ITEMS,
        RequestedAspect.PREDICTION_SUMMARY,
    },
    IntentKind.COMPARISON: {RequestedAspect.COMPARISON_CHANGES},
    IntentKind.KNOWLEDGE: {
        RequestedAspect.KNOWLEDGE_OVERVIEW,
    },
    IntentKind.OTHER: set(RequestedAspect),
}

_COMPARISON_DETAIL_ASPECTS = {
    RequestedAspect.PREDICTION_SUMMARY,
    RequestedAspect.LABEL,
    RequestedAspect.PROBABILITY,
    RequestedAspect.CONFIDENCE,
    RequestedAspect.MODEL_VERSION,
    RequestedAspect.IMPORTANT_FEATURES,
    RequestedAspect.DECISION_PATH,
    RequestedAspect.COMPARISON_CHANGES,
}

_SKILL_GOALS = {
    RequestedSkill.EXPLAIN_PREDICTION: {
        (IntentKind.EXPLANATION, TargetKind.CURRENT_PREDICTION),
        (IntentKind.EXPLANATION, TargetKind.EXPLICIT_PREDICTION),
    },
    RequestedSkill.COMPARE_PREDICTION_HISTORY: {
        (IntentKind.COMPARISON, TargetKind.LATEST_TWO_PREDICTIONS),
    },
    RequestedSkill.PPH_EVIDENCE_EDUCATION: {
        (IntentKind.KNOWLEDGE, TargetKind.GENERAL_KNOWLEDGE),
    },
}

_CLARIFICATION = {
    UnresolvedReference.MISSING_SAMPLE_INDEX: "sample_index_missing",
    UnresolvedReference.MISSING_PREDICTION_TARGET:
        "prediction_target_missing",
    UnresolvedReference.MISSING_COMPARISON_TARGET:
        "comparison_target_missing",
    UnresolvedReference.AMBIGUOUS_REFERENCE: "ambiguous_reference",
}

_EXPLICIT_SKILL_MARKERS = (
    "技能", "激活", "skill", "activate",
    "explain_prediction", "compare_prediction_history",
    "pph_evidence_education",
)
_EXPLICIT_SKILL_PATTERNS = (
    re.compile(
        r"(?:可信|稳定|患者版|医生版|循证|预测解释|历史比较|教育).{0,12}流程"),
    re.compile(
        r"\b(?:use|run|activate)\b.{0,40}\bworkflow\b"),
    re.compile(
        r"\b(?:trusted|stable)\b.{0,20}\b(?:workflow|process)\b"),
)


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _explicit_skill_requested(normalized_message: str) -> bool:
    return (
        any(marker in normalized_message
            for marker in _EXPLICIT_SKILL_MARKERS) or
        any(pattern.search(normalized_message)
            for pattern in _EXPLICIT_SKILL_PATTERNS)
    )


def _clarification_for_unresolved(
        values: list[UnresolvedReference]) -> str:
    priority = (
        UnresolvedReference.AMBIGUOUS_REFERENCE,
        UnresolvedReference.MISSING_SAMPLE_INDEX,
        UnresolvedReference.MISSING_COMPARISON_TARGET,
        UnresolvedReference.MISSING_PREDICTION_TARGET,
    )
    for item in priority:
        if item in values:
            return _CLARIFICATION[item]
    raise IntentFrameViolation("invalid_clarification_state")


def _bind_target(goal, references: ReferenceExtraction,
                 request: AgentRunRequest) -> BoundTarget | str:
    target = goal.target
    if target.type == TargetKind.EXPLICIT_PREDICTION:
        index = target.explicit_reference_index
        if index is None or index >= len(references.prediction_ids):
            raise IntentFrameViolation("invalid_candidate_index")
        return BoundTarget(target.type, (references.prediction_ids[index],))
    if target.type == TargetKind.EXPLICIT_PREDICTION_PAIR:
        first = target.explicit_reference_index
        second = target.second_explicit_reference_index
        if (first is None or second is None or
                first >= len(references.prediction_ids) or
                second >= len(references.prediction_ids)):
            raise IntentFrameViolation("invalid_candidate_index")
        return BoundTarget(target.type, (
            references.prediction_ids[first],
            references.prediction_ids[second],
        ))
    if target.type == TargetKind.DEMO_SAMPLE:
        index = target.sample_reference_index
        if index is None or index >= len(references.sample_indexes):
            raise IntentFrameViolation("invalid_candidate_index")
        return BoundTarget(
            target.type, sample_index=references.sample_indexes[index])
    if target.type == TargetKind.CURRENT_PREDICTION:
        if request.current_prediction is None:
            return "current_prediction_missing"
        return BoundTarget(
            target.type, (request.current_prediction.prediction_id,))
    return BoundTarget(target.type)


def _validated_aspects(goal) -> tuple[RequestedAspect, ...]:
    """One pure cross-field contract, shared by parsing and business binding."""
    if goal.target.type not in _TARGETS[goal.intent]:
        raise IntentFrameViolation("incompatible_target")
    aspects = tuple(goal.requested_aspects)
    if (goal.intent == IntentKind.COMPARISON and aspects and
            set(aspects) <= _COMPARISON_DETAIL_ASPECTS):
        aspects = (RequestedAspect.COMPARISON_CHANGES,)
    if not set(aspects) <= _ASPECTS[goal.intent]:
        raise IntentFrameViolation("incompatible_aspect")
    if ((goal.intent == IntentKind.KNOWLEDGE and goal.knowledge_scope is None) or
            (goal.intent != IntentKind.KNOWLEDGE and goal.knowledge_scope is not None)):
        raise IntentFrameViolation("invalid_knowledge_scope")
    return aspects


def _source_resolves_missing(frame: IntentFrame) -> bool:
    return (set(frame.unresolved_references) == {UnresolvedReference.MISSING_PREDICTION_TARGET}
            and len(frame.goals) == 1
            and frame.goals[0].target.type == TargetKind.EXPLICIT_PREDICTION)


def validate_frame_contract(frame: IntentFrame) -> None:
    """Validate repairable structure only; never resolve IDs or query resources."""
    if frame.needs_clarification != bool(frame.unresolved_references):
        raise IntentFrameViolation("invalid_clarification_state")
    if frame.unresolved_references and not _source_resolves_missing(frame):
        return
    if any(goal.intent in frame.constraints.excluded_intents or
           set(goal.requested_aspects).intersection(frame.constraints.excluded_aspects)
           for goal in frame.goals):
        return  # Existing business validator asks for clarification on conflict.
    for goal in frame.goals:
        _validated_aspects(goal)


def contract_repair_guidance(frame: IntentFrame) -> dict:
    """Enum-only feedback derived from the authoritative contract, not Tool data."""
    return {intent.value: {
        "targets": sorted(t.value for t in _TARGETS[intent]),
        "aspects": sorted(a.value for a in _ASPECTS[intent]),
        "knowledge_scope": "model|clinical|all" if intent == IntentKind.KNOWLEDGE else None,
    } for intent in dict.fromkeys(goal.intent for goal in frame.goals)}


def validate_and_bind_intent(
        frame: IntentFrame, references: ReferenceExtraction,
        request: AgentRunRequest) -> IntentValidationResult:
    normalized_message = _normalize(references.router_message)
    for goal in frame.goals:
        if any(_normalize(item) not in normalized_message
               for item in goal.evidence):
            raise IntentFrameViolation("fabricated_evidence")

    has_unresolved = bool(frame.unresolved_references)
    if frame.needs_clarification != has_unresolved:
        raise IntentFrameViolation("invalid_clarification_state")

    excluded_intents = frozenset(frame.constraints.excluded_intents)
    excluded_aspects = frozenset(frame.constraints.excluded_aspects)
    if (any(goal.intent in excluded_intents for goal in frame.goals) or
            any(excluded_aspects.intersection(goal.requested_aspects)
                for goal in frame.goals)):
        return IntentValidationResult(
            clarification_code="conflicting_request")

    # A source-backed explicit candidate can resolve only the narrow "missing
    # target" claim. It cannot establish record existence or override ambiguity.
    source_resolves_missing = _source_resolves_missing(frame)
    if source_resolves_missing:
        _bind_target(frame.goals[0], references, request)
    if has_unresolved and not source_resolves_missing:
        return IntentValidationResult(
            clarification_code=_clarification_for_unresolved(
                frame.unresolved_references))

    bound_goals: list[BoundGoal] = []
    for goal in frame.goals:
        requested_aspects = _validated_aspects(goal)
        bound = _bind_target(goal, references, request)
        if isinstance(bound, str):
            return IntentValidationResult(clarification_code=bound)
        bound_goals.append(BoundGoal(
            goal.intent,
            bound,
            requested_aspects,
            goal.knowledge_scope,
        ))

    requested_skill = frame.requested_skill
    if (requested_skill is not None and
            not _explicit_skill_requested(normalized_message)):
        requested_skill = None
    if requested_skill is not None:
        signature = (bound_goals[0].intent, bound_goals[0].target.kind)
        if (len(bound_goals) != 1 or signature not in
                _SKILL_GOALS[requested_skill]):
            raise IntentFrameViolation("incompatible_skill")

    return IntentValidationResult(validated=ValidatedIntent(
        frame=frame,
        goals=tuple(bound_goals),
        excluded_intents=excluded_intents,
        excluded_aspects=excluded_aspects,
        requested_skill=requested_skill,
    ))
