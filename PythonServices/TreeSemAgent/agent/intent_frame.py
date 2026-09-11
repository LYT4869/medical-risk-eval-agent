from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentKind(str, Enum):
    PREDICTION = "prediction"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    HISTORY = "history"
    COMPARISON = "comparison"
    KNOWLEDGE = "knowledge"
    OTHER = "other"


class TargetKind(str, Enum):
    DEMO_SAMPLE = "demo_sample"
    CURRENT_PREDICTION = "current_prediction"
    PREVIOUS_PREDICTION = "previous_prediction"
    LATEST_TWO_PREDICTIONS = "latest_two_predictions"
    EXPLICIT_PREDICTION = "explicit_prediction"
    EXPLICIT_PREDICTION_PAIR = "explicit_prediction_pair"
    SESSION_HISTORY = "session_history"
    GENERAL_KNOWLEDGE = "general_knowledge"
    NONE = "none"


class RequestedAspect(str, Enum):
    PREDICTION_SUMMARY = "prediction_summary"
    LABEL = "label"
    PROBABILITY = "probability"
    CONFIDENCE = "confidence"
    MODEL_VERSION = "model_version"
    IMPORTANT_FEATURES = "important_features"
    DECISION_PATH = "decision_path"
    HISTORY_ITEMS = "history_items"
    COMPARISON_CHANGES = "comparison_changes"
    KNOWLEDGE_OVERVIEW = "knowledge_overview"


class RequestedSkill(str, Enum):
    EXPLAIN_PREDICTION = "explain_prediction"
    COMPARE_PREDICTION_HISTORY = "compare_prediction_history"
    PPH_EVIDENCE_EDUCATION = "pph_evidence_education"


class KnowledgeScope(str, Enum):
    MODEL = "model"
    CLINICAL = "clinical"
    ALL = "all"


class UnresolvedReference(str, Enum):
    MISSING_SAMPLE_INDEX = "missing_sample_index"
    MISSING_PREDICTION_TARGET = "missing_prediction_target"
    MISSING_COMPARISON_TARGET = "missing_comparison_target"
    AMBIGUOUS_REFERENCE = "ambiguous_reference"


IntentValue = Annotated[IntentKind, Field(strict=False)]
TargetValue = Annotated[TargetKind, Field(strict=False)]
AspectValue = Annotated[RequestedAspect, Field(strict=False)]
KnowledgeScopeValue = Annotated[KnowledgeScope, Field(strict=False)]
UnresolvedValue = Annotated[UnresolvedReference, Field(strict=False)]
RequestedSkillValue = Annotated[RequestedSkill, Field(strict=False)]


class StrictFrameModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class IntentTarget(StrictFrameModel):
    type: TargetValue
    explicit_reference_index: int | None = Field(default=None, ge=0, le=7)
    second_explicit_reference_index: int | None = Field(
        default=None, ge=0, le=7)
    sample_reference_index: int | None = Field(default=None, ge=0, le=7)

    @model_validator(mode="after")
    def validate_candidate_indexes(self) -> "IntentTarget":
        explicit = self.explicit_reference_index
        second = self.second_explicit_reference_index
        sample = self.sample_reference_index
        if self.type == TargetKind.EXPLICIT_PREDICTION:
            if explicit is None or second is not None or sample is not None:
                raise ValueError(
                    "explicit_prediction requires one explicit reference")
        elif self.type == TargetKind.EXPLICIT_PREDICTION_PAIR:
            if (explicit is None or second is None or explicit == second or
                    sample is not None):
                raise ValueError(
                    "explicit_prediction_pair requires two distinct references")
        elif self.type == TargetKind.DEMO_SAMPLE:
            if sample is None or explicit is not None or second is not None:
                raise ValueError("demo_sample requires one sample reference")
        elif explicit is not None or second is not None or sample is not None:
            raise ValueError("symbolic targets cannot contain candidate indexes")
        return self


class IntentGoal(StrictFrameModel):
    intent: IntentValue
    target: IntentTarget
    requested_aspects: list[AspectValue] = Field(
        default_factory=list, max_length=8)
    knowledge_scope: KnowledgeScopeValue | None = None
    evidence: list[str] = Field(min_length=1, max_length=3)

    @field_validator("requested_aspects")
    @classmethod
    def unique_aspects(cls, value: list[RequestedAspect]) -> list[RequestedAspect]:
        if len(set(value)) != len(value):
            raise ValueError("requested_aspects contains duplicates")
        return value

    @field_validator("evidence")
    @classmethod
    def valid_evidence(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 160 for item in value):
            raise ValueError("evidence must contain short non-blank strings")
        if len(set(value)) != len(value):
            raise ValueError("evidence contains duplicates")
        return value


class IntentConstraints(StrictFrameModel):
    excluded_intents: list[IntentValue] = Field(
        default_factory=list, max_length=8)
    excluded_aspects: list[AspectValue] = Field(
        default_factory=list, max_length=8)

    @field_validator("excluded_intents", "excluded_aspects")
    @classmethod
    def unique_values(cls, value: list[Enum]) -> list[Enum]:
        if len(set(value)) != len(value):
            raise ValueError("constraint list contains duplicates")
        return value


class IntentFrame(StrictFrameModel):
    schema_version: int = Field(ge=2, le=2)
    goals: list[IntentGoal] = Field(max_length=3)
    constraints: IntentConstraints
    unresolved_references: list[UnresolvedValue] = Field(
        default_factory=list, max_length=4)
    needs_clarification: bool
    requested_skill: RequestedSkillValue | None = None

    @model_validator(mode="after")
    def validate_empty_clarification(self) -> "IntentFrame":
        if not self.goals and not (
                self.needs_clarification and self.unresolved_references and
                self.requested_skill is None):
            raise ValueError(
                "empty goals require an unresolved clarification")
        return self

    @field_validator("unresolved_references")
    @classmethod
    def unique_unresolved(
            cls, value: list[UnresolvedReference]) -> list[UnresolvedReference]:
        if len(set(value)) != len(value):
            raise ValueError("unresolved_references contains duplicates")
        return value
