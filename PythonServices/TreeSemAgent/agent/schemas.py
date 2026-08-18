from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RecentMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class PredictionContext(StrictModel):
    prediction_id: str = Field(pattern=r"^pred_[0-9a-f]{32}$")
    model_version: str | None = Field(default=None, max_length=128)


class AgentRunRequest(StrictModel):
    run_id: str = Field(pattern=r"^run_[0-9a-f]{32}$")
    session_id: str = Field(pattern=r"^ses_[0-9a-f]{32}$")
    message: str = Field(min_length=1, max_length=4000)
    recent_messages: list[RecentMessage] = Field(default_factory=list, max_length=12)
    current_prediction: PredictionContext | None = None
    capability_token: str | None = Field(default=None, max_length=8192)

    @field_validator("message")
    @classmethod
    def non_blank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class ToolUse(StrictModel):
    name: str = Field(min_length=1, max_length=64)
    status: Literal["success", "error"]
    duration_ms: int = Field(ge=0)


class AgentRunResponse(StrictModel):
    answer: str = Field(min_length=1, max_length=16000)
    step_count: int = Field(ge=1)
    tools_used: list[ToolUse]
    grounding_prediction_ids: list[str]


class PredictSampleArgs(StrictModel):
    sample_index: int = Field(ge=0)


class PredictionIdArgs(StrictModel):
    prediction_id: str = Field(pattern=r"^pred_[0-9a-f]{32}$")


class HistoryArgs(StrictModel):
    limit: int = Field(default=5, ge=1, le=20)
    cursor: str | None = Field(default=None, max_length=512)


class CompareArgs(StrictModel):
    prediction_id_a: str = Field(pattern=r"^pred_[0-9a-f]{32}$")
    prediction_id_b: str = Field(pattern=r"^pred_[0-9a-f]{32}$")


class LlmToolCall(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class LlmTurn(StrictModel):
    content: str | None = None
    tool_calls: list[LlmToolCall] = Field(default_factory=list)
    grounding_prediction_ids: list[str] = Field(default_factory=list)


class ToolOutputModel(BaseModel):
    """Typed minimum contract while preserving versioned C++ response fields."""
    model_config = ConfigDict(extra="allow", strict=True)


class PredictionToolOutput(ToolOutputModel):
    prediction_id: str = Field(pattern=r"^pred_[0-9a-f]{32}$")


class ExplanationToolOutput(PredictionToolOutput):
    important_features: list[dict[str, Any]]
    decision_path: list[dict[str, Any]]


class PredictionSummaryOutput(ToolOutputModel):
    prediction_id: str = Field(pattern=r"^pred_[0-9a-f]{32}$")


class HistoryToolOutput(ToolOutputModel):
    items: list[PredictionSummaryOutput]
    next_cursor: str | None = None


class ComparisonToolOutput(ToolOutputModel):
    prediction_a: PredictionSummaryOutput
    prediction_b: PredictionSummaryOutput
    label_changed: bool
    model_version_changed: bool
    positive_probability_delta: float
    confidence_delta: float
    cluster_changed: bool
    tree_leaf_changed: bool
    path_changed: bool
    changed_features: list[dict[str, Any]]
