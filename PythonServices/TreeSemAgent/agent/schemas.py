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
    knowledge_capability_token: str | None = Field(default=None, max_length=8192)
    actor_role: Literal["patient", "doctor"] = "patient"
    request_id: str | None = Field(default=None, pattern=r"^req_[0-9a-f]{32}$")
    traceparent: str | None = Field(
        default=None, pattern=r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

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
    grounding_source_ids: list[str] = Field(default_factory=list)
    citations: list["KnowledgeCitation"] = Field(default_factory=list)
    knowledge_index_version: str | None = Field(default=None, max_length=128)
    skill_used: "SkillUse | None" = None


class KnowledgeCitation(StrictModel):
    citation_id: str = Field(pattern=r"^cite_[0-9a-f]{20}$")
    source_id: str = Field(pattern=r"^src_[A-Za-z0-9_-]{1,76}$")
    title: str = Field(min_length=1, max_length=300)
    section: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    publisher: str = Field(min_length=1, max_length=200)
    published_at: str = Field(max_length=64)
    url: str = Field(max_length=2048)


class SkillUse(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    version: str = Field(pattern=r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
    catalog_version: str = Field(pattern=r"^[0-9a-f]{64}$")


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


class KnowledgeSearchArgs(StrictModel):
    query: str = Field(min_length=1, max_length=500)
    scope: Literal["model", "clinical", "all"] = "all"
    top_k: int = Field(default=5, ge=1, le=6)


class SkillActivationArgs(StrictModel):
    skill_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")


class LlmToolCall(StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]


class LlmUsage(StrictModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class LlmTurn(StrictModel):
    content: str | None = None
    tool_calls: list[LlmToolCall] = Field(default_factory=list)
    grounding_prediction_ids: list[str] = Field(default_factory=list)
    grounding_source_ids: list[str] = Field(default_factory=list)
    usage: LlmUsage | None = None


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


class KnowledgeResult(StrictModel):
    citation_id: str = Field(pattern=r"^cite_[0-9a-f]{20}$")
    source_id: str = Field(pattern=r"^src_[A-Za-z0-9_-]{1,76}$")
    title: str = Field(min_length=1, max_length=300)
    section: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    excerpt: str = Field(min_length=1, max_length=800)
    publisher: str = Field(min_length=1, max_length=200)
    published_at: str = Field(max_length=64)
    url: str = Field(max_length=2048)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    score: float


class KnowledgeToolOutput(StrictModel):
    index_version: str = Field(min_length=1, max_length=128)
    retrieval_mode: Literal["hybrid", "lexical_degraded", "dense_degraded", "rrf_degraded"]
    results: list[KnowledgeResult] = Field(max_length=6)
