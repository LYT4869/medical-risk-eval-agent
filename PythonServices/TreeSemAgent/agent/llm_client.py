from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

try:
    import httpx
except ModuleNotFoundError:  # Unit tests using ScriptedLlmClient need no HTTP stack.
    httpx = None  # type: ignore[assignment]

from .schemas import LlmToolCall, LlmTurn, LlmUsage
from .final_response import parse_final_response


class LlmError(RuntimeError):
    _CODES = frozenset({
        "llm_failed",
        "upstream_unavailable",
        "upstream_rejected",
        "transport_error",
        "invalid_response",
    })

    def __init__(self, message: str, *, code: str = "llm_failed",
                 retryable: bool = True):
        if code not in self._CODES:
            raise ValueError("unknown LLM error code")
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class LlmToolPolicy:
    mode: Literal["auto", "none", "required"]
    required_tool: str | None = None

    def __post_init__(self) -> None:
        if (self.mode == "required") != (self.required_tool is not None):
            raise ValueError(
                "required Tool policy must name exactly one Tool")

    @classmethod
    def auto(cls) -> LlmToolPolicy:
        return cls("auto")

    @classmethod
    def none(cls) -> LlmToolPolicy:
        return cls("none")

    @classmethod
    def required(cls, name: str) -> LlmToolPolicy:
        if not name:
            raise ValueError("required Tool name must not be empty")
        return cls("required", name)


def optional_boolean_environment(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value == "true"


class LlmClient(Protocol):
    async def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
            timeout: float,
            tool_policy: LlmToolPolicy = LlmToolPolicy.auto()) -> LlmTurn: ...


@dataclass(frozen=True)
class OpenAiCompatibleConfig:
    base_url: str
    model: str
    api_key: str = ""
    connect_timeout_seconds: float = 1.0
    request_timeout_seconds: float = 20.0
    temperature: float = 0.0
    max_output_tokens: int = 1024
    enable_thinking: bool | None = None
    maximum_attempts: int = 2

    def __post_init__(self) -> None:
        if self.maximum_attempts not in {1, 2}:
            raise ValueError("maximum_attempts must be 1 or 2")


class OpenAiCompatibleClient:
    def __init__(self, config: OpenAiCompatibleConfig):
        if httpx is None:
            raise RuntimeError("httpx is required for the OpenAI-compatible client")
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds, connect=config.connect_timeout_seconds),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        self._http_error_type = httpx.HTTPError
        self._usage_lock = threading.Lock()
        self._usage = {
            "request_count": 0,
            "unknown_usage_request_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def usage_snapshot(self) -> dict[str, int]:
        with self._usage_lock:
            return dict(self._usage)

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
            timeout: float,
            tool_policy: LlmToolPolicy = LlmToolPolicy.auto()) -> LlmTurn:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        payload = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_output_tokens,
        }
        if self._config.enable_thinking is not None:
            payload["enable_thinking"] = self._config.enable_thinking
        if tool_policy.mode == "required":
            names = {item["function"]["name"] for item in tools}
            if tool_policy.required_tool not in names:
                raise ValueError("required Tool is not defined")
            payload.update({
                "tools": tools,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": tool_policy.required_tool},
                },
                "parallel_tool_calls": False,
            })
        elif tool_policy.mode == "auto" and tools:
            payload.update({
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            })
        elif tool_policy.mode == "none":
            payload["tool_choice"] = "none"
        last_error: Exception | None = None
        for attempt in range(self._config.maximum_attempts):
            try:
                # Count transport admission, including retries and cancellation.
                # Unknown usage is NOT zero cost; provider billing is authoritative.
                with self._usage_lock:
                    self._usage["request_count"] += 1
                    self._usage["unknown_usage_request_count"] += 1
                response = await self._client.post(
                    "/chat/completions", json=payload, headers=headers,
                    timeout=min(timeout, self._config.request_timeout_seconds),
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise LlmError(
                        f"retryable upstream status {response.status_code}",
                        code="upstream_unavailable", retryable=True)
                if response.status_code >= 400:
                    raise LlmError(
                        f"non-retryable upstream status {response.status_code}",
                        code="upstream_rejected", retryable=False)
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage")
                parsed_usage = None if usage is None else LlmUsage(
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"])
                if parsed_usage is not None:
                    with self._usage_lock:
                        self._usage["unknown_usage_request_count"] -= 1
                        self._usage["prompt_tokens"] += parsed_usage.prompt_tokens
                        self._usage["completion_tokens"] += parsed_usage.completion_tokens
                        self._usage["total_tokens"] += parsed_usage.total_tokens
                message = body["choices"][0]["message"]
                calls = []
                for raw in message.get("tool_calls", []):
                    arguments = raw["function"].get("arguments", "{}")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    calls.append(LlmToolCall(id=raw["id"], name=raw["function"]["name"], arguments=arguments))
                content = message.get("content")
                grounding = []
                source_grounding = []
                final_error = None
                final_structured = False
                if not calls:
                    parsed = parse_final_response(content)
                    final_structured = (not parsed.error and parsed.answer != content)
                    content = parsed.answer
                    grounding = list(parsed.prediction_ids)
                    source_grounding = list(parsed.source_ids)
                    final_error = parsed.error
                return LlmTurn(
                    content=content, tool_calls=calls,
                    grounding_prediction_ids=grounding,
                    grounding_source_ids=source_grounding,
                    usage=parsed_usage, final_response_error=final_error,
                    final_response_is_structured=final_structured)
            except LlmError as exc:
                last_error = exc
                if (exc.retryable and
                        attempt + 1 < self._config.maximum_attempts):
                    await asyncio.sleep(0.05)
                    continue
                raise
            except self._http_error_type as exc:
                last_error = LlmError(
                    "LLM transport failed", code="transport_error",
                    retryable=True)
                if attempt + 1 < self._config.maximum_attempts:
                    await asyncio.sleep(0.05)
                    continue
                raise last_error from exc
            except (ValueError, KeyError, TypeError) as exc:
                raise LlmError(
                    "LLM response was invalid", code="invalid_response",
                    retryable=False) from exc
        raise LlmError("LLM request failed") from last_error


class ScriptedLlmClient:
    """Deterministic test client; each complete() consumes one scripted turn."""
    def __init__(self, turns: list[LlmTurn]):
        self.turns = list(turns)
        self.requests: list[list[dict[str, Any]]] = []
        self.tool_policies: list[LlmToolPolicy] = []

    async def complete(
            self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
            timeout: float,
            tool_policy: LlmToolPolicy = LlmToolPolicy.auto()) -> LlmTurn:
        del tools, timeout
        self.requests.append(messages)
        self.tool_policies.append(tool_policy)
        if not self.turns:
            raise LlmError("script exhausted")
        return self.turns.pop(0)


class ScriptedDemoClient:
    """Local-only deterministic model used by the reproducible interview demo."""

    @staticmethod
    def _tool_names(messages: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for message in messages:
            for call in message.get("tool_calls", []):
                names.append(call.get("function", {}).get("name", ""))
        return names

    @staticmethod
    def _tool_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") != "tool":
                continue
            try:
                value = json.loads(message.get("content", "{}"))
                if isinstance(value, dict): values.append(value)
            except ValueError:
                pass
        return values

    async def close(self) -> None:
        return None

    async def complete(self, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]], timeout: float,
                       tool_policy: LlmToolPolicy = LlmToolPolicy.auto()
                       ) -> LlmTurn:
        del tool_policy
        del timeout
        user = next((str(item.get("content", "")) for item in reversed(messages)
                     if item.get("role") == "user"), "")
        called = self._tool_names(messages)
        available = {item["function"]["name"] for item in tools}
        prediction_match = re.search(r"pred_[0-9a-f]{32}", json.dumps(messages))
        prediction_id = prediction_match.group(0) if prediction_match else None

        def call(name: str, arguments: dict[str, Any]) -> LlmTurn:
            return LlmTurn(tool_calls=[LlmToolCall(
                id=f"demo_{len(called)}", name=name, arguments=arguments)])

        if not called:
            if "activate_skill" in available:
                if any(word in user.lower() for word in ("比较", "compare", "上一次")):
                    return call("activate_skill", {"skill_id": "compare_prediction_history"})
                if any(word in user.lower() for word in ("解释", "explain")):
                    return call("activate_skill", {"skill_id": "explain_prediction"})
                if not any(word in user.lower() for word in ("预测", "predict")):
                    return call("activate_skill", {"skill_id": "pph_evidence_education"})
            if any(word in user.lower() for word in ("预测", "predict")) and "predict_sample" in available:
                return call("predict_sample", {"sample_index": 0})
            if any(word in user.lower() for word in ("解释", "explain")) and prediction_id and "get_explanation" in available:
                return call("get_explanation", {"prediction_id": prediction_id})
            if any(word in user.lower() for word in ("比较", "compare", "上一次")) and "get_prediction_history" in available:
                return call("get_prediction_history", {"limit": 5})
            if "search_medical_knowledge" in available:
                return call("search_medical_knowledge", {
                    "query": user[:500], "scope": "all", "top_k": 5})
        if "activate_skill" in called:
            if (any(word in user.lower() for word in ("解释", "explain")) and
                    prediction_id and "get_explanation" in available and
                    "get_explanation" not in called):
                return call("get_explanation", {"prediction_id": prediction_id})
            if (any(word in user.lower() for word in ("比较", "compare", "上一次")) and
                    "get_prediction_history" in available and
                    "get_prediction_history" not in called):
                return call("get_prediction_history", {"limit": 5})
            if ("search_medical_knowledge" in available and
                    "search_medical_knowledge" not in called):
                return call("search_medical_knowledge", {
                    "query": user[:500], "scope": "all", "top_k": 5})
        if "get_prediction_history" in called and "compare_predictions" not in called:
            ids = re.findall(r"pred_[0-9a-f]{32}", json.dumps(self._tool_payloads(messages)))
            unique = list(dict.fromkeys(ids))
            if len(unique) >= 2 and "compare_predictions" in available:
                return call("compare_predictions", {
                    "prediction_id_a": unique[0], "prediction_id_b": unique[1]})

        payloads = self._tool_payloads(messages)
        prediction_ids = list(dict.fromkeys(re.findall(
            r"pred_[0-9a-f]{32}", json.dumps(payloads))))
        citation_ids = list(dict.fromkeys(re.findall(
            r"cite_[0-9a-f]{20}", json.dumps(payloads))))
        citations = " ".join(citation_ids)
        answer = "这是可复现离线演示回答。模型结果仅用于辅助解释，不能替代医生判断。"
        if "compare_predictions" in called:
            answer = "两次结果的差异来自后端确定性比较接口；不能仅凭模型变化判断病情进展。"
        elif "get_explanation" in called:
            answer = "解释来自已保存的重要特征和决策路径；重要性表示模型关联，不代表因果关系。"
        elif "predict_sample" in called:
            answer = "演示预测已经由 treeSem 工具完成。"
        elif "search_medical_knowledge" in called:
            answer = "以下说明来自受控知识库，不能替代个体化医疗判断。"
        if citations:
            answer += " 引用：" + citations
        content = json.dumps({"answer": answer,
                              "grounding_prediction_ids": prediction_ids,
                              "grounding_source_ids": citation_ids}, ensure_ascii=False)
        return LlmTurn(content=content,
                       grounding_prediction_ids=prediction_ids,
                       grounding_source_ids=citation_ids)
