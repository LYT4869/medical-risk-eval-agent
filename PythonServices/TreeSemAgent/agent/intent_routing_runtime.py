from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from .intent_frame import IntentFrame
from .intent_router_prompt import ROUTER_PROMPT_SHA256
from .llm_client import OpenAiCompatibleClient, OpenAiCompatibleConfig
from .structured_router import (
    RouterContext,
    StructuredIntentRouter,
    StructuredRoute,
    StructuredRouterConfig,
)
from .workflow_registry import default_workflow_registry


class IntentRoutingMode(str, Enum):
    LEGACY_RULE = "legacy_rule"
    STRUCTURED_SHADOW = "structured_shadow"
    STRUCTURED_LLM = "structured_llm"


def _integer(values: Mapping[str, str], name: str, default: int, *,
             minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(values.get(name, str(default)))
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else \
            f"at least {minimum}"
        raise RuntimeError(f"{name} must be {bounds}")
    return value


@dataclass(frozen=True)
class IntentRoutingSettings:
    mode: IntentRoutingMode
    llm_mode: str
    base_url: str
    router_model: str
    api_key: str
    router_config: StructuredRouterConfig

    @classmethod
    def from_environment(
            cls, environment: Mapping[str, str] | None = None
            ) -> "IntentRoutingSettings":
        values = os.environ if environment is None else environment
        raw_mode = values.get(
            "TREESEM_AGENT_ROUTING_MODE",
            IntentRoutingMode.LEGACY_RULE.value)
        try:
            mode = IntentRoutingMode(raw_mode)
        except ValueError as exc:
            raise RuntimeError(
                "TREESEM_AGENT_ROUTING_MODE must be legacy_rule, "
                "structured_shadow or structured_llm") from exc

        llm_mode = values.get("TREESEM_AGENT_LLM_MODE", "real").strip()
        if llm_mode not in {"real", "scripted_demo"}:
            raise RuntimeError(
                "TREESEM_AGENT_LLM_MODE must be real or scripted_demo")
        base_url = values.get("TREESEM_AGENT_LLM_BASE_URL", "").strip()
        model = values.get("TREESEM_AGENT_LLM_MODEL", "").strip()
        if (mode != IntentRoutingMode.LEGACY_RULE and llm_mode == "real" and
                (not base_url or not model)):
            raise RuntimeError(
                "TREESEM_AGENT_LLM_BASE_URL and TREESEM_AGENT_LLM_MODEL "
                "are required for structured routing")

        request_ms = _integer(
            values, "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS", 3000)
        deadline_ms = _integer(
            values, "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS", 7000)
        attempts = _integer(
            values, "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS", 2, maximum=2)
        backoff_ms = _integer(
            values, "TREESEM_AGENT_ROUTER_RETRY_BACKOFF_MS", 100,
            minimum=0)
        context_messages = _integer(
            values, "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES", 4,
            minimum=0, maximum=4)
        context_chars = _integer(
            values, "TREESEM_AGENT_ROUTER_CONTEXT_MAX_CHARS", 6000)
        required_ms = request_ms * attempts
        if attempts > 1:
            required_ms += backoff_ms
        if deadline_ms < required_ms:
            raise RuntimeError(
                "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS cannot cover "
                "the configured attempts and retry backoff")

        return cls(
            mode=mode,
            llm_mode=llm_mode,
            base_url=base_url,
            router_model=(model if llm_mode == "real" else "scripted_demo"),
            api_key=values.get("TREESEM_AGENT_LLM_API_KEY", ""),
            router_config=StructuredRouterConfig(
                request_timeout_seconds=request_ms / 1000.0,
                total_deadline_seconds=deadline_ms / 1000.0,
                maximum_attempts=attempts,
                retry_backoff_seconds=backoff_ms / 1000.0,
                context_messages=context_messages,
                context_max_chars=context_chars,
            ),
        )


class ScriptedStructuredRouter:
    """Deterministic local-demo fixture; never used in a real deployment."""

    @staticmethod
    def _frame(context: RouterContext) -> IntentFrame:
        text = context.message
        lowered = text.lower()
        evidence = text.strip()[:160] or "request"
        intent = "other"
        target = "none"
        sample_index = None
        aspects: list[str] = []
        knowledge_scope = None
        if context.references.sample_indexes:
            intent = "prediction"
            target = "demo_sample"
            sample_index = 0
        elif any(word in lowered for word in ("比较", "compare")):
            intent = "comparison"
            target = "latest_two_predictions"
            aspects = ["comparison_changes"]
        elif any(word in lowered for word in ("解释", "explain")):
            intent = "explanation"
            target = "current_prediction"
            aspects = ["important_features", "decision_path"]
        elif any(word in lowered for word in (
                "产后出血", "pph", "指南", "知识", "what is")):
            intent = "knowledge"
            target = "general_knowledge"
            aspects = ["knowledge_overview", "citations"]
            knowledge_scope = "all"
        return IntentFrame.model_validate({
            "schema_version": 1,
            "goals": [{
                "intent": intent,
                "target": {
                    "type": target,
                    "explicit_reference_index": None,
                    "second_explicit_reference_index": None,
                    "sample_reference_index": sample_index,
                },
                "requested_aspects": aspects,
                "knowledge_scope": knowledge_scope,
                "evidence": [evidence],
            }],
            "constraints": {
                "excluded_intents": [],
                "excluded_aspects": [],
            },
            "unresolved_references": [],
            "needs_clarification": False,
        })

    async def route(self, context: RouterContext) -> StructuredRoute:
        return StructuredRoute(
            self._frame(context), usage=None, attempt_count=1,
            repaired=False)


class IntentRoutingRuntime:
    def __init__(self, settings: IntentRoutingSettings,
                 structured_router=None, owned_client=None):
        self._settings = settings
        self.structured_router = structured_router
        self._owned_client = owned_client
        self._closed = False

    @property
    def mode(self) -> IntentRoutingMode:
        return self._settings.mode

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "routing_mode": self.mode.value,
            "router_model": self._settings.router_model,
            "router_prompt_sha256": ROUTER_PROMPT_SHA256,
            "intent_frame_schema_version": 1,
            "workflow_registry_version": default_workflow_registry().version,
        }

    async def route(self, context: RouterContext) -> StructuredRoute:
        if self.structured_router is None:
            raise RuntimeError("structured Router is disabled")
        return await self.structured_router.route(context)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owned_client is not None:
            await self._owned_client.close()


def build_intent_routing_runtime(
        settings: IntentRoutingSettings, *,
        client_factory: Callable[[OpenAiCompatibleConfig], object] =
        OpenAiCompatibleClient,
        scripted_router_factory: Callable[[StructuredRouterConfig], object] =
        lambda config: ScriptedStructuredRouter()) -> IntentRoutingRuntime:
    if settings.mode == IntentRoutingMode.LEGACY_RULE:
        return IntentRoutingRuntime(settings)
    if settings.llm_mode == "scripted_demo":
        router = scripted_router_factory(settings.router_config)
        return IntentRoutingRuntime(settings, router)

    client = client_factory(OpenAiCompatibleConfig(
        base_url=settings.base_url,
        model=settings.router_model,
        api_key=settings.api_key,
        connect_timeout_seconds=min(
            1.0, settings.router_config.request_timeout_seconds),
        request_timeout_seconds=settings.router_config.request_timeout_seconds,
        temperature=0.0,
        max_output_tokens=384,
        enable_thinking=False,
        maximum_attempts=1,
    ))
    router = StructuredIntentRouter(client, settings.router_config)
    return IntentRoutingRuntime(settings, router, client)
