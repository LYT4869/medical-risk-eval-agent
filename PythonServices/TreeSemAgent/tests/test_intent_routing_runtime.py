from __future__ import annotations

import asyncio
import unittest

from agent.intent_frame import IntentFrame
from agent.intent_routing_runtime import (
    IntentRoutingMode,
    IntentRoutingSettings,
    build_intent_routing_runtime,
)
from agent.llm_client import LlmTurn, ScriptedLlmClient
from agent.loop import AgentLoop
from agent.schemas import AgentRunRequest
from agent.structured_router import StructuredRoute, StructuredRouterError
from agent.tool_registry import ToolRegistry


class CloseCountingClient:
    def __init__(self):
        self.close_calls = 0

    async def complete(self, messages, tools, timeout, tool_policy):
        del messages, tools, timeout, tool_policy
        raise AssertionError("the runtime construction test must not call the LLM")

    async def close(self):
        self.close_calls += 1


class RecordingRouter:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    async def route(self, context):
        del context
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return StructuredRoute(
            self.outcome, usage=None, attempt_count=1, repaired=False)


class EmptyBackend:
    pass


def other_frame() -> IntentFrame:
    return IntentFrame.model_validate({
        "schema_version": 1,
        "goals": [{
            "intent": "other",
            "target": {
                "type": "none",
                "explicit_reference_index": None,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": [],
            "knowledge_scope": None,
            "evidence": ["你好"],
        }],
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": [],
        },
        "unresolved_references": [],
        "needs_clarification": False,
    })


def request() -> AgentRunRequest:
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message="你好",
    )


class IntentRoutingSettingsTest(unittest.TestCase):
    def test_environment_parses_exact_structured_limits(self):
        settings = IntentRoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
            "TREESEM_AGENT_LLM_MODE": "real",
            "TREESEM_AGENT_LLM_BASE_URL": "https://llm.example/v1",
            "TREESEM_AGENT_LLM_MODEL": "qwen-test",
            "TREESEM_AGENT_LLM_API_KEY": "secret",
            "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS": "3000",
            "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS": "7000",
            "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS": "2",
            "TREESEM_AGENT_ROUTER_RETRY_BACKOFF_MS": "100",
            "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES": "4",
            "TREESEM_AGENT_ROUTER_CONTEXT_MAX_CHARS": "6000",
        })

        self.assertEqual(settings.mode, IntentRoutingMode.STRUCTURED_LLM)
        self.assertEqual(settings.router_model, "qwen-test")
        self.assertEqual(settings.router_config.request_timeout_seconds, 3.0)
        self.assertEqual(settings.router_config.total_deadline_seconds, 7.0)
        self.assertEqual(settings.router_config.maximum_attempts, 2)
        self.assertEqual(settings.router_config.retry_backoff_seconds, 0.1)
        self.assertEqual(settings.router_config.context_messages, 4)
        self.assertEqual(settings.router_config.context_max_chars, 6000)

    def test_environment_rejects_unknown_mode_and_invalid_limits(self):
        base = {
            "TREESEM_AGENT_LLM_MODE": "real",
            "TREESEM_AGENT_LLM_BASE_URL": "https://llm.example/v1",
            "TREESEM_AGENT_LLM_MODEL": "qwen-test",
        }
        cases = (
            ({"TREESEM_AGENT_ROUTING_MODE": "hybrid_optional"}, "ROUTING_MODE"),
            ({"TREESEM_AGENT_ROUTING_MODE": "structured_llm",
              "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS": "0"},
             "REQUEST_TIMEOUT"),
            ({"TREESEM_AGENT_ROUTING_MODE": "structured_llm",
              "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS": "3"},
             "MAX_ATTEMPTS"),
            ({"TREESEM_AGENT_ROUTING_MODE": "structured_llm",
              "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES": "5"},
             "CONTEXT_MESSAGES"),
            ({"TREESEM_AGENT_ROUTING_MODE": "structured_llm",
              "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS": "6000"},
             "TOTAL_DEADLINE"),
        )
        for values, expected in cases:
            with self.subTest(values=values), self.assertRaisesRegex(
                    RuntimeError, expected):
                IntentRoutingSettings.from_environment({**base, **values})

    def test_real_structured_mode_requires_shared_agent_endpoint_and_model(self):
        for missing in ("TREESEM_AGENT_LLM_BASE_URL",
                        "TREESEM_AGENT_LLM_MODEL"):
            values = {
                "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
                "TREESEM_AGENT_LLM_MODE": "real",
                "TREESEM_AGENT_LLM_BASE_URL": "https://llm.example/v1",
                "TREESEM_AGENT_LLM_MODEL": "qwen-test",
            }
            values.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(
                    RuntimeError, "BASE_URL and TREESEM_AGENT_LLM_MODEL"):
                IntentRoutingSettings.from_environment(values)


class IntentRoutingRuntimeTest(unittest.TestCase):
    def test_legacy_mode_builds_no_router_client(self):
        settings = IntentRoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "legacy_rule",
            "TREESEM_AGENT_LLM_MODE": "scripted_demo",
        })
        calls = []
        runtime = build_intent_routing_runtime(
            settings, client_factory=lambda config: calls.append(config))

        self.assertEqual(calls, [])
        self.assertIsNone(runtime.structured_router)
        self.assertEqual(runtime.mode, IntentRoutingMode.LEGACY_RULE)

    def test_real_router_client_has_one_transport_attempt_and_fixed_generation(self):
        settings = IntentRoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
            "TREESEM_AGENT_LLM_MODE": "real",
            "TREESEM_AGENT_LLM_BASE_URL": "https://llm.example/v1",
            "TREESEM_AGENT_LLM_MODEL": "qwen-test",
            "TREESEM_AGENT_LLM_API_KEY": "secret",
        })
        client = CloseCountingClient()
        configs = []
        runtime = build_intent_routing_runtime(
            settings,
            client_factory=lambda config: configs.append(config) or client)

        self.assertIsNotNone(runtime.structured_router)
        self.assertEqual(configs[0].maximum_attempts, 1)
        self.assertEqual(configs[0].temperature, 0.0)
        self.assertEqual(configs[0].max_output_tokens, 384)
        self.assertFalse(configs[0].enable_thinking)
        asyncio.run(runtime.close())
        asyncio.run(runtime.close())
        self.assertEqual(client.close_calls, 1)

    def test_scripted_mode_uses_injected_router_without_cloud_client(self):
        settings = IntentRoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
            "TREESEM_AGENT_LLM_MODE": "scripted_demo",
        })
        router = RecordingRouter(other_frame())
        runtime = build_intent_routing_runtime(
            settings,
            client_factory=lambda config: (_ for _ in ()).throw(
                AssertionError(f"cloud client constructed: {config}")),
            scripted_router_factory=lambda config: router)

        self.assertIs(runtime.structured_router, router)
        asyncio.run(runtime.close())

    def test_runtime_metadata_contains_only_immutable_router_contract(self):
        settings = IntentRoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
            "TREESEM_AGENT_LLM_MODE": "scripted_demo",
        })
        runtime = build_intent_routing_runtime(
            settings,
            scripted_router_factory=lambda config: RecordingRouter(
                other_frame()))

        metadata = runtime.metadata
        self.assertEqual(metadata["routing_mode"], "structured_llm")
        self.assertEqual(metadata["router_model"], "scripted_demo")
        self.assertRegex(metadata["router_prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(metadata["intent_frame_schema_version"], 1)
        self.assertRegex(
            metadata["workflow_registry_version"], r"^[0-9a-f]{64}$")
        self.assertNotIn("base_url", metadata)
        self.assertNotIn("api_key", metadata)

    def test_shadow_observes_structured_failure_but_keeps_legacy_result(self):
        router = RecordingRouter(StructuredRouterError(
            "intent_router_unavailable"))
        llm = ScriptedLlmClient([LlmTurn(content="legacy response")])
        loop = AgentLoop(
            llm, ToolRegistry(EmptyBackend()),
            structured_router=router,
            routing_mode="structured_shadow")

        response = asyncio.run(loop.run(request()))

        self.assertEqual(response.answer, "legacy response")
        self.assertEqual(router.calls, 1)

    def test_legacy_mode_never_calls_structured_router(self):
        router = RecordingRouter(other_frame())
        llm = ScriptedLlmClient([LlmTurn(content="legacy response")])
        loop = AgentLoop(
            llm, ToolRegistry(EmptyBackend()),
            structured_router=router,
            routing_mode="legacy_rule")

        response = asyncio.run(loop.run(request()))

        self.assertEqual(response.answer, "legacy response")
        self.assertEqual(router.calls, 0)


if __name__ == "__main__":
    unittest.main()
