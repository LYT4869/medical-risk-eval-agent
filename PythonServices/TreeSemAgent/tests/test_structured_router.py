from __future__ import annotations

import asyncio
import json
import unittest

from agent.llm_client import LlmError, LlmToolCall, LlmTurn, LlmUsage
from agent.reference_extractor import extract_references
from agent.schemas import RecentMessage
from agent.structured_router import (
    RouterContext,
    StructuredIntentRouter,
    StructuredRouterConfig,
    StructuredRouterError,
)
from agent.intent_router_prompt import ROUTER_SYSTEM_PROMPT


def frame_arguments(**updates) -> dict:
    payload = {
        "schema_version": 2,
        "goals": [{
            "intent": "explanation",
            "target": {
                "type": "explicit_prediction",
                "explicit_reference_index": 0,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": ["decision_path"],
            "knowledge_scope": None,
            "evidence": ["<prediction_ref_0>"],
        }],
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": [],
        },
        "unresolved_references": [],
        "needs_clarification": False,
        "requested_skill": None,
    }
    payload.update(updates)
    return payload


def route_turn(arguments: dict | None = None, *, name="route_user_request"):
    return LlmTurn(
        tool_calls=[LlmToolCall(
            id="route_1", name=name,
            arguments=arguments if arguments is not None else frame_arguments())],
        usage=LlmUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )


class RecordingLlm:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    async def complete(self, messages, tools, timeout, tool_policy):
        self.requests.append({
            "messages": messages,
            "tools": tools,
            "timeout": timeout,
            "tool_policy": tool_policy,
        })
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


async def no_sleep(_seconds):
    return None


class StructuredRouterTest(unittest.TestCase):
    @staticmethod
    def context() -> RouterContext:
        prediction_id = "pred_" + "a" * 32
        references = extract_references(f"解释 {prediction_id} 的决策路径")
        return RouterContext(
            message=references.router_message,
            recent_messages=(
                RecentMessage(
                    role="assistant",
                    content="上一次是 pred_" + "b" * 32),
                RecentMessage(role="user", content="刚才那个"),
            ),
            current_prediction_available=True,
            references=references,
        )

    @staticmethod
    def router(client, **updates):
        values = {
            "request_timeout_seconds": 3.0,
            "total_deadline_seconds": 7.0,
            "maximum_attempts": 2,
            "retry_backoff_seconds": 0.1,
            "context_messages": 4,
            "context_max_chars": 6000,
        }
        values.update(updates)
        return StructuredIntentRouter(
            client, StructuredRouterConfig(**values), sleep=no_sleep)

    def test_uses_one_required_side_effect_free_function(self):
        client = RecordingLlm([route_turn()])

        result = asyncio.run(self.router(client).route(self.context()))

        request = client.requests[0]
        self.assertEqual(request["tool_policy"].mode, "required")
        self.assertEqual(
            request["tool_policy"].required_tool, "route_user_request")
        self.assertEqual(len(request["tools"]), 1)
        self.assertEqual(
            request["tools"][0]["function"]["name"], "route_user_request")
        self.assertEqual(result.attempt_count, 1)
        self.assertFalse(result.repaired)

    def test_router_context_hides_business_ids_and_credentials(self):
        client = RecordingLlm([route_turn()])

        asyncio.run(self.router(client).route(self.context()))

        serialized = json.dumps(
            client.requests[0]["messages"], ensure_ascii=False)
        self.assertNotIn("pred_" + "a" * 32, serialized)
        self.assertNotIn("pred_" + "b" * 32, serialized)
        self.assertIn("<prediction_ref_0>", serialized)
        self.assertIn("current_prediction_available", serialized)
        self.assertNotIn("current_prediction\": {", serialized)

    def test_context_keeps_only_four_recent_messages_within_budget(self):
        client = RecordingLlm([route_turn()])
        context = self.context()
        context = RouterContext(
            message=context.message,
            recent_messages=tuple(
                RecentMessage(role="user", content=str(index) * 40)
                for index in range(6)),
            current_prediction_available=True,
            references=context.references,
        )

        asyncio.run(self.router(
            client, context_messages=4, context_max_chars=100).route(context))

        payload = json.loads(client.requests[0]["messages"][1]["content"])
        recent = payload["recent_final_messages"]
        content = json.dumps(recent, ensure_ascii=False)
        self.assertNotIn("0" * 40, content)
        self.assertNotIn("1" * 40, content)
        self.assertLessEqual(len(recent), 4)
        self.assertLessEqual(
            sum(len(item["content"]) for item in recent), 100)

    def test_retries_one_retryable_transport_failure(self):
        client = RecordingLlm([
            LlmError("temporary", code="upstream_unavailable", retryable=True),
            route_turn(),
        ])

        result = asyncio.run(self.router(client).route(self.context()))

        self.assertEqual(result.attempt_count, 2)
        self.assertEqual(len(client.requests), 2)
        self.assertFalse(result.repaired)

    def test_repairs_one_invalid_frame(self):
        invalid = frame_arguments()
        invalid.pop("constraints")
        client = RecordingLlm([route_turn(invalid), route_turn()])

        result = asyncio.run(self.router(client).route(self.context()))

        self.assertEqual(result.attempt_count, 2)
        self.assertTrue(result.repaired)
        repair = client.requests[1]["messages"][-1]["content"]
        self.assertIn('"reason":"missing_required_field"', repair)
        self.assertIn('"paths":["constraints"]', repair)

    def test_protocol_repair_reports_only_a_safe_error_category(self):
        client = RecordingLlm([
            LlmTurn(content="sensitive malformed router output"),
            route_turn(),
        ])

        result = asyncio.run(self.router(client).route(self.context()))

        self.assertTrue(result.repaired)
        repair = client.requests[1]["messages"][-1]["content"]
        self.assertIn('"reason":"wrong_tool_call_count"', repair)
        self.assertIn('"paths":[]', repair)
        self.assertNotIn("sensitive malformed router output", repair)

    def test_transport_retry_and_repair_share_two_attempt_budget(self):
        client = RecordingLlm([
            LlmError("temporary", code="upstream_unavailable", retryable=True),
            route_turn(frame_arguments(schema_version=1)),
            route_turn(),
        ])

        with self.assertRaises(StructuredRouterError) as caught:
            asyncio.run(self.router(client).route(self.context()))

        self.assertEqual(caught.exception.code, "invalid_intent_frame")
        self.assertEqual(len(client.requests), 2)

    def test_wrong_or_multiple_function_calls_are_protocol_failures(self):
        multiple = LlmTurn(tool_calls=[
            LlmToolCall(id="a", name="route_user_request",
                        arguments=frame_arguments()),
            LlmToolCall(id="b", name="route_user_request",
                        arguments=frame_arguments()),
        ])
        cases = [
            route_turn(name="get_prediction"),
            multiple,
            LlmTurn(content="{}"),
        ]
        for first in cases:
            with self.subTest(first=first):
                client = RecordingLlm([first, first])
                with self.assertRaises(StructuredRouterError) as caught:
                    asyncio.run(self.router(client).route(self.context()))
                self.assertEqual(caught.exception.code, "invalid_intent_frame")
                self.assertEqual(len(client.requests), 2)

    def test_non_retryable_upstream_rejection_fails_without_second_call(self):
        client = RecordingLlm([
            LlmError("unauthorized", code="upstream_rejected", retryable=False),
            route_turn(),
        ])

        with self.assertRaises(StructuredRouterError) as caught:
            asyncio.run(self.router(client).route(self.context()))

        self.assertEqual(caught.exception.code, "intent_router_unavailable")
        self.assertEqual(len(client.requests), 1)

    def test_rejects_an_impossible_retry_budget(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            StructuredRouterConfig(
                request_timeout_seconds=3.0,
                total_deadline_seconds=6.0,
                maximum_attempts=2,
                retry_backoff_seconds=0.1,
                context_messages=4,
                context_max_chars=6000,
            )

    def test_short_prompt_defines_business_intent_contract(self):
        prompt = ROUTER_SYSTEM_PROMPT
        for required in (
                "knowledge_scope", "general_knowledge",
                "summary", "model_version", "requested_skill",
                "other", "needs_clarification", "missing_sample_index",
                "prediction_summary", "decision_path", "history_items"):
            self.assertIn(required, prompt)

    def test_short_prompt_separates_skill_preference_from_business_goal(self):
        prompt = ROUTER_SYSTEM_PROMPT
        for required in (
                "explain_prediction",
                "compare_prediction_history",
                "pph_evidence_education",
                "system usage -> other + none",
                "history + session_history; knowledge + general_knowledge",
                "model metrics are knowledge"):
            self.assertIn(required, prompt)
        self.assertNotIn("skill is only", prompt)
        self.assertNotIn("citations when", prompt)


if __name__ == "__main__":
    unittest.main()
