from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.llm_client import (LlmError, LlmToolPolicy, OpenAiCompatibleClient,
                              OpenAiCompatibleConfig,
                              optional_boolean_environment)


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict):
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.requests: list[dict] = []

    async def post(self, path: str, **kwargs):
        self.requests.append({"path": path, **kwargs})
        return self.response

    async def aclose(self) -> None:
        return None


class _SequenceAsyncClient(_FakeAsyncClient):
    def __init__(self, responses):
        super().__init__(responses[-1])
        self.responses = list(responses)

    async def post(self, path: str, **kwargs):
        self.requests.append({"path": path, **kwargs})
        return self.responses.pop(0)


class OpenAiCompatibleClientTest(unittest.TestCase):
    @staticmethod
    def fake_httpx(client: _FakeAsyncClient):
        return SimpleNamespace(
            AsyncClient=lambda **_kwargs: client,
            Timeout=lambda *_args, **_kwargs: None,
            Limits=lambda **_kwargs: None,
            HTTPError=RuntimeError,
        )

    def test_llm_error_rejects_unknown_error_code(self):
        with self.assertRaisesRegex(ValueError, "unknown LLM error code"):
            LlmError("unsafe upstream detail", code="provider_account_123")

    def test_real_request_has_deterministic_output_limits(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                      "total_tokens": 12},
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model",
                temperature=0.0, max_output_tokens=256))
        asyncio.run(client.complete(
            [{"role": "user", "content": "hello"}], [], 5.0))

        payload = fake.requests[0]["json"]
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertNotIn("enable_thinking", payload)

    def test_real_request_can_explicitly_disable_thinking(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model",
                enable_thinking=False))

        asyncio.run(client.complete([], [], 5.0))

        self.assertFalse(fake.requests[0]["json"]["enable_thinking"])

    def test_optional_boolean_environment_validates_explicit_value(self):
        with patch.dict("os.environ", {
                "TREESEM_TEST_ENABLE_THINKING": "false"}, clear=False):
            self.assertFalse(optional_boolean_environment(
                "TREESEM_TEST_ENABLE_THINKING"))
        with patch.dict("os.environ", {
                "TREESEM_TEST_ENABLE_THINKING": "invalid"}, clear=False):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                optional_boolean_environment("TREESEM_TEST_ENABLE_THINKING")

    def test_real_request_disables_parallel_tool_calls(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": None, "tool_calls": []}}],
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))
        asyncio.run(client.complete([], [{
            "type": "function",
            "function": {"name": "activate_skill", "parameters": {}},
        }], 5.0))

        payload = fake.requests[0]["json"]
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["parallel_tool_calls"])

    def test_required_tool_choice_selects_one_named_function(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": None, "tool_calls": []}}],
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))
        definition = {
            "type": "function",
            "function": {"name": "get_prediction", "parameters": {}},
        }

        asyncio.run(client.complete(
            [], [definition], 5.0,
            LlmToolPolicy.required("get_prediction")))

        payload = fake.requests[0]["json"]
        self.assertEqual(payload["tool_choice"], {
            "type": "function",
            "function": {"name": "get_prediction"},
        })
        self.assertFalse(payload["parallel_tool_calls"])

    def test_none_tool_choice_is_explicit_during_finalization(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": "done"}}],
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))

        asyncio.run(client.complete(
            [], [], 5.0, LlmToolPolicy.none()))

        payload = fake.requests[0]["json"]
        self.assertEqual(payload["tool_choice"], "none")
        self.assertNotIn("tools", payload)

    def test_required_policy_rejects_missing_definition(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": "unused"}}],
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))

        with self.assertRaisesRegex(ValueError,
                                    "required Tool is not defined"):
            asyncio.run(client.complete(
                [], [], 5.0,
                LlmToolPolicy.required("get_prediction")))
        self.assertEqual(fake.requests, [])

    def test_parses_fenced_structured_answer_and_usage(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": (
                "```json\n{\"answer\":\"done\","
                "\"grounding_prediction_ids\":[\"pred_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"],"
                "\"grounding_source_ids\":[]}\n```")}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120},
        }))
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))
        turn = asyncio.run(client.complete([], [], 5.0))

        self.assertEqual(turn.content, "done")
        self.assertEqual(
            turn.grounding_prediction_ids,
            ["pred_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"])
        self.assertIsNotNone(turn.usage)
        self.assertEqual(turn.usage.total_tokens, 120)
        self.assertEqual(client.usage_snapshot(), {
            "request_count": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        })

    def test_single_attempt_client_does_not_retry_retryable_status(self):
        first = _FakeResponse({})
        first.status_code = 500
        second = _FakeResponse({
            "choices": [{"message": {"content": "unused"}}],
        })
        fake = _SequenceAsyncClient([first, second])
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model",
                maximum_attempts=1))

        with self.assertRaises(LlmError) as caught:
            asyncio.run(client.complete([], [], 5.0))

        self.assertTrue(caught.exception.retryable)
        self.assertEqual(len(fake.requests), 1)

    def test_default_client_retries_one_retryable_status(self):
        first = _FakeResponse({})
        first.status_code = 500
        second = _FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
        })
        fake = _SequenceAsyncClient([first, second])
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))

        result = asyncio.run(client.complete([], [], 5.0))

        self.assertEqual(result.content, "ok")
        self.assertEqual(len(fake.requests), 2)

    def test_non_retryable_4xx_is_not_retried(self):
        first = _FakeResponse({})
        first.status_code = 401
        second = _FakeResponse({
            "choices": [{"message": {"content": "unused"}}],
        })
        fake = _SequenceAsyncClient([first, second])
        with patch("agent.llm_client.httpx", self.fake_httpx(fake)):
            client = OpenAiCompatibleClient(OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model"))

        with self.assertRaises(LlmError) as caught:
            asyncio.run(client.complete([], [], 5.0))

        self.assertFalse(caught.exception.retryable)
        self.assertEqual(caught.exception.code, "upstream_rejected")
        self.assertEqual(len(fake.requests), 1)

    def test_rejects_invalid_maximum_attempts(self):
        with self.assertRaisesRegex(ValueError, "maximum_attempts"):
            OpenAiCompatibleConfig(
                base_url="https://example.invalid/v1", model="model",
                maximum_attempts=3)


if __name__ == "__main__":
    unittest.main()
