from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from agent.llm_client import OpenAiCompatibleClient, OpenAiCompatibleConfig


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


class OpenAiCompatibleClientTest(unittest.TestCase):
    def test_real_request_has_deterministic_output_limits(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                      "total_tokens": 12},
        }))
        with patch("agent.llm_client.httpx.AsyncClient", return_value=fake):
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

    def test_parses_fenced_structured_answer_and_usage(self):
        fake = _FakeAsyncClient(_FakeResponse({
            "choices": [{"message": {"content": (
                "```json\n{\"answer\":\"done\","
                "\"grounding_prediction_ids\":[\"pred_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"],"
                "\"grounding_source_ids\":[]}\n```")}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20,
                      "total_tokens": 120},
        }))
        with patch("agent.llm_client.httpx.AsyncClient", return_value=fake):
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


if __name__ == "__main__":
    unittest.main()
