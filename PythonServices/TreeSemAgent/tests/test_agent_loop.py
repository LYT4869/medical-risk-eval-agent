import asyncio
import unittest

from agent.llm_client import ScriptedDemoClient, ScriptedLlmClient
from agent.loop import AgentExecutionError, AgentLoop
from agent.schemas import AgentRunRequest, LlmToolCall, LlmTurn, PredictionContext, RecentMessage
from agent.tool_registry import ToolRegistry
from agent.tools.backend import BackendToolClient, ToolExecutionError


class FakeBackend:
    def __init__(self):
        self.contexts = []
        self.calls = []

    async def predict_sample(self, context, sample_index):
        self.contexts.append(context)
        self.calls.append(("predict_sample", sample_index))
        return {"prediction_id": "pred_" + "a" * 32, "prediction": {"label": 1}}

    async def get_prediction(self, context, prediction_id):
        self.calls.append(("get_prediction", prediction_id))
        return {"prediction_id": prediction_id}

    async def get_explanation(self, context, prediction_id):
        self.calls.append(("get_explanation", prediction_id))
        return {"prediction_id": prediction_id, "important_features": [], "decision_path": []}

    async def get_history(self, context, limit, cursor):
        self.calls.append(("get_prediction_history", limit))
        return {"items": [{"prediction_id": "pred_" + "a" * 32}], "next_cursor": None}

    async def compare(self, context, prediction_id_a, prediction_id_b):
        self.calls.append(("compare_predictions", prediction_id_a, prediction_id_b))
        return {
            "prediction_a": {"prediction_id": prediction_id_a},
            "prediction_b": {"prediction_id": prediction_id_b},
            "label_changed": False,
            "model_version_changed": False,
            "positive_probability_delta": 0.0,
            "confidence_delta": 0.0,
            "cluster_changed": False,
            "tree_leaf_changed": False,
            "path_changed": False,
            "changed_features": [],
        }


class InvalidOutputBackend(FakeBackend):
    async def predict_sample(self, context, sample_index):
        return {"unexpected": "missing prediction id"}


def request():
    return AgentRunRequest(run_id="run_" + "1" * 32, session_id="ses_" + "2" * 32,
                           message="predict sample zero")


class AgentLoopTest(unittest.TestCase):
    def test_offline_demo_activates_explanation_skill_then_calls_tool(self):
        prediction_id = "pred_" + "a" * 32
        client = ScriptedDemoClient()
        messages = [{"role": "user", "content": f"解释 {prediction_id}"}]
        activate = [{"type": "function", "function": {
            "name": "activate_skill", "parameters": {}}}]
        first = asyncio.run(client.complete(messages, activate, 1.0))
        self.assertEqual(first.tool_calls[0].name, "activate_skill")
        self.assertEqual(first.tool_calls[0].arguments["skill_id"],
                         "explain_prediction")
        messages.extend([
            {"role": "assistant", "tool_calls": [{"function": {
                "name": "activate_skill"}}]},
            {"role": "tool", "content": '{"status":"activated"}'},
        ])
        explanation = [{"type": "function", "function": {
            "name": "get_explanation", "parameters": {}}}]
        second = asyncio.run(client.complete(messages, explanation, 1.0))
        self.assertEqual(second.tool_calls[0].name, "get_explanation")

    def test_tool_then_grounded_answer(self):
        prediction_id = "pred_" + "a" * 32
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0})]),
            LlmTurn(content=f"Result {prediction_id}", grounding_prediction_ids=[prediction_id]),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.step_count, 2)
        self.assertEqual(result.tools_used[0].name, "predict_sample")
        self.assertEqual(result.grounding_prediction_ids, [prediction_id])

    def test_repeated_call_stops(self):
        call = LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0})
        llm = ScriptedLlmClient([LlmTurn(tool_calls=[call]), LlmTurn(tool_calls=[call])])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))

    def test_unknown_tool_is_safe_error_and_can_recover(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="delete_database", arguments={})]),
            LlmTurn(content="That operation is unavailable."),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.tools_used[0].status, "error")

    def test_invalid_arguments_are_safe_error(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": -1})]),
            LlmTurn(content="The sample index is invalid."),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.tools_used[0].status, "error")

    def test_invalid_tool_output_is_safe_error(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0})]),
            LlmTurn(content="The prediction data is unavailable."),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(InvalidOutputBackend())).run(request()))
        self.assertEqual(result.tools_used[0].status, "error")

    def test_step_limit_stops_changing_calls(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0})]),
            LlmTurn(tool_calls=[LlmToolCall(id="c2", name="predict_sample", arguments={"sample_index": 1})]),
        ])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend()), max_steps=2).run(request()))

    def test_tool_call_limit_stops_batch(self):
        llm = ScriptedLlmClient([LlmTurn(tool_calls=[
            LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0}),
            LlmToolCall(id="c2", name="predict_sample", arguments={"sample_index": 1}),
        ])])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend()), max_tool_calls=1).run(request()))

    def test_unavailable_grounding_id_is_rejected(self):
        llm = ScriptedLlmClient([
            LlmTurn(content="A stored result was used.",
                    grounding_prediction_ids=["pred_" + "b" * 32]),
        ])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))

    def test_unavailable_id_mentioned_in_answer_is_rejected(self):
        llm = ScriptedLlmClient([LlmTurn(content="See pred_" + "b" * 32 + ".")])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))

    def test_prediction_numeric_fact_requires_grounding(self):
        llm = ScriptedLlmClient([LlmTurn(content="The probability is 20%.")])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))

    def test_general_tool_free_answer_is_allowed(self):
        llm = ScriptedLlmClient([LlmTurn(content="I can explain a stored treeSem result.")])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.tools_used, [])

    def test_history_ids_can_ground_final_answer(self):
        prediction_id = "pred_" + "a" * 32
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="c1", name="get_prediction_history", arguments={"limit": 5})]),
            LlmTurn(content=f"The history includes {prediction_id}.",
                    grounding_prediction_ids=[prediction_id]),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.grounding_prediction_ids, [prediction_id])

    def test_capability_is_bound_in_context(self):
        backend = FakeBackend()
        secured_request = request().model_copy(update={"capability_token": "signed-capability"})
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample", arguments={"sample_index": 0})]),
            LlmTurn(content="The result is available.",
                    grounding_prediction_ids=["pred_" + "a" * 32]),
        ])
        asyncio.run(AgentLoop(llm, ToolRegistry(backend)).run(secured_request))
        self.assertEqual(backend.contexts[0].capability_token, "signed-capability")
        self.assertEqual(backend.contexts[0].session_id, secured_request.session_id)

    def test_recent_messages_and_current_prediction_enter_context(self):
        enriched = request().model_copy(update={
            "recent_messages": [RecentMessage(role="assistant", content="Earlier answer")],
            "current_prediction": PredictionContext(
                prediction_id="pred_" + "a" * 32, model_version="version-1"),
        })
        llm = ScriptedLlmClient([LlmTurn(content="I can explain the current result.")])
        asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(enriched))
        messages = llm.requests[0]
        self.assertTrue(any(item.get("content") == "Earlier answer" for item in messages))
        self.assertTrue(any("Current prediction context" in item.get("content", "") for item in messages))

    def test_non_finite_tool_value_is_rejected(self):
        with self.assertRaises(ToolExecutionError):
            BackendToolClient._validate_finite({"probability": float("nan")})

    def test_empty_final_answer_is_rejected(self):
        llm = ScriptedLlmClient([LlmTurn(content="   ")])
        with self.assertRaises(AgentExecutionError):
            asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))

    def test_explanation_can_ground_answer(self):
        prediction_id = "pred_" + "a" * 32
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(
                id="c1", name="get_explanation",
                arguments={"prediction_id": prediction_id})]),
            LlmTurn(content=f"Explanation for {prediction_id}.",
                    grounding_prediction_ids=[prediction_id]),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.tools_used[0].name, "get_explanation")

    def test_comparison_returns_both_grounding_ids(self):
        first, second = "pred_" + "a" * 32, "pred_" + "b" * 32
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="compare_predictions",
                arguments={"prediction_id_a": first, "prediction_id_b": second})]),
            LlmTurn(content=f"Compared {first} and {second}.",
                    grounding_prediction_ids=[first, second]),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(set(result.grounding_prediction_ids), {first, second})

    def test_multiple_tool_calls_execute_serially(self):
        prediction_id = "pred_" + "a" * 32
        backend = FakeBackend()
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[
                LlmToolCall(id="c1", name="get_prediction",
                            arguments={"prediction_id": prediction_id}),
                LlmToolCall(id="c2", name="get_explanation",
                            arguments={"prediction_id": prediction_id}),
            ]),
            LlmTurn(content=f"Reviewed {prediction_id}.",
                    grounding_prediction_ids=[prediction_id]),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(backend)).run(request()))
        self.assertEqual([item[0] for item in backend.calls],
                         ["get_prediction", "get_explanation"])
        self.assertEqual(len(result.tools_used), 2)

    def test_extra_tool_argument_is_rejected(self):
        llm = ScriptedLlmClient([
            LlmTurn(tool_calls=[LlmToolCall(id="c1", name="predict_sample",
                arguments={"sample_index": 0, "session_id": "ses_" + "f" * 32})]),
            LlmTurn(content="The tool arguments were rejected."),
        ])
        result = asyncio.run(AgentLoop(llm, ToolRegistry(FakeBackend())).run(request()))
        self.assertEqual(result.tools_used[0].status, "error")


if __name__ == "__main__":
    unittest.main()
