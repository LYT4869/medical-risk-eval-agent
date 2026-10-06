import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import yaml

from agent.routing_types import RequestScope
from agent.task_registry import TaskRegistry, TaskRegistryError


TOOLS = {
    "predict_sample", "get_prediction", "get_explanation",
    "get_prediction_history", "compare_predictions",
    "search_medical_knowledge",
}


def valid_payload() -> dict:
    definitions = {
        "prediction": (["predict_sample"], ["predict_sample"]),
        "summary": (["get_prediction"], ["get_prediction"]),
        "explanation": (["get_prediction", "get_explanation"],
                        ["get_explanation"]),
        "history": (["get_prediction_history"], ["get_prediction_history"]),
        "comparison": (["get_prediction_history", "compare_predictions"],
                       ["get_prediction_history", "compare_predictions"]),
        "knowledge": (["search_medical_knowledge"],
                      ["search_medical_knowledge"]),
    }
    return {
        "schema_version": 1,
        "tasks": [
            {
                "scope": scope,
                "mode": "deterministic",
                "rule_terms": [f"rule-{scope}"],
                "intent_examples": [f"example-{scope}"],
                "allowed_tools": allowed,
                "workflow_stages": stages,
            }
            for scope, (allowed, stages) in definitions.items()
        ],
    }


def load_payload(payload: dict) -> TaskRegistry:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tasks.yaml"
        path.write_text(yaml.safe_dump(payload, allow_unicode=True),
                        encoding="utf-8")
        return TaskRegistry.load(path, TOOLS)


class TaskRegistryTest(unittest.TestCase):
    def test_valid_registry_is_immutable_and_addressable_by_scope(self):
        registry = load_payload(valid_payload())

        definition = registry.definition(RequestScope.COMPARISON)
        self.assertEqual(definition.workflow_stages,
                         ("get_prediction_history", "compare_predictions"))
        self.assertEqual(definition.allowed_tools,
                         frozenset({"get_prediction_history", "compare_predictions"}))
        self.assertEqual(len(registry.business_definitions), 6)

    def test_rejects_duplicate_scope(self):
        payload = valid_payload()
        payload["tasks"].append(deepcopy(payload["tasks"][0]))
        with self.assertRaisesRegex(TaskRegistryError, "duplicate scope"):
            load_payload(payload)

    def test_rejects_unknown_task_key(self):
        payload = valid_payload()
        payload["tasks"][0]["network_url"] = "https://example.invalid"
        with self.assertRaisesRegex(TaskRegistryError, "unknown task fields"):
            load_payload(payload)

    def test_rejects_unknown_tool(self):
        payload = valid_payload()
        payload["tasks"][0]["allowed_tools"] = ["shell"]
        payload["tasks"][0]["workflow_stages"] = ["shell"]
        with self.assertRaisesRegex(TaskRegistryError, "unknown tool"):
            load_payload(payload)

    def test_rejects_unsupported_mode(self):
        payload = valid_payload()
        payload["tasks"][0]["mode"] = "dag"
        with self.assertRaisesRegex(TaskRegistryError, "unsupported mode"):
            load_payload(payload)

    def test_rejects_missing_semantic_examples(self):
        payload = valid_payload()
        payload["tasks"][0]["intent_examples"] = []
        with self.assertRaisesRegex(TaskRegistryError, "intent_examples"):
            load_payload(payload)

    def test_rejects_safety_scope(self):
        payload = valid_payload()
        payload["tasks"][0]["scope"] = "security_abuse"
        with self.assertRaisesRegex(TaskRegistryError, "business scope"):
            load_payload(payload)

    def test_rejects_stage_outside_allowed_tools(self):
        payload = valid_payload()
        payload["tasks"][0]["workflow_stages"] = ["get_prediction"]
        with self.assertRaisesRegex(TaskRegistryError, "outside allowed_tools"):
            load_payload(payload)


if __name__ == "__main__":
    unittest.main()
