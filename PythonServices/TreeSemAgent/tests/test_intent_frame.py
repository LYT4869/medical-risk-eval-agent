import unittest

from pydantic import ValidationError

from agent.intent_frame import IntentFrame, IntentKind, TargetKind


def explanation_frame() -> dict:
    return {
        "schema_version": 1,
        "goals": [{
            "intent": "explanation",
            "target": {
                "type": "previous_prediction",
                "explicit_reference_index": None,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": ["decision_path"],
            "knowledge_scope": None,
            "evidence": ["上次的决策树"],
        }],
        "constraints": {
            "excluded_intents": [],
            "excluded_aspects": [],
        },
        "unresolved_references": [],
        "needs_clarification": False,
    }


class IntentFrameTest(unittest.TestCase):
    def test_accepts_minimal_explanation_frame(self):
        frame = IntentFrame.model_validate(explanation_frame())

        self.assertEqual(frame.schema_version, 1)
        self.assertEqual(frame.goals[0].intent, IntentKind.EXPLANATION)
        self.assertEqual(
            frame.goals[0].target.type, TargetKind.PREVIOUS_PREDICTION)

    def test_forbids_unknown_fields_at_every_level(self):
        payload = explanation_frame()
        payload["goals"][0]["unexpected"] = True

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_rejects_duplicate_requested_aspects(self):
        payload = explanation_frame()
        payload["goals"][0]["requested_aspects"] = [
            "decision_path", "decision_path"]

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_rejects_more_than_three_goals(self):
        payload = explanation_frame()
        payload["goals"] = payload["goals"] * 4

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_rejects_oversized_evidence(self):
        payload = explanation_frame()
        payload["goals"][0]["evidence"] = ["x" * 161]

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_explicit_pair_requires_two_distinct_candidate_indexes(self):
        payload = explanation_frame()
        payload["goals"][0]["target"] = {
            "type": "explicit_prediction_pair",
            "explicit_reference_index": 0,
            "second_explicit_reference_index": 0,
            "sample_reference_index": None,
        }

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_demo_sample_requires_sample_candidate_index(self):
        payload = explanation_frame()
        payload["goals"][0]["intent"] = "prediction"
        payload["goals"][0]["target"] = {
            "type": "demo_sample",
            "explicit_reference_index": None,
            "second_explicit_reference_index": None,
            "sample_reference_index": None,
        }

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_symbolic_target_rejects_candidate_indexes(self):
        payload = explanation_frame()
        payload["goals"][0]["target"]["explicit_reference_index"] = 0

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
