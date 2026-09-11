import unittest

from pydantic import ValidationError

from agent.intent_frame import IntentFrame, IntentKind, TargetKind


def explanation_frame() -> dict:
    return {
        "schema_version": 2,
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
        "requested_skill": None,
    }


class IntentFrameTest(unittest.TestCase):
    def test_accepts_minimal_explanation_frame(self):
        frame = IntentFrame.model_validate(explanation_frame())

        self.assertEqual(frame.schema_version, 2)
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

    def test_allows_empty_goals_only_for_an_unresolved_clarification(self):
        payload = explanation_frame()
        payload["goals"] = []
        payload["unresolved_references"] = ["missing_prediction_target"]
        payload["needs_clarification"] = True

        frame = IntentFrame.model_validate(payload)

        self.assertEqual(frame.goals, [])

        payload["unresolved_references"] = []
        payload["needs_clarification"] = False
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

    def test_accepts_a_trusted_skill_as_an_execution_preference(self):
        payload = explanation_frame()
        payload["requested_skill"] = "explain_prediction"

        frame = IntentFrame.model_validate(payload)

        self.assertEqual(frame.requested_skill.value, "explain_prediction")
        self.assertEqual(frame.goals[0].intent, IntentKind.EXPLANATION)

    def test_rejects_skill_as_a_business_intent(self):
        payload = explanation_frame()
        payload["goals"][0]["intent"] = "skill"

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)

    def test_rejects_an_untrusted_requested_skill(self):
        payload = explanation_frame()
        payload["requested_skill"] = "download_remote_plugin"

        with self.assertRaises(ValidationError):
            IntentFrame.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
