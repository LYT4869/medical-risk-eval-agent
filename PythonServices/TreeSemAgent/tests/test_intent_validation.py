from __future__ import annotations

import unittest

from agent.intent_frame import IntentFrame, IntentKind, TargetKind
from agent.intent_validation import (
    IntentFrameViolation,
    validate_and_bind_intent,
)
from agent.reference_extractor import extract_references
from agent.schemas import AgentRunRequest, PredictionContext


def request(message: str, *, current=True) -> AgentRunRequest:
    prediction = None
    if current:
        prediction = PredictionContext(
            prediction_id="pred_" + "c" * 32,
            model_version="model-v1")
    return AgentRunRequest(
        run_id="run_" + "1" * 32,
        session_id="ses_" + "2" * 32,
        message=message,
        current_prediction=prediction,
    )


def frame(*, intent="explanation", target="current_prediction",
          aspects=None, evidence=None, excluded_intents=None,
          excluded_aspects=None, unresolved=None, clarification=False,
          explicit_index=None, second_index=None, sample_index=None,
          knowledge_scope=None) -> IntentFrame:
    return IntentFrame.model_validate({
        "schema_version": 1,
        "goals": [{
            "intent": intent,
            "target": {
                "type": target,
                "explicit_reference_index": explicit_index,
                "second_explicit_reference_index": second_index,
                "sample_reference_index": sample_index,
            },
            "requested_aspects": aspects or [],
            "knowledge_scope": knowledge_scope,
            "evidence": evidence or ["解释"],
        }],
        "constraints": {
            "excluded_intents": excluded_intents or [],
            "excluded_aspects": excluded_aspects or [],
        },
        "unresolved_references": unresolved or [],
        "needs_clarification": clarification,
    })


class IntentValidationTest(unittest.TestCase):
    def validate(self, value: IntentFrame, user_request: AgentRunRequest):
        references = extract_references(user_request.message)
        return validate_and_bind_intent(value, references, user_request)

    def test_binds_current_prediction_from_trusted_request_context(self):
        user_request = request("解释一下当前结果")

        result = self.validate(frame(), user_request)

        self.assertIsNotNone(result.validated)
        target = result.validated.goals[0].target
        self.assertEqual(target.kind, TargetKind.CURRENT_PREDICTION)
        self.assertEqual(
            target.prediction_ids, (user_request.current_prediction.prediction_id,))

    def test_binds_explicit_prediction_by_source_candidate_index(self):
        first = "pred_" + "a" * 32
        second = "pred_" + "b" * 32
        user_request = request(f"解释 {first}，不要看 {second}")
        value = frame(
            target="explicit_prediction", explicit_index=0,
            evidence=["<prediction_ref_0>"])

        result = self.validate(value, user_request)

        self.assertEqual(
            result.validated.goals[0].target.prediction_ids, (first,))

    def test_binds_explicit_pair_without_accepting_generated_ids(self):
        first = "pred_" + "a" * 32
        second = "pred_" + "b" * 32
        user_request = request(f"比较 {first} 和 {second}")
        value = frame(
            intent="comparison", target="explicit_prediction_pair",
            aspects=["comparison_changes"],
            evidence=["比较", "<prediction_ref_0>", "<prediction_ref_1>"],
            explicit_index=0, second_index=1)

        result = self.validate(value, user_request)

        self.assertEqual(
            result.validated.goals[0].target.prediction_ids, (first, second))

    def test_binds_demo_sample_by_labelled_source_candidate(self):
        user_request = request("请运行样本索引 12")
        value = frame(
            intent="prediction", target="demo_sample",
            evidence=["运行", "<sample_ref_0>"], sample_index=0)

        result = self.validate(value, user_request)

        self.assertEqual(result.validated.goals[0].target.sample_index, 12)

    def test_rejects_candidate_index_outside_extracted_values(self):
        user_request = request("解释当前结果")
        value = frame(
            target="explicit_prediction", explicit_index=0,
            evidence=["解释"])

        with self.assertRaises(IntentFrameViolation) as caught:
            self.validate(value, user_request)

        self.assertEqual(caught.exception.code, "invalid_candidate_index")

    def test_rejects_fabricated_evidence(self):
        with self.assertRaises(IntentFrameViolation) as caught:
            self.validate(frame(evidence=["上一次"]), request("解释当前结果"))

        self.assertEqual(caught.exception.code, "fabricated_evidence")

    def test_evidence_validation_normalizes_width_case_and_whitespace(self):
        result = self.validate(
            frame(evidence=["EXPLAIN RESULT"]),
            request("ｅｘｐｌａｉｎ   result"))

        self.assertIsNotNone(result.validated)

    def test_positive_goal_excluded_by_constraint_requires_clarification(self):
        result = self.validate(
            frame(excluded_intents=["explanation"]),
            request("解释当前结果，但又不要解释"))

        self.assertIsNone(result.validated)
        self.assertEqual(result.clarification_code, "conflicting_request")

    def test_rejects_incompatible_intent_and_target(self):
        with self.assertRaises(IntentFrameViolation) as caught:
            self.validate(
                frame(intent="knowledge", target="current_prediction",
                      evidence=["资料"], knowledge_scope="clinical"),
                request("查询资料"))

        self.assertEqual(caught.exception.code, "incompatible_target")

    def test_rejects_incompatible_requested_aspect(self):
        with self.assertRaises(IntentFrameViolation) as caught:
            self.validate(
                frame(intent="knowledge", target="general_knowledge",
                      aspects=["decision_path"], evidence=["资料"],
                      knowledge_scope="clinical"),
                request("查询资料"))

        self.assertEqual(caught.exception.code, "incompatible_aspect")

    def test_missing_current_prediction_is_a_clarification(self):
        result = self.validate(frame(), request("解释当前结果", current=False))

        self.assertIsNone(result.validated)
        self.assertEqual(
            result.clarification_code, "current_prediction_missing")

    def test_previous_and_latest_two_targets_remain_symbolic(self):
        previous = self.validate(
            frame(target="previous_prediction", evidence=["上次"]),
            request("解释上次结果"))
        latest = self.validate(
            frame(intent="comparison", target="latest_two_predictions",
                  aspects=["comparison_changes"], evidence=["最近两次"]),
            request("比较最近两次"))

        self.assertEqual(previous.validated.goals[0].target.prediction_ids, ())
        self.assertEqual(latest.validated.goals[0].target.prediction_ids, ())

    def test_unresolved_reference_uses_deterministic_clarification(self):
        result = self.validate(
            frame(target="none", evidence=["那个"],
                  unresolved=["ambiguous_reference"], clarification=True),
            request("解释那个", current=False))

        self.assertIsNone(result.validated)
        self.assertEqual(result.clarification_code, "ambiguous_reference")

    def test_clarification_flag_must_match_unresolved_references(self):
        with self.assertRaises(IntentFrameViolation) as caught:
            self.validate(
                frame(clarification=True), request("解释当前结果"))

        self.assertEqual(caught.exception.code, "invalid_clarification_state")

    def test_preserves_validated_constraints_and_multiple_goals(self):
        payload = frame(
            excluded_aspects=["prediction_summary"]).model_dump(mode="json")
        payload["goals"].append({
            "intent": "summary",
            "target": {
                "type": "current_prediction",
                "explicit_reference_index": None,
                "second_explicit_reference_index": None,
                "sample_reference_index": None,
            },
            "requested_aspects": ["probability"],
            "knowledge_scope": None,
            "evidence": ["概率"],
        })
        value = IntentFrame.model_validate(payload)

        result = self.validate(value, request("解释当前结果并告诉我概率"))

        self.assertEqual(len(result.validated.goals), 2)
        self.assertIn(
            IntentKind.EXPLANATION,
            {goal.intent for goal in result.validated.goals})


if __name__ == "__main__":
    unittest.main()
