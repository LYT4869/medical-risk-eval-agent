from copy import deepcopy
import unittest

from pydantic import ValidationError

from agent.intent_frame import IntentKind, TargetKind
from agent.intent_validation import BoundGoal, BoundTarget, ValidatedIntent
from agent.schemas import ComparisonToolOutput
from agent.tool_result_projection import project_tool_result


LATEST = "pred_" + "a" * 32
PREVIOUS = "pred_" + "b" * 32


def intent(target=TargetKind.LATEST_TWO_PREDICTIONS):
    goal = BoundGoal(IntentKind.COMPARISON, BoundTarget(target), (), None)
    return ValidatedIntent(None, (goal,), frozenset(), frozenset(), None)


def comparison():
    return {
        "prediction_a": {"prediction_id": LATEST, "label": 1,
                         "positive_probability": 0.82, "confidence": 0.82},
        "prediction_b": {"prediction_id": PREVIOUS, "label": 0,
                         "positive_probability": 0.34, "confidence": 0.66},
        "positive_probability_delta": -0.48, "confidence_delta": -0.16,
        "label_changed": True, "model_version_changed": False,
        "cluster_changed": False, "tree_leaf_changed": True, "path_changed": True,
        "changed_features": [{"index": 48, "name": "Intrapartum_Bleeding",
            "standardized_value_a": 2.0, "standardized_value_b": 0.5,
            "standardized_delta": -1.5, "original_value_a": 200,
            "original_value_b": 80, "original_delta": -120, "unit": None}],
    }


class ComparisonPresentationTest(unittest.TestCase):
    def project(self, content, target=TargetKind.LATEST_TWO_PREDICTIONS,
                head=(LATEST, PREVIOUS)):
        evidence = project_tool_result("compare_predictions", content, intent(target),
                                       history_head_prediction_ids=head)
        self.assertIn("comparison_order", evidence)
        return evidence

    def test_recent_comparison_is_previous_to_latest_not_backend_a_to_b(self):
        raw = comparison()
        before = deepcopy(raw)
        evidence = self.project(raw)
        self.assertEqual(evidence["comparison_order"], "previous_to_latest")
        self.assertEqual(evidence["from_prediction"]["prediction_id"], PREVIOUS)
        self.assertEqual(evidence["to_prediction"]["prediction_id"], LATEST)
        self.assertEqual(evidence["positive_probability_delta"], 0.48)
        self.assertEqual(evidence["positive_probability_direction"], "increase")
        self.assertEqual(evidence["positive_probability_delta_percentage_points"], 48.0)
        self.assertEqual(evidence["confidence_delta"], 0.16)
        self.assertNotIn("prediction_a", evidence)
        self.assertNotIn("prediction_b", evidence)
        self.assertEqual(raw, before)

    def test_probability_change_has_unambiguous_value_and_unit_semantics(self):
        change = self.project(comparison())["positive_probability_change"]
        self.assertEqual(change, {
            "metric": "positive_class_probability",
            "from": {"value": 0.34, "display_value": 34.0,
                     "display_unit": "percent"},
            "to": {"value": 0.82, "display_value": 82.0,
                   "display_unit": "percent"},
            "absolute_delta": {"value": 0.48, "display_value": 48.0,
                               "display_unit": "percentage_points"},
            "direction": "increase",
            "relative_change": "not_computed",
        })
        confidence = self.project(comparison())["confidence_change"]
        self.assertEqual(confidence["metric"], "predicted_class_confidence")
        self.assertEqual(confidence["from"], {
            "value": 0.66, "display_value": 66.0, "display_unit": "percent"})
        self.assertEqual(confidence["to"], {
            "value": 0.82, "display_value": 82.0, "display_unit": "percent"})
        self.assertEqual(confidence["absolute_delta"], {
            "value": 0.16, "display_value": 16.0,
            "display_unit": "percentage_points"})

    def test_comparison_marks_model_semantics_without_clinical_inference(self):
        evidence = self.project(comparison())
        self.assertEqual(evidence["model_class_change"], {
            "changed": True,
            "from": {"code": 0, "name": "model_negative_class"},
            "to": {"code": 1, "name": "model_positive_class"},
        })
        self.assertEqual(evidence["interpretation_scope"], {
            "probability": "model_positive_class_probability",
            "class_labels": "model_encoding_only",
            "clinical_interpretation_supported": False,
            "feature_threshold_is_clinical_reference_range": False,
            "causal_attribution_supported": False,
        })

    def test_feature_direction_and_values_follow_same_time_order(self):
        feature = self.project(comparison())["changed_features"][0]
        self.assertEqual(feature["original_value_from"], 80)
        self.assertEqual(feature["original_value_to"], 200)
        self.assertEqual(feature["original_delta"], 120)
        self.assertEqual(feature["standardized_delta"], 1.5)
        self.assertIsNone(feature["unit"])
        self.assertNotIn("original_value_a", feature)

    def test_already_time_ordered_backend_is_not_reversed_again(self):
        raw = comparison()
        raw["prediction_a"], raw["prediction_b"] = raw["prediction_b"], raw["prediction_a"]
        raw["positive_probability_delta"] = 0.48
        raw["confidence_delta"] = 0.16
        raw["changed_features"] = []
        evidence = self.project(raw)
        self.assertEqual(evidence["from_prediction"]["prediction_id"], PREVIOUS)
        self.assertEqual(evidence["to_prediction"]["prediction_id"], LATEST)
        self.assertEqual(evidence["positive_probability_delta"], 0.48)

    def test_decrease_and_no_change_are_not_always_increase(self):
        for latest_probability, backend_delta, expected_delta, direction in (
                (0.12, 0.22, -0.22, "decrease"),
                (0.34, 0.0, 0.0, "unchanged")):
            with self.subTest(direction=direction):
                raw = comparison()
                raw["prediction_a"]["positive_probability"] = latest_probability
                raw["positive_probability_delta"] = backend_delta
                evidence = self.project(raw)
                self.assertEqual(evidence["positive_probability_delta"], expected_delta)
                self.assertEqual(evidence["positive_probability_direction"], direction)
                change = evidence["positive_probability_change"]
                self.assertEqual(change["absolute_delta"]["value"],
                                 abs(expected_delta))
                self.assertEqual(change["absolute_delta"]["display_value"],
                                 abs(expected_delta) * 100)

    def test_comparison_contract_rejects_invalid_optional_score_values(self):
        for invalid in (None, "0.82", float("nan"), -0.1, 1.1):
            with self.subTest(invalid=invalid):
                raw = comparison()
                raw["prediction_a"]["positive_probability"] = invalid
                with self.assertRaises(ValidationError):
                    ComparisonToolOutput.model_validate(raw)

    def test_comparison_contract_rejects_invalid_or_inconsistent_deltas(self):
        for field in ("positive_probability_delta", "confidence_delta"):
            for invalid in (float("nan"), float("inf"), -1.1, 1.1):
                with self.subTest(field=field, invalid=invalid):
                    raw = comparison()
                    raw[field] = invalid
                    with self.assertRaises(ValidationError):
                        ComparisonToolOutput.model_validate(raw)

        raw = comparison()
        raw["positive_probability_delta"] = -0.47
        with self.assertRaises(ValidationError):
            ComparisonToolOutput.model_validate(raw)

    def test_explicit_pair_does_not_acquire_unrequested_time_semantics(self):
        evidence = self.project(comparison(), TargetKind.EXPLICIT_PREDICTION_PAIR)
        self.assertEqual(evidence["comparison_order"], "requested_a_to_b")
        self.assertEqual(evidence["from_prediction"]["prediction_id"], LATEST)
        self.assertEqual(evidence["positive_probability_delta"], -0.48)

    def test_unknown_or_unrelated_history_does_not_invent_latest_previous(self):
        for head in ((), (LATEST,), ("other-a", "other-b")):
            with self.subTest(head=head):
                evidence = self.project(comparison(), head=head)
                self.assertEqual(evidence["comparison_order"], "requested_a_to_b")
                self.assertEqual(evidence["positive_probability_delta"], -0.48)

    def test_nullable_feature_values_and_deltas_stay_unknown(self):
        raw = comparison()
        raw["changed_features"] = [{"name": "unknown", "original_value_a": None,
                                   "original_value_b": 5, "original_delta": None}]
        feature = self.project(raw)["changed_features"][0]
        self.assertEqual(feature["original_value_from"], 5)
        self.assertIsNone(feature["original_value_to"])
        self.assertIsNone(feature["original_delta"])
