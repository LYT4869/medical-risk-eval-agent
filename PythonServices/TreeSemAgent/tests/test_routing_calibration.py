import unittest

from agent.routing_types import RequestScope
from agent.semantic_routing import SemanticScores
from evaluation.calibrate_routing import (
    RoutingObservation,
    apply_thresholds,
    choose_thresholds,
)


class RoutingCalibrationTest(unittest.TestCase):
    def test_calibration_separates_known_from_unknown_and_compositional(self):
        observations = [
            RoutingObservation("known", RequestScope.PREDICTION,
                               SemanticScores(RequestScope.PREDICTION, 0.91, 0.40, 0.51)),
            RoutingObservation("known", RequestScope.HISTORY,
                               SemanticScores(RequestScope.HISTORY, 0.86, 0.30, 0.56)),
            RoutingObservation("unknown", RequestScope.UNKNOWN,
                               SemanticScores(RequestScope.KNOWLEDGE, 0.55, 0.20, 0.35)),
            RoutingObservation("compositional", RequestScope.UNKNOWN,
                               SemanticScores(RequestScope.COMPARISON, 0.88, 0.03, 0.85)),
        ]

        thresholds = choose_thresholds(
            observations, minimum_known_accuracy=1.0,
            minimum_fallback_recall=1.0)
        actual = [apply_thresholds(item, thresholds) for item in observations]

        self.assertEqual(actual, [
            RequestScope.PREDICTION,
            RequestScope.HISTORY,
            RequestScope.UNKNOWN,
            RequestScope.UNKNOWN,
        ])

    def test_fixed_rule_decision_is_not_changed_by_thresholds(self):
        observation = RoutingObservation(
            "known", RequestScope.HISTORY, scores=None,
            fixed_scope=RequestScope.HISTORY)
        thresholds = choose_thresholds(
            [observation], minimum_known_accuracy=1.0,
            minimum_fallback_recall=0.0)
        self.assertEqual(apply_thresholds(observation, thresholds),
                         RequestScope.HISTORY)

    def test_impossible_quality_constraints_fail_explicitly(self):
        observations = [RoutingObservation(
            "known", RequestScope.HISTORY,
            SemanticScores(RequestScope.PREDICTION, 0.9, 0.5, 0.4))]
        with self.assertRaisesRegex(ValueError, "calibration constraints"):
            choose_thresholds(
                observations, minimum_known_accuracy=1.0,
                minimum_fallback_recall=1.0)


if __name__ == "__main__":
    unittest.main()
