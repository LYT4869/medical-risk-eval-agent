import unittest

from agent.execution_state import ExecutionState
from agent.intent_frame import IntentKind, TargetKind
from agent.intent_validation import BoundGoal, BoundTarget, ValidatedIntent
from agent.schemas import ToolUse


class ExecutionStateTest(unittest.TestCase):
    def state(self):
        goal = BoundGoal(IntentKind.EXPLANATION,
                         BoundTarget(TargetKind.PREVIOUS_PREDICTION), (), None)
        intent = ValidatedIntent(None, (goal,), frozenset(), frozenset(), None)
        return ExecutionState(intent, 5, 8)

    def record(self, state, name, arguments, content):
        state.record(name, arguments, content,
                     ToolUse(name=name, status="success", duration_ms=0), content)

    def test_relative_target_is_not_bound_from_paginated_history(self):
        state = self.state()
        self.record(state, "get_prediction_history", {"cursor": "older-page"},
                    {"items": [{"prediction_id": "old-a"},
                               {"prediction_id": "old-b"}]})
        self.record(state, "get_explanation", {"prediction_id": "old-b"},
                    {"prediction_id": "old-b"})
        self.assertFalse(state.all_goals_completed)

    def test_head_history_binds_previous_and_state_is_request_local(self):
        state, other = self.state(), self.state()
        self.record(state, "get_prediction_history", {"limit": 2},
                    {"items": [{"prediction_id": "current"},
                               {"prediction_id": "previous"}]})
        self.record(state, "get_explanation", {"prediction_id": "current"},
                    {"prediction_id": "current"})
        self.assertFalse(state.all_goals_completed)
        self.record(state, "get_explanation", {"prediction_id": "previous"},
                    {"prediction_id": "previous"})
        self.assertTrue(state.all_goals_completed)
        self.assertEqual(other.evidence, [])
        self.assertEqual(other.completed_goal_indexes, [])

    def test_prediction_completion_requires_requested_sample(self):
        goal = BoundGoal(IntentKind.PREDICTION,
                         BoundTarget(TargetKind.DEMO_SAMPLE, sample_index=7), (), None)
        intent = ValidatedIntent(None, (goal,), frozenset(), frozenset(), None)
        state = ExecutionState(intent, 5, 8)
        self.record(state, "predict_sample", {"sample_index": 8},
                    {"prediction_id": "different-sample"})
        self.assertFalse(state.all_goals_completed)
        self.record(state, "predict_sample", {"sample_index": 7},
                    {"prediction_id": "requested-sample"})
        self.assertTrue(state.all_goals_completed)
