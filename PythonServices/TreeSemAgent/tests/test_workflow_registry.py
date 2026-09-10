import re
import unittest

from agent.intent_frame import IntentKind, TargetKind
from agent.workflow_registry import (
    ArgumentSource,
    GoalPattern,
    RendererKind,
    WorkflowRecipe,
    WorkflowRegistry,
    WorkflowRegistryError,
    WorkflowStage,
    default_workflow_registry,
)


class WorkflowRegistryTest(unittest.TestCase):
    def test_default_registry_contains_the_version_one_recipes(self):
        registry = default_workflow_registry()

        self.assertEqual({item.recipe_id for item in registry.recipes}, {
            "predict_demo_sample",
            "read_current_or_explicit_prediction",
            "explain_current_or_explicit_prediction",
            "explain_previous_prediction",
            "list_session_history",
            "compare_latest_two",
            "compare_explicit_pair",
            "search_general_knowledge",
            "activate_explanation_skill",
            "activate_comparison_skill",
            "activate_education_skill",
            "compare_and_explain_latest_two",
        })
        self.assertRegex(registry.version, r"^[0-9a-f]{64}$")

    @staticmethod
    def recipe(recipe_id="one", *, tool="get_prediction",
               source=ArgumentSource.BOUND_PREDICTION):
        return WorkflowRecipe(
            recipe_id=recipe_id,
            goal_patterns=(GoalPattern(
                IntentKind.SUMMARY,
                frozenset({TargetKind.CURRENT_PREDICTION})),),
            stages=(WorkflowStage(tool, source),),
            renderer=RendererKind.PREDICTION_DATA_AVAILABLE,
        )

    def test_rejects_duplicate_recipe_ids(self):
        recipe = self.recipe()

        with self.assertRaisesRegex(WorkflowRegistryError, "duplicate recipe"):
            WorkflowRegistry((recipe, recipe))

    def test_rejects_unknown_tools(self):
        with self.assertRaisesRegex(WorkflowRegistryError, "unknown tool"):
            WorkflowRegistry((self.recipe(tool="delete_patient"),))

    def test_rejects_argument_source_incompatible_with_tool(self):
        with self.assertRaisesRegex(WorkflowRegistryError, "argument source"):
            WorkflowRegistry((self.recipe(
                source=ArgumentSource.BOUND_SAMPLE),))

    def test_rejects_overlapping_recipe_patterns(self):
        with self.assertRaisesRegex(WorkflowRegistryError, "overlapping"):
            WorkflowRegistry((self.recipe("one"), self.recipe("two")))

    def test_version_is_deterministic(self):
        first = default_workflow_registry()
        second = WorkflowRegistry(first.recipes)

        self.assertEqual(first.version, second.version)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", first.version))


if __name__ == "__main__":
    unittest.main()
