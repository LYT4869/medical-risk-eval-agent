import unittest

from agent.rule_evidence import (
    IntentAction,
    IntentObject,
    IntentReference,
    RuleEvidence,
)
from agent.rule_resolver import RuleIntentResolver
from agent.routing_types import RequestScope, RoutingSource


def evidence(*, actions=(), objects=(), references=(), scopes=(),
             explicit_skill=False, vague_reference=False):
    return RuleEvidence(
        actions=frozenset(actions),
        objects=frozenset(objects),
        references=frozenset(references),
        registry_scopes=frozenset(scopes),
        explicit_skill=explicit_skill,
        vague_reference=vague_reference,
    )


class RuleIntentResolverTest(unittest.TestCase):
    def setUp(self):
        self.resolver = RuleIntentResolver()

    def assert_scope(self, raw_evidence, expected, *, include_summary=False):
        decision = self.resolver.resolve(raw_evidence)
        self.assertIsNotNone(decision)
        self.assertEqual(decision.scope, expected)
        self.assertEqual(decision.source, RoutingSource.RULE)
        self.assertEqual(decision.include_summary, include_summary)

    def test_resolves_each_complete_business_intent(self):
        cases = (
            (evidence(
                actions={IntentAction.PREDICT},
                objects={IntentObject.DEMO_SAMPLE},
                references={IntentReference.EXPLICIT_SAMPLE}),
             RequestScope.PREDICTION),
            (evidence(
                actions={IntentAction.READ},
                objects={IntentObject.PREDICTION_FACT},
                references={IntentReference.CURRENT_PREDICTION}),
             RequestScope.SUMMARY),
            (evidence(
                actions={IntentAction.EXPLAIN},
                objects={IntentObject.EXPLANATION_DETAIL},
                references={IntentReference.CURRENT_PREDICTION}),
             RequestScope.EXPLANATION),
            (evidence(
                actions={IntentAction.LIST},
                objects={IntentObject.HISTORY}),
             RequestScope.HISTORY),
            (evidence(
                actions={IntentAction.COMPARE},
                objects={IntentObject.PREDICTION_FACT},
                references={IntentReference.MULTIPLE_PREDICTIONS}),
             RequestScope.COMPARISON),
            (evidence(
                actions={IntentAction.RETRIEVE},
                objects={IntentObject.KNOWLEDGE}),
             RequestScope.KNOWLEDGE),
        )

        for raw_evidence, expected in cases:
            with self.subTest(expected=expected):
                self.assert_scope(raw_evidence, expected)

    def test_summary_question_does_not_require_an_explicit_read_verb(self):
        raw_evidence = evidence(
            objects={IntentObject.PREDICTION_FACT},
            references={IntentReference.CURRENT_PREDICTION},
        )

        self.assert_scope(raw_evidence, RequestScope.SUMMARY)

    def test_general_knowledge_explanation_without_prediction_context(self):
        raw_evidence = evidence(
            actions={IntentAction.EXPLAIN},
            objects={IntentObject.KNOWLEDGE},
        )

        self.assert_scope(raw_evidence, RequestScope.KNOWLEDGE)

    def test_knowledge_source_only_modifies_prediction_explanation(self):
        raw_evidence = evidence(
            actions={IntentAction.EXPLAIN},
            objects={IntentObject.KNOWLEDGE,
                     IntentObject.PREDICTION_FACT},
            references={IntentReference.CURRENT_PREDICTION},
        )

        self.assert_scope(raw_evidence, RequestScope.EXPLANATION,
                          include_summary=True)

    def test_comparison_absorbs_history_and_summary_dependencies(self):
        raw_evidence = evidence(
            actions={IntentAction.COMPARE, IntentAction.LIST,
                     IntentAction.READ},
            objects={IntentObject.PREDICTION_FACT, IntentObject.HISTORY},
            references={IntentReference.CURRENT_PREDICTION,
                        IntentReference.PRIOR_PREDICTION,
                        IntentReference.MULTIPLE_PREDICTIONS},
        )

        self.assert_scope(raw_evidence, RequestScope.COMPARISON)

    def test_comparison_absorbs_detail_without_an_explain_action(self):
        raw_evidence = evidence(
            actions={IntentAction.COMPARE},
            objects={IntentObject.EXPLANATION_DETAIL},
            references={IntentReference.MULTIPLE_PREDICTIONS},
        )

        self.assert_scope(raw_evidence, RequestScope.COMPARISON)

    def test_listing_history_is_a_complete_history_intent(self):
        raw_evidence = evidence(
            actions={IntentAction.LIST},
            objects={IntentObject.HISTORY},
            references={IntentReference.PRIOR_PREDICTION},
        )

        self.assert_scope(raw_evidence, RequestScope.HISTORY)

    def test_explanation_absorbs_summary_for_the_same_prediction(self):
        raw_evidence = evidence(
            actions={IntentAction.EXPLAIN, IntentAction.READ},
            objects={IntentObject.EXPLANATION_DETAIL,
                     IntentObject.PREDICTION_FACT},
            references={IntentReference.CURRENT_PREDICTION},
        )

        self.assert_scope(
            raw_evidence, RequestScope.EXPLANATION, include_summary=True)

    def test_independent_complete_intents_are_compositional(self):
        cases = (
            evidence(
                actions={IntentAction.PREDICT, IntentAction.EXPLAIN},
                objects={IntentObject.DEMO_SAMPLE,
                         IntentObject.EXPLANATION_DETAIL},
                references={IntentReference.EXPLICIT_SAMPLE,
                            IntentReference.CURRENT_PREDICTION}),
            evidence(
                actions={IntentAction.COMPARE, IntentAction.RETRIEVE},
                objects={IntentObject.PREDICTION_FACT,
                         IntentObject.KNOWLEDGE},
                references={IntentReference.MULTIPLE_PREDICTIONS}),
            evidence(
                actions={IntentAction.EXPLAIN, IntentAction.RETRIEVE},
                objects={IntentObject.EXPLANATION_DETAIL,
                         IntentObject.KNOWLEDGE},
                references={IntentReference.CURRENT_PREDICTION}),
        )

        for raw_evidence in cases:
            with self.subTest(raw_evidence=raw_evidence):
                decision = self.resolver.resolve(raw_evidence)
                self.assertEqual(decision.scope, RequestScope.UNKNOWN)
                self.assertEqual(decision.reason, "rule_compositional")

    def test_incomplete_registry_match_does_not_force_a_scope(self):
        raw_evidence = evidence(scopes={RequestScope.SUMMARY})

        self.assertIsNone(self.resolver.resolve(raw_evidence))

    def test_explicit_skill_has_priority(self):
        raw_evidence = evidence(
            actions={IntentAction.COMPARE},
            objects={IntentObject.HISTORY},
            explicit_skill=True,
        )

        self.assert_scope(raw_evidence, RequestScope.SKILL)

    def test_vague_reference_abstains_with_stable_reason(self):
        decision = self.resolver.resolve(evidence(vague_reference=True))

        self.assertEqual(decision.scope, RequestScope.UNKNOWN)
        self.assertEqual(decision.reason, "rule_ambiguous_reference")


if __name__ == "__main__":
    unittest.main()
