import unittest

from agent.rule_evidence import (
    IntentAction,
    IntentObject,
    IntentReference,
    RuleEvidenceExtractor,
)


class RuleEvidenceExtractorTest(unittest.TestCase):
    def setUp(self):
        self.extractor = RuleEvidenceExtractor()

    def test_extracts_prediction_action_object_and_sample_reference(self):
        evidence = self.extractor.extract("运行第 3 个演示样本")

        self.assertEqual(evidence.actions, {IntentAction.PREDICT})
        self.assertEqual(evidence.objects, {IntentObject.DEMO_SAMPLE})
        self.assertEqual(
            evidence.references, {IntentReference.EXPLICIT_SAMPLE})

    def test_extracts_stored_prediction_read_evidence(self):
        evidence = self.extractor.extract("查看当前预测的概率")

        self.assertEqual(evidence.actions, {IntentAction.READ})
        self.assertEqual(evidence.objects, {
            IntentObject.PREDICTION_RECORD,
            IntentObject.PREDICTION_FACT,
        })
        self.assertEqual(
            evidence.references, {IntentReference.CURRENT_PREDICTION})

    def test_extracts_explanation_detail_and_current_reference(self):
        evidence = self.extractor.extract("解释刚才结果的决策路径")

        self.assertEqual(evidence.actions, {IntentAction.EXPLAIN})
        self.assertEqual(evidence.objects, {
            IntentObject.PREDICTION_RECORD,
            IntentObject.EXPLANATION_DETAIL,
        })
        self.assertEqual(
            evidence.references, {IntentReference.CURRENT_PREDICTION})

    def test_extracts_comparison_references(self):
        evidence = self.extractor.extract("比较这次与上一次结果")

        self.assertEqual(evidence.actions, {IntentAction.COMPARE})
        self.assertEqual(evidence.objects, {IntentObject.PREDICTION_RECORD})
        self.assertEqual(evidence.references, {
            IntentReference.CURRENT_PREDICTION,
            IntentReference.PRIOR_PREDICTION,
            IntentReference.MULTIPLE_PREDICTIONS,
        })

    def test_extracts_knowledge_retrieval_evidence(self):
        evidence = self.extractor.extract("查找产后出血指南")

        self.assertEqual(evidence.actions, {IntentAction.RETRIEVE})
        self.assertEqual(evidence.objects, {IntentObject.KNOWLEDGE})
        self.assertEqual(evidence.references, frozenset())

    def test_generic_result_is_not_an_explicit_summary_fact(self):
        evidence = self.extractor.extract("解释当前结果的重要特征")

        self.assertIn(IntentObject.PREDICTION_RECORD, evidence.objects)
        self.assertIn(IntentObject.EXPLANATION_DETAIL, evidence.objects)
        self.assertNotIn(IntentObject.PREDICTION_FACT, evidence.objects)

    def test_marks_explicit_skill_requests(self):
        evidence = self.extractor.extract("使用解释技能")

        self.assertTrue(evidence.explicit_skill)

    def test_normalizes_unicode_case_and_spacing(self):
        evidence = self.extractor.extract("  PREDICT   SAMPLE ３ ")

        self.assertEqual(evidence.actions, {IntentAction.PREDICT})
        self.assertEqual(evidence.objects, {IntentObject.DEMO_SAMPLE})
        self.assertEqual(
            evidence.references, {IntentReference.EXPLICIT_SAMPLE})

    def test_unrelated_text_has_no_business_evidence(self):
        evidence = self.extractor.extract("你好，今天天气不错")

        self.assertEqual(evidence.actions, frozenset())
        self.assertEqual(evidence.objects, frozenset())
        self.assertEqual(evidence.references, frozenset())
        self.assertEqual(evidence.registry_scopes, frozenset())
        self.assertFalse(evidence.explicit_skill)
        self.assertFalse(evidence.vague_reference)


if __name__ == "__main__":
    unittest.main()
