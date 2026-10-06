import unittest

from agent.routing import RuleRouter
from agent.routing_types import RequestScope


class RoutingTaxonomyContractTest(unittest.TestCase):
    def setUp(self):
        self.router = RuleRouter()

    def test_workflow_dependencies_resolve_to_one_primary_scope(self):
        cases = (
            ("比较最近两次预测", RequestScope.COMPARISON, False),
            ("给出最新置信度，并与上一条记录比较",
             RequestScope.COMPARISON, False),
            ("show history，再 compare 最新两条",
             RequestScope.COMPARISON, False),
            ("解释当前预测的标签、概率和决策路径",
             RequestScope.EXPLANATION, True),
            ("Provide the stored feature and tree-path explanation",
             RequestScope.EXPLANATION, False),
            ("Retrieve technical documentation explaining AUC and F1",
             RequestScope.KNOWLEDGE, False),
            ("Explain model metrics with evidence.",
             RequestScope.KNOWLEDGE, False),
        )

        for message, expected_scope, include_summary in cases:
            with self.subTest(message=message):
                decision = self.router.route(message)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.scope, expected_scope)
                self.assertEqual(decision.include_summary, include_summary)

    def test_independent_business_goals_fall_back_as_compositional(self):
        messages = (
            "运行样本 3，然后解释刚生成的结果",
            "解释当前预测，再查找产后出血指南",
            "比较最近两次预测，并检索模型限制",
            "比较两条结果并提供医学资料",
            "show current label and provide clinical guidance",
            "列出历史记录，然后预测演示样本 5",
            "查看历史然后运行样本 2",
            "run sample 6 then display recent predictions",
            "运行样本 4，再列出我之前做过的预测",
            "比较两条预测后运行 56 号演示样本",
            "比较结果并分别解释两条决策路径",
            "解释当前预测的决策路径，并给出上一条预测的概率",
            "解释当前预测以及产后出血指南",
        )

        for message in messages:
            with self.subTest(message=message):
                decision = self.router.route(message)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.scope, RequestScope.UNKNOWN)
                self.assertEqual(decision.reason, "rule_compositional")

    def test_unbound_references_abstain_with_a_stable_reason(self):
        messages = (
            "帮我看看这个",
            "处理一下当前情况",
            "Can you help with this?",
        )

        for message in messages:
            with self.subTest(message=message):
                decision = self.router.route(message)
                self.assertIsNotNone(decision)
                self.assertEqual(decision.scope, RequestScope.UNKNOWN)
                self.assertEqual(decision.reason,
                                 "rule_ambiguous_reference")

    def test_clinical_word_does_not_turn_record_access_into_knowledge(self):
        decision = self.router.route(
            "read an unassigned patient's clinical prediction")

        self.assertIsNone(decision)

    def test_knowledge_source_is_only_a_modifier_for_stored_explanation(self):
        decision = self.router.route(
            "请根据资料解释当前预测概率")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.scope, RequestScope.EXPLANATION)
        self.assertTrue(decision.include_summary)

    def test_unanchored_nouns_do_not_force_domain_workflows(self):
        messages = (
            "What is the history of chess?",
            "What is a demo sample?",
            "Compare earlier versions",
            "请运行示例代码",
            "运行这个示例脚本",
        )

        for message in messages:
            with self.subTest(message=message):
                self.assertIsNone(self.router.route(message))

    def test_prediction_action_can_bind_an_explicit_case_reference(self):
        decision = self.router.route("run case 3")

        self.assertIsNotNone(decision)
        self.assertEqual(decision.scope, RequestScope.PREDICTION)


if __name__ == "__main__":
    unittest.main()
