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
            "列出历史记录，然后预测演示样本 5",
            "比较结果并分别解释两条决策路径",
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


if __name__ == "__main__":
    unittest.main()
