import unittest

from agent.run_guard import AgentRunGuard
from agent.workflow import WorkflowMode, WorkflowPlanner


class WorkflowPlannerTest(unittest.TestCase):
    @staticmethod
    def plan(message: str):
        guard = AgentRunGuard.for_request(message)
        return WorkflowPlanner.for_request(message, guard)

    def test_clear_scopes_have_minimal_ordered_stages(self):
        cases = [
            ("预测演示样本 3", ("predict_sample",)),
            ("读取当前预测的标签和概率", ("get_prediction",)),
            ("解释当前结果的重要特征和决策路径",
             ("get_explanation",)),
            ("查看最近五次预测历史", ("get_prediction_history",)),
            ("比较最近两次预测",
             ("get_prediction_history", "compare_predictions")),
            ("查询产后出血权威资料并引用",
             ("search_medical_knowledge",)),
        ]

        for message, stages in cases:
            with self.subTest(message=message):
                result = self.plan(message)
                self.assertEqual(result.mode, WorkflowMode.DETERMINISTIC)
                self.assertEqual(result.stages, stages)

    def test_explanation_with_summary_facts_reads_both_in_order(self):
        result = self.plan("读取当前预测概率并解释重要特征")

        self.assertEqual(
            result.stages, ("get_prediction", "get_explanation"))

    def test_explicit_skills_have_required_minimal_stages(self):
        comparison = self.plan("使用可信历史比较流程分析最近两次结果")
        self.assertEqual(comparison.expected_skill_id,
                         "compare_prediction_history")
        self.assertEqual(comparison.stages, (
            "activate_skill", "get_prediction_history",
            "compare_predictions"))

        education = self.plan("按循证教育技能介绍产后出血")
        self.assertEqual(education.expected_skill_id,
                         "pph_evidence_education")
        self.assertEqual(education.stages, (
            "activate_skill", "search_medical_knowledge"))

        explanation = self.plan("使用稳定的预测解释流程说明当前结果")
        self.assertEqual(explanation.expected_skill_id,
                         "explain_prediction")
        self.assertEqual(explanation.stages, (
            "activate_skill", "get_prediction", "get_explanation",
            "search_medical_knowledge"))

    def test_ambiguous_request_keeps_open_agent(self):
        result = self.plan("帮我看看这个情况")

        self.assertEqual(result.mode, WorkflowMode.OPEN_AGENT)
        self.assertEqual(result.stages, ())

    def test_ambiguous_skill_request_keeps_open_agent(self):
        result = self.plan("请使用一个稳定流程帮助我")

        self.assertEqual(result.mode, WorkflowMode.OPEN_AGENT)
        self.assertEqual(result.expected_skill_id, None)


if __name__ == "__main__":
    unittest.main()
