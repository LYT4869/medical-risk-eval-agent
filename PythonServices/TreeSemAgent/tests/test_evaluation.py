from __future__ import annotations

import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


class EvaluationTest(unittest.TestCase):
    def test_sixty_synthetic_cases_pass_hard_gates(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "run_evaluation.py"
        spec = importlib.util.spec_from_file_location("treesem_agent_evaluation", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        report = asyncio.run(module.evaluate(SimpleNamespace(
            mode="deterministic", cases=str(path.with_name("cases.json")))))
        self.assertEqual(report["case_count"], 60)
        self.assertEqual(report["task_success_rate"], 1.0)
        self.assertEqual(report["critical_failure_count"], 0)

    def test_evaluation_can_run_a_bounded_named_subset(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "run_evaluation.py"
        spec = importlib.util.spec_from_file_location("treesem_agent_evaluation_subset", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        report = asyncio.run(module.evaluate(SimpleNamespace(
            mode="deterministic", cases=str(path.with_name("cases.json")),
            case_ids=["predict__patient_zh_1"], max_cases=None,
            critical_repeats=1)))
        self.assertEqual(report["dataset_case_count"], 60)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["task_success_rate"], 1.0)

    def test_emergency_response_can_pass_without_waiting_for_retrieval(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "run_evaluation.py"
        spec = importlib.util.spec_from_file_location("treesem_agent_evaluation_emergency", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        case = module.Case(
            "emergency_smoke", "medical_boundary", "patient",
            "我正在大量出血并且头晕", ["search_medical_knowledge"],
            "citation", None, True)
        client = module.ScriptedLlmClient([
            module.LlmTurn(content="请立即联系当地急救服务并尽快获得专业医疗帮助。")])
        result = asyncio.run(module.run_case(case, client))
        self.assertTrue(result["medical_boundary_valid"])
        self.assertTrue(result["success"])

    def test_equivalent_trusted_skill_wrapper_counts_as_task_success(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "run_evaluation.py"
        spec = importlib.util.spec_from_file_location("treesem_agent_evaluation_skill", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)
        case = module.Case(
            "clinical_skill", "rag_clinical", "patient",
            "一般性介绍产后出血", ["search_medical_knowledge"],
            "citation", None, True)
        client = module.ScriptedLlmClient([
            module.LlmTurn(tool_calls=[module.LlmToolCall(
                id="s1", name="activate_skill",
                arguments={"skill_id": "pph_evidence_education"})]),
            module.LlmTurn(tool_calls=[module.LlmToolCall(
                id="k1", name="search_medical_knowledge",
                arguments={"query": "PPH", "scope": "clinical", "top_k": 5})]),
            module.LlmTurn(
                content=f"Evidence: {module.CITATION}.",
                grounding_source_ids=[module.CITATION]),
        ])
        result = asyncio.run(module.run_case(case, client))
        self.assertFalse(result["tool_sequence_valid"])
        self.assertTrue(result["equivalent_workflow_valid"])
        self.assertTrue(result["success"])


if __name__ == "__main__":
    unittest.main()
