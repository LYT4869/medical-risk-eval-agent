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


if __name__ == "__main__":
    unittest.main()
