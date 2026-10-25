"""True separate processes exercise mandatory Contract confirmation."""
import json
import os
import subprocess
import sys
import tempfile
import unittest


class CrossProcessTests(unittest.TestCase):
    def test_revise_then_reconfirm_in_distinct_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(*args):
                result = subprocess.run(
                    [sys.executable, "-m", "v2_m1.cli", "--data-dir", directory, *args],
                    env=dict(os.environ), check=True, text=True, capture_output=True,
                )
                return json.loads(result.stdout)

            started = run("start")
            run_id = started["run_id"]
            self.assertEqual(started["next"], ["contract_gate"])
            self.assertIn("PPH", run("status", run_id)["contract"]["summary"])
            revised = run("confirm", run_id, "--decision", "revise",
                          "--revision", "0", "--contract-revision", "1",
                          "--summary", "修改后的模拟调查目标",
                          "--overview", "医生先确认范围，Agent 动态安排下一步")
            self.assertEqual((revised["next"], revised["contract"]["revision"]),
                             (["contract_gate"], 2))
            finished = run("confirm", run_id, "--decision", "confirm",
                           "--revision", "1", "--contract-revision", "2")
            self.assertEqual((finished["business"], finished["next"], finished["operations"]),
                             ("contract_confirmed", [], 2))

    def test_clarify_then_reconfirm_in_distinct_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(*args):
                output = subprocess.run(
                    [sys.executable, "-m", "v2_m1.cli", "--data-dir", directory, *args],
                    env=dict(os.environ), check=True, text=True, capture_output=True,
                )
                return json.loads(output.stdout)

            started = run("start", "--ask-clarification")
            run_id = started["run_id"]
            self.assertTrue(started["contract"]["clarification_question"])
            answered = run("confirm", run_id, "--decision", "clarify",
                           "--revision", "0", "--contract-revision", "1",
                           "--answer", "可以生成受限草稿，但明确标注限制")
            self.assertEqual(answered["contract"]["revision"], 2)
            self.assertEqual(answered["next"], ["contract_gate"])
            final = run("confirm", run_id, "--decision", "confirm",
                        "--revision", "1", "--contract-revision", "2")
            self.assertEqual(final["business"], "contract_confirmed")


if __name__ == "__main__":
    unittest.main()
