"""True separate Python processes share durable SQLite checkpoints."""
import json
import os
import subprocess
import sys
import tempfile
import unittest


class CrossProcessTests(unittest.TestCase):
    def test_restart_with_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            def run(*args):
                env = dict(os.environ)
                # In CI, PYTHONPATH points at PythonServices/TreeSemAgent.
                result = subprocess.run(
                    [sys.executable, "-m", "v2_m1.cli", "--data-dir", directory, *args],
                    env=env, check=True, text=True, capture_output=True)
                return json.loads(result.stdout)
            started = run("start")
            run_id = started["run_id"]
            self.assertEqual(started["next"], ["contract_gate"])
            self.assertEqual(run("status", run_id)["business"], "awaiting_confirmation")
            confirmed = run("confirm", run_id, "--decision", "confirm", "--revision", "0")
            self.assertEqual(confirmed["next"], ["review_gate"])
            finished = run("review", run_id, "--decision", "acknowledge", "--revision", "1")
            self.assertEqual((finished["next"], finished["operations"]), ([], 2))


if __name__ == "__main__":
    unittest.main()
