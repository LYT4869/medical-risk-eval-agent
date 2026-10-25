"""M2 full synthetic graph, durable business replay and authorization boundaries."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from v2_m2.artifact_store import SyntheticArtifactStore
from v2_m2.fixture_tools import SyntheticP001Tools
from v2_m2.service import M2SyntheticService


class M2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.m1 = DemoWorkflowService(self.temp.name)
        self.m2 = M2SyntheticService(self.temp.name)

    def confirmed_run(self):
        run = self.m1.start()
        self.m1.resume(run["run_id"], "confirm", 0, 1)
        return run["run_id"]

    def test_unconfirmed_run_has_no_artifacts(self):
        run_id = self.m1.start()["run_id"]
        with self.assertRaises(WorkflowConflict):
            self.m2.execute(run_id)
        self.assertEqual(self.m2.artifacts.count(run_id), 0)

    def test_revise_requires_reconfirmation_before_tools(self):
        run_id = self.m1.start()["run_id"]
        self.m1.resume(run_id, "revise", 0, 1,
                       summary="只分析来源冲突", overview="动态调查来源差异")
        with self.assertRaises(WorkflowConflict):
            self.m2.execute(run_id)
        self.m1.resume(run_id, "confirm", 1, 2)
        self.assertEqual(self.m2.execute(run_id)["report"]["contract_revision"], 2)

    def test_one_issue_two_sources_six_section_restricted_draft(self):
        run_id = self.confirmed_run()
        outcome = self.m2.execute(run_id)
        self.assertEqual(outcome["business"], "draft_ready")
        self.assertEqual(outcome["next"], [])
        self.assertEqual(outcome["artifacts"], 4)
        self.assertEqual(outcome["operations"], 4)
        draft = outcome["report"]
        self.assertEqual(draft["status"], "restricted_draft")
        self.assertEqual(draft["source_snapshot"], ["E-002:v1", "E-003:v1"])
        self.assertEqual(len(draft["sections"]), 6)
        self.assertTrue(draft["not_signed"])
        self.assertIsNone(draft["model_score"])
        self.assertNotIn("已确诊", json.dumps(draft, ensure_ascii=False))
        self.assertNotIn("实际失血量 800", json.dumps(draft, ensure_ascii=False))
        issue = self.m2.artifacts.get(run_id, "issue", "I-04:r1")
        self.assertEqual(issue["status"], "closed_unresolved")
        self.assertEqual(set(issue["statements"][2]["evidence_ids"]),
                         {"E-002", "E-003"})

    def test_replay_does_not_duplicate_draft_or_evidence(self):
        run_id = self.confirmed_run()
        first = self.m2.execute(run_id)
        second = M2SyntheticService(self.temp.name).execute(run_id)
        self.assertEqual(first["report"], second["report"])
        self.assertEqual((second["artifacts"], second["operations"]), (4, 4))

    def test_wrong_actor_subject_or_cancelled_contract_rejected(self):
        run_id = self.confirmed_run()
        with self.assertRaises(WorkflowConflict):
            self.m2.execute(run_id, actor_id="DOC-999")
        bad = self.m1.start()
        self.m1.resume(bad["run_id"], "cancel", 0, 1)
        with self.assertRaises(WorkflowConflict):
            self.m2.execute(bad["run_id"])

    def test_replayed_artifact_operation_rejects_changed_payload(self):
        run_id = self.confirmed_run()
        self.m2.execute(run_id)
        with self.assertRaises(WorkflowConflict):
            self.m2.artifacts.save(run_id, "issue", "I-04:r1",
                                   {"invalid": "changed"})
        self.assertEqual(self.m2.artifacts.count(run_id), 4)

    def test_tools_reject_future_and_unapproved_sources(self):
        tools = SyntheticP001Tools()
        with self.assertRaises(WorkflowConflict):
            tools.call("read_patient_record", actor_id="DOC-001", patient_id="P001",
                       as_of="2026-10-11", record_id="E-003")
        with self.assertRaises(WorkflowConflict):
            tools.call("read_patient_record", actor_id="DOC-001", patient_id="P001",
                       as_of="2026-10-18", record_id="E-005")
        with self.assertRaises(WorkflowConflict):
            tools.call("execute_arbitrary_tool", actor_id="DOC-001", patient_id="P001",
                       as_of="2026-10-18")
        with self.assertRaises(WorkflowConflict):
            tools.call("get_patient_timeline", actor_id="DOC-001", patient_id="P002",
                       as_of="2026-10-18")

    def test_different_runs_do_not_read_each_others_artifacts(self):
        a, b = self.confirmed_run(), self.confirmed_run()
        self.m2.execute(a)
        with self.assertRaises(WorkflowConflict):
            self.m2.artifacts.get(b, "issue", "I-04:r1")
        self.assertEqual(self.m2.artifacts.count(b), 0)

    def test_m1_and_m2_cli_in_separate_python_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            def cli(*args):
                result = subprocess.run(
                    [sys.executable, "-m", *args], env=dict(os.environ),
                    check=True, capture_output=True, text=True,
                )
                return json.loads(result.stdout)
            started = cli("v2_m1.cli", "--data-dir", directory, "start")
            run_id = started["run_id"]
            cli("v2_m1.cli", "--data-dir", directory,
                "confirm", run_id, "--decision", "confirm",
                "--revision", "0", "--contract-revision", "1")
            completed = cli("v2_m2.cli", "--data-dir", directory, run_id)
            self.assertEqual((completed["next"], completed["artifacts"]), ([], 4))
            again = cli("v2_m2.cli", "--data-dir", directory, run_id)
            self.assertEqual(completed["report"], again["report"])


if __name__ == "__main__":
    unittest.main()
