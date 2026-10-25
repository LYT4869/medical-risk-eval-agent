"""M3 SIM single-Agent, Agenda, guard, durability and context regression."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from v2_m3.agent import FakeInvestigationAgent, OpenAICompatibleInvestigationAgent
from v2_m3.policy import (
    M3FixtureTools, checked_decision, initial_agenda, MAX_STEPS,
)
from v2_m3.repository import LocalM3Repository
from v2_m3.service import M3LocalWorkflow


class BadAgent:
    def decide(self, context):
        return {"action": "tool", "tool": "delete_patient_records"}


class UnjustifiedCloseAgent:
    def decide(self, context):
        return {"action": "close", "evidence_ids": [], "followup": None}


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.m1 = DemoWorkflowService(self.temp.name)

    def approved(self) -> str:
        start = self.m1.start()
        self.m1.resume(start["run_id"], "confirm", 0, 1)
        return start["run_id"]

    def test_real_langgraph_dynamic_i04_and_unresolved_conflict(self):
        rid = self.approved()
        graph = M3LocalWorkflow(self.temp.name)
        out = graph.execute(rid, 1)
        self.assertEqual(out["phase"], "draft_ready")
        self.assertEqual(out["next"], [])
        self.assertEqual([item["issue_id"] for item in out["agenda"]],
                         ["I-01", "I-02", "I-03", "I-04"])
        self.assertEqual(out["tool_calls"], 4)
        self.assertLessEqual(out["steps"], MAX_STEPS)
        self.assertIn("agenda:add:I-04", out["trace"])
        self.assertEqual(
            graph.artifacts.count(rid), 8)  # 3 Evidence, 4 Issue, 1 Draft
        issue = graph.artifacts.get(rid, "issue", "I-04:r1")
        self.assertEqual(issue["status"], "closed_unresolved")
        self.assertEqual(issue["evidence_ids"], ["E-002", "E-003"])
        self.assertTrue(issue["no_diagnosis"])
        draft = out["report"]
        self.assertTrue(draft["template_only"])
        self.assertTrue(draft["not_signed"])
        self.assertEqual(draft["source_as_of"], "2026-10-12")
        self.assertEqual(draft["source_validity"], "historical_T1_not_current")
        self.assertEqual(draft["coverage"]["I-03"], "blocked")
        self.assertEqual(len(draft["sections"]), 6)
        self.assertIsNone(draft["model_score"])
        self.assertNotIn("已确诊", json.dumps(draft, ensure_ascii=False))

    def test_cross_process_graph_resume_does_not_duplicate_artifacts(self):
        run_id = self.approved()
        first = M3LocalWorkflow(self.temp.name).execute(run_id, 1)
        recovered = M3LocalWorkflow(self.temp.name).execute(run_id, 1)
        self.assertEqual(first["agenda"], recovered["agenda"])
        self.assertEqual(first["report"], recovered["report"])
        self.assertEqual(recovered["next"], [])
        self.assertEqual(M3LocalWorkflow(self.temp.name).artifacts.count(run_id), 8)

    def test_no_confirm_no_agent_or_artifact(self):
        run_id = self.m1.start()["run_id"]
        graph = M3LocalWorkflow(self.temp.name)
        with self.assertRaises(WorkflowConflict):
            graph.execute(run_id, 1)
        self.assertEqual(graph.artifacts.count(run_id), 0)

    def test_cancelled_contract_stops(self):
        run_id = self.m1.start()["run_id"]
        self.m1.resume(run_id, "cancel", 0, 1)
        with self.assertRaises(WorkflowConflict):
            M3LocalWorkflow(self.temp.name).execute(run_id, 1)

    def test_wrong_contract_revision_rejected(self):
        run_id = self.approved()
        with self.assertRaises(WorkflowConflict):
            M3LocalWorkflow(self.temp.name).execute(run_id, 2)

    def test_invalid_llm_tool_terminates_without_side_effects(self):
        run_id = self.approved()
        graph = M3LocalWorkflow(self.temp.name, agent=BadAgent())
        result = graph.execute(run_id, 1)
        self.assertEqual(result["phase"], "blocked")
        self.assertEqual(result["steps"], 2)
        self.assertNotIn("report", result)
        self.assertEqual(graph.artifacts.count(run_id), 0)
        self.assertIn("guard:invalid_proposal", result["trace"])

    def test_llm_cannot_close_without_seen_evidence(self):
        run_id = self.approved()
        result = M3LocalWorkflow(
            self.temp.name, agent=UnjustifiedCloseAgent()).execute(run_id, 1)
        self.assertEqual(result["phase"], "blocked")
        self.assertNotIn("report", result)

    def test_max_step_budget_stops_before_report(self):
        run_id = self.approved()
        output = M3LocalWorkflow(self.temp.name).execute(
            run_id, 1, max_steps=2)
        self.assertEqual(output["phase"], "blocked")
        self.assertNotIn("report", output)

    def test_tool_rejects_future_or_scope_violation(self):
        tools = M3FixtureTools()
        with self.assertRaises(WorkflowConflict):
            tools.call("read_patient_record", actor_id="DOC-001",
                       patient_id="P001", as_of="2026-10-12", record_id="E-005")
        with self.assertRaises(WorkflowConflict):
            tools.call("read_patient_record", actor_id="DOC-001",
                       patient_id="P002", as_of="2026-10-12", record_id="E-002")
        with self.assertRaises(WorkflowConflict):
            tools.call("read_patient_record", actor_id="DOC-001",
                       patient_id="P001", as_of="2026-10-18", record_id="E-002")

    def test_guard_tool_whitelist_and_issue_scoping(self):
        agenda = initial_agenda()
        state = {"agenda": agenda, "tool_calls": 0,
                 "timeline_loaded": True, "seen_evidence": []}
        with self.assertRaises(WorkflowConflict):
            checked_decision({"action": "tool", "tool": "run_model"},
                             agenda[0], state)
        with self.assertRaises(WorkflowConflict):
            checked_decision(
                {"action": "tool", "tool": "read_patient_record",
                 "record_id": "E-003"}, agenda[0], state)
        with self.assertRaises(WorkflowConflict):
            checked_decision(
                {"action": "close", "evidence_ids": ["E-001"],
                 "followup": "model_probability"}, agenda[0], state)

    def test_repeat_operation_payload_conflict(self):
        rid = self.approved()
        repo = LocalM3Repository(self.temp.name + "/m3_business.sqlite3")
        ref = {"classification": "SIM", "run_id": rid,
               "evidence_id": "E-001", "record": {"id": "E-001"}}
        self.assertEqual(
            repo.save(rid, "evidence", "E-001:v1", ref),
            repo.save(rid, "evidence", "E-001:v1", ref))
        with self.assertRaises(WorkflowConflict):
            repo.save(rid, "evidence", "E-001:v1",
                      {**ref, "evidence_id": "E-002"})
        self.assertEqual(repo.count(rid), 1)

    def test_cross_run_artifact_cannot_be_read(self):
        a, b = self.approved(), self.approved()
        repo = M3LocalWorkflow(self.temp.name)
        repo.execute(a, 1)
        with self.assertRaises(WorkflowConflict):
            repo.artifacts.get(b, "issue", "I-04:r1")

    def test_offline_model_adapter_rejects_arbitrary_plain_http(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleInvestigationAgent(
                "http://example.invalid/v1/chat/completions", "qwen")
        with self.assertRaises(ValueError):
            OpenAICompatibleInvestigationAgent(
                "http://localhost/v1/chat/completions", "")


if __name__ == "__main__":
    unittest.main()
