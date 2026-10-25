"""Real LangGraph SQLite Saver contract HITL loop and crash-replay tests."""
import tempfile
import unittest

from v2_m1.run_store import WorkflowConflict, operation_id
from v2_m1.service import DemoWorkflowService


class GraphDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.svc = DemoWorkflowService(self.temp.name)

    def test_initial_confirmation_is_mandatory_and_terminal(self):
        started = self.svc.start()
        self.assertEqual(started["business"], "awaiting_confirmation")
        self.assertEqual(started["next"], ["contract_gate"])
        self.assertIn("PPH", started["contract"]["summary"])
        self.assertIn("动态决定", started["contract"]["investigation_overview"])
        finished = DemoWorkflowService(self.temp.name).resume(
            started["run_id"], "confirm", 0, 1)
        self.assertEqual((finished["business"], finished["next"], finished["operations"]),
                         ("contract_confirmed", [], 1))
        self.assertEqual(finished["contract"]["revision"], 1)

    def test_revise_then_reconfirm_on_new_process(self):
        run_id = self.svc.start()["run_id"]
        revised = DemoWorkflowService(self.temp.name).resume(
            run_id, "revise", 0, 1, summary="模拟目标修改为检查记录差异",
            overview="动态选择病史、检验和模型适用性")
        self.assertEqual((revised["business"], revised["next"]),
                         ("awaiting_confirmation", ["contract_gate"]))
        self.assertEqual((revised["revision"], revised["contract"]["revision"]), (1, 2))
        self.assertEqual(revised["contract"]["summary"], "模拟目标修改为检查记录差异")
        finished = DemoWorkflowService(self.temp.name).resume(
            run_id, "confirm", 1, 2)
        self.assertEqual((finished["business"], finished["next"], finished["operations"]),
                         ("contract_confirmed", [], 2))
        self.assertEqual(self.svc.store.contract_count(run_id), 2)

    def test_clarification_answer_requires_reconfirmation(self):
        started = self.svc.start(ask_clarification=True)
        run_id = started["run_id"]
        self.assertIsNotNone(started["contract"]["clarification_question"])
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "confirm", 0, 1)
        answered = DemoWorkflowService(self.temp.name).resume(
            run_id, "clarify", 0, 1, answer="允许受限草稿，但列清未决项")
        self.assertEqual((answered["business"], answered["next"]),
                         ("awaiting_confirmation", ["contract_gate"]))
        self.assertEqual(answered["contract"]["revision"], 2)
        self.assertIsNone(answered["contract"]["clarification_question"])
        finished = DemoWorkflowService(self.temp.name).resume(
            run_id, "confirm", 1, 2)
        self.assertEqual((finished["business"], finished["next"]),
                         ("contract_confirmed", []))

    def test_revise_commit_before_checkpoint_replayed_once(self):
        started = self.svc.start()
        run_id = started["run_id"]
        data = {"summary": "业务已经提交的新草案", "overview": "再次确认后动态调查"}
        self.svc.store.apply(
            run_id=run_id, actor_id="DOC-001", phase="contract", decision="revise",
            op_id=operation_id(run_id, "contract", 0),
            expected_revision=0, contract_revision=1, **data,
        )
        self.assertEqual(self.svc.status(run_id)["next"], ["contract_gate"])
        self.assertEqual(self.svc.store.contract_count(run_id), 2)
        recovered = DemoWorkflowService(self.temp.name).resume(
            run_id, "revise", 0, 1, **data)
        self.assertEqual((recovered["next"], recovered["operations"]), (["contract_gate"], 1))
        self.assertEqual(self.svc.store.contract_count(run_id), 2)
        self.assertEqual(DemoWorkflowService(self.temp.name).resume(
            run_id, "confirm", 1, 2)["business"], "contract_confirmed")

    def test_confirm_commit_before_checkpoint_replayed_once(self):
        started = self.svc.start()
        run_id = started["run_id"]
        self.svc.store.apply(
            run_id=run_id, actor_id="DOC-001", phase="contract",
            decision="confirm", op_id=operation_id(run_id, "contract", 0),
            expected_revision=0, contract_revision=1,
        )
        recovered = DemoWorkflowService(self.temp.name).resume(
            run_id, "confirm", 0, 1)
        self.assertEqual((recovered["business"], recovered["next"], recovered["operations"]),
                         ("contract_confirmed", [], 1))

    def test_old_revision_duplicate_and_wrong_actor_rejected(self):
        run_id = self.svc.start()["run_id"]
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "confirm", 0, 2)
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "confirm", 0, 1, actor_id="DOC-999")
        revised = self.svc.resume(run_id, "revise", 0, 1,
                                  summary="新目标", overview="新调查方向")
        self.assertEqual(revised["contract"]["revision"], 2)
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "confirm", 0, 1)
        self.svc.resume(run_id, "confirm", 1, 2)
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "confirm", 1, 2)
        self.assertEqual(self.svc.status(run_id)["operations"], 2)

    def test_cancel_never_enters_investigation(self):
        started = self.svc.start()
        stopped = self.svc.resume(started["run_id"], "cancel", 0, 1)
        self.assertEqual((stopped["business"], stopped["next"]),
                         ("cancelled", []))

    def test_checkpoint_contains_refs_and_revisions_only(self):
        started = self.svc.start()
        run_id = started["run_id"]
        from langgraph.checkpoint.sqlite import SqliteSaver
        from v2_m1.workflow import build_graph
        with SqliteSaver.from_conn_string(self.svc.checkpoints) as saver:
            snapshot = build_graph(self.svc.store, "DOC-001", saver).get_state(self.svc.config(run_id))
        state = snapshot.values
        self.assertEqual(set(state), {
            "run_id", "contract_revision", "business_revision",
            "contract_operation_id", "phase",
        })
        self.assertNotIn("DOC-001", repr(state))
        self.assertNotIn("P001", repr(state))
        self.assertNotIn("PPH", repr(state))
        self.assertNotIn("token", repr(state).lower())


if __name__ == "__main__":
    unittest.main()
