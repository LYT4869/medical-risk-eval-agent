"""Real SqliteSaver interrupt/resume tests (LangGraph dependency required)."""
import tempfile
import unittest

from v2_m1.run_store import WorkflowConflict, operation_id
from v2_m1.service import DemoWorkflowService


class GraphDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.svc = DemoWorkflowService(self.temp.name)

    def test_process_reconstruction_resumes_both_gates(self):
        started = self.svc.start()
        self.assertEqual(started["next"], ["contract_gate"])
        resumed = DemoWorkflowService(self.temp.name).resume(
            started["run_id"], "contract", "confirm", 0)
        self.assertEqual((resumed["business"], resumed["next"]),
                         ("awaiting_review", ["review_gate"]))
        finished = DemoWorkflowService(self.temp.name).resume(
            started["run_id"], "review", "acknowledge", 1)
        self.assertEqual((finished["business"], finished["next"], finished["operations"]),
                         ("completed_demo", [], 2))

    def test_side_effect_commit_before_checkpoint_is_idempotent(self):
        started = self.svc.start()
        run_id = started["run_id"]
        # Simulate a crash *after* the business transaction but *before*
        # the graph has stored the completed node output.
        self.svc.store.apply(run_id=run_id, actor_id="DOC-001", phase="contract",
                             decision="confirm", op_id=operation_id(run_id, "contract", 0),
                             expected_revision=0, contract_revision=1)
        self.assertEqual(self.svc.status(run_id)["next"], ["contract_gate"])
        recovered = DemoWorkflowService(self.temp.name).resume(
            run_id, "contract", "confirm", 0)
        self.assertEqual((recovered["business"], recovered["operations"]),
                         ("awaiting_review", 1))

    def test_duplicate_confirmation_does_not_advance(self):
        run_id = self.svc.start()["run_id"]
        self.svc.resume(run_id, "contract", "confirm", 0)
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "contract", "confirm", 0)
        self.assertEqual(self.svc.status(run_id)["operations"], 1)

    def test_wrong_actor_and_wrong_gate(self):
        run_id = self.svc.start()["run_id"]
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "contract", "confirm", 0, "DOC-999")
        with self.assertRaises(WorkflowConflict):
            self.svc.resume(run_id, "review", "acknowledge", 0)

    def test_cancel_and_revise_do_not_enter_review(self):
        for action in ("cancel", "revise"):
            run_id = self.svc.start()["run_id"]
            status = self.svc.resume(run_id, "contract", action, 0)
            self.assertEqual(status["next"], [])
            self.assertEqual(status["business"],
                             "cancelled" if action == "cancel" else "needs_revision")

    def test_graph_checkpoint_does_not_hold_actor_or_patient(self):
        run_id = self.svc.start()["run_id"]
        from langgraph.checkpoint.sqlite import SqliteSaver
        from v2_m1.workflow import build_graph
        with SqliteSaver.from_conn_string(self.svc.checkpoints) as saver:
            state = build_graph(self.svc.store, "DOC-001", saver).get_state(self.svc.config(run_id)).values
        self.assertEqual(set(state), {"run_id", "contract_revision", "business_revision",
                                      "contract_operation_id", "review_operation_id", "phase"})
        self.assertNotIn("token", repr(state).lower())
        self.assertNotIn("DOC-001", repr(state))
        self.assertNotIn("P001", repr(state))


if __name__ == "__main__":
    unittest.main()
