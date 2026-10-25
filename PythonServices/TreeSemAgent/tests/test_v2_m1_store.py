"""Business idempotency checks need no LangGraph dependency."""
import tempfile
import unittest

from v2_m1.run_store import DemoRunStore, WorkflowConflict, operation_id


class RunStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = DemoRunStore(self.tmp.name + "/business.sqlite")
        self.store.setup()
        self.run_id = "run_" + "1" * 32
        self.store.create(self.run_id, "DOC-001", "P001")

    def commit(self, decision="confirm", phase="contract", revision=0):
        return self.store.apply(run_id=self.run_id, actor_id="DOC-001", phase=phase,
                                decision=decision, expected_revision=revision,
                                contract_revision=1,
                                op_id=operation_id(self.run_id, phase, revision))

    def test_repeated_commit_returns_prior_result(self):
        a = self.commit()
        b = self.commit()
        self.assertEqual(a, b)
        self.assertEqual(self.store.operation_count(self.run_id), 1)
        self.assertEqual(self.store.get(self.run_id, "DOC-001")["revision"], 1)

    def test_same_operation_id_different_payload_conflicts(self):
        self.commit("confirm")
        with self.assertRaises(WorkflowConflict):
            self.commit("cancel")
        self.assertEqual(self.store.operation_count(self.run_id), 1)

    def test_unguarded_stale_revision_rejected(self):
        self.commit()
        with self.assertRaises(WorkflowConflict):
            self.commit("acknowledge", "review", 0)

    def test_second_phase_creates_only_one_more_operation(self):
        self.commit()
        self.assertEqual(self.commit("acknowledge", "review", 1),
                         {"status": "completed_demo", "revision": 2})
        self.assertEqual(self.store.operation_count(self.run_id), 2)

    def test_wrong_actor_cannot_read_or_commit(self):
        with self.assertRaises(WorkflowConflict):
            self.store.get(self.run_id, "DOC-999")
        with self.assertRaises(WorkflowConflict):
            self.store.apply(run_id=self.run_id, actor_id="DOC-999", phase="contract",
                             decision="confirm", op_id=operation_id(self.run_id, "contract", 0),
                             expected_revision=0, contract_revision=1)

    def test_forged_operation_id_rejected(self):
        with self.assertRaises(WorkflowConflict):
            self.store.apply(run_id=self.run_id, actor_id="DOC-001", phase="contract",
                             decision="confirm", op_id="op_changed", expected_revision=0,
                             contract_revision=1)

    def test_cancel_is_terminal(self):
        self.commit("cancel")
        with self.assertRaises(WorkflowConflict):
            self.commit("acknowledge", "review", 1)

    def test_revise_is_terminal_pending_future_contract_edit(self):
        self.assertEqual(self.commit("revise")["status"], "needs_revision")
        self.assertEqual(self.store.operation_count(self.run_id), 1)


if __name__ == "__main__":
    unittest.main()
