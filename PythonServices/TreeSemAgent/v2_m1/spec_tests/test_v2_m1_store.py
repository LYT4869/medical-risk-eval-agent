"""Business idempotency, versioned Contract drafts and atomic transitions."""
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

    def commit(self, decision="confirm", revision=0, contract_revision=1,
               **data):
        return self.store.apply(
            run_id=self.run_id, actor_id="DOC-001", phase="contract",
            decision=decision, expected_revision=revision,
            contract_revision=contract_revision,
            op_id=operation_id(self.run_id, "contract", revision), **data
        )

    def test_confirm_is_terminal_without_review(self):
        result = self.commit()
        self.assertEqual(result, {"status": "contract_confirmed",
                                  "revision": 1, "contract_revision": 1})
        self.assertEqual(self.store.operation_count(self.run_id), 1)
        self.assertEqual(self.store.contract_count(self.run_id), 1)

    def test_same_commit_replay_returns_prior(self):
        self.assertEqual(self.commit(), self.commit())
        self.assertEqual(self.store.operation_count(self.run_id), 1)

    def test_conflicting_payload_on_same_operation_rejected(self):
        self.commit("cancel")
        with self.assertRaises(WorkflowConflict):
            self.commit("confirm")

    def test_revise_is_a_reconfirmable_version_not_terminal(self):
        result = self.commit("revise", summary="更新后的模拟目标",
                             overview="由 Agent 动态核查实验来源")
        self.assertEqual(result, {"status": "awaiting_confirmation",
                                  "revision": 1, "contract_revision": 2})
        draft = self.store.contract(self.run_id, "DOC-001")
        self.assertEqual(draft["summary"], "更新后的模拟目标")
        self.assertEqual(draft["revision"], 2)
        self.assertEqual(self.commit("confirm", revision=1, contract_revision=2)["status"],
                         "contract_confirmed")

    def test_revise_replay_after_business_commit_does_not_duplicate_draft(self):
        kw = {"summary": "更新目标", "overview": "更新调查概要"}
        self.assertEqual(self.commit("revise", **kw),
                         self.commit("revise", **kw))
        self.assertEqual(self.store.contract_count(self.run_id), 2)
        self.assertEqual(self.store.operation_count(self.run_id), 1)

    def test_reused_operation_id_with_different_revise_payload_rejected(self):
        self.commit("revise", summary="原目标", overview="原概要")
        with self.assertRaises(WorkflowConflict):
            self.commit("revise", summary="伪造变更", overview="原概要")
        self.assertEqual(self.store.contract_count(self.run_id), 2)

    def test_stale_contract_after_revise_rejected(self):
        self.commit("revise", summary="更新目标", overview="更新概要")
        with self.assertRaises(WorkflowConflict):
            self.commit("confirm", revision=1, contract_revision=1)

    def test_clarification_blocks_until_answered(self):
        other = "run_" + "2" * 32
        self.store.create(other, "DOC-001", "P001", ask_clarification=True)
        def act(decision, revision, contract_revision, **kwargs):
            return self.store.apply(
                run_id=other, actor_id="DOC-001", phase="contract",
                decision=decision, expected_revision=revision,
                contract_revision=contract_revision,
                op_id=operation_id(other, "contract", revision), **kwargs
            )
        with self.assertRaises(WorkflowConflict):
            act("confirm", 0, 1)
        answered = act("clarify", 0, 1, answer="可以，但明确保留缺口")
        self.assertEqual(answered["contract_revision"], 2)
        self.assertIsNone(self.store.contract(other, "DOC-001")["clarification_question"])
        self.assertEqual(self.store.contract(other, "DOC-001")["clarification_answer"],
                         "可以，但明确保留缺口")
        self.assertEqual(act("confirm", 1, 2)["status"], "contract_confirmed")

    def test_invalid_edits_fail_atomically(self):
        for kwargs in ({"summary": "", "overview": "x"},
                       {"summary": "x", "overview": ""},
                       {"summary": "x" * 1201, "overview": "ok"}):
            with self.assertRaises(WorkflowConflict):
                self.commit("revise", **kwargs)
        with self.assertRaises(WorkflowConflict):
            self.commit("clarify", answer="answer without question")
        self.assertEqual(self.store.operation_count(self.run_id), 0)
        self.assertEqual(self.store.contract_count(self.run_id), 1)

    def test_cancel_is_terminal(self):
        self.assertEqual(self.commit("cancel")["status"], "cancelled")
        with self.assertRaises(WorkflowConflict):
            self.commit("confirm", revision=1, contract_revision=1)

    def test_wrong_actor_and_forged_operation_rejected(self):
        with self.assertRaises(WorkflowConflict):
            self.store.get(self.run_id, "DOC-999")
        with self.assertRaises(WorkflowConflict):
            self.store.apply(
                run_id=self.run_id, actor_id="DOC-999", phase="contract",
                decision="confirm", op_id=operation_id(self.run_id, "contract", 0),
                expected_revision=0, contract_revision=1
            )
        with self.assertRaises(WorkflowConflict):
            self.store.apply(
                run_id=self.run_id, actor_id="DOC-001", phase="contract",
                decision="confirm", op_id="forged",
                expected_revision=0, contract_revision=1
            )


if __name__ == "__main__":
    unittest.main()
