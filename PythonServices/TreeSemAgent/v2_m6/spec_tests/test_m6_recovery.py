"""M6 pure recovery contract tests. Real DB/PG tests are separate."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import GatewayError
from v2_m6.service import (
    proposal_key, prepare_proposal, _assert_staged, checkpoint_backend,
)


class Repo:
    def __init__(self, *, already=None, competing=False):
        self.value = already
        self.competing = competing
        self.get_count = 0
        self.stage_count = 0

    def get_m6_proposal(self, run_id, key):
        self.get_count += 1
        if self.value is None:
            raise GatewayError(404)
        return self.value

    def stage_m6_proposal(self, run_id, key, doc):
        self.stage_count += 1
        if self.competing:
            self.value = {
                "run_id": run_id, "proposal_key": key,
                "contract_revision": 1, "status": "staged_untrusted",
                "classification": "SIM",
                "document": {"report_id": "m5_" + run_id,
                             "contract_revision": 1, "different_writer": True},
            }
            raise GatewayError(409)
        self.value = {
            "run_id": run_id, "proposal_key": key,
            "contract_revision": 1, "status": "staged_untrusted",
            "classification": "SIM", "document": doc,
        }
        return self.value


class RecoveryContracts(TestCase):
    def setUp(self):
        self.run = "run_m6test"
        self.key = proposal_key(self.run, 1)
        self.ctx = SimpleNamespace(run_id=self.run, contract_revision=1)
        self.draft = {"report_id": "m5_" + self.run, "contract_revision": 1}

    def test_stable_keys_bind_the_run_and_contract(self):
        self.assertEqual(self.key, proposal_key(self.run, 1))
        self.assertNotEqual(self.key, proposal_key(self.run, 2))
        self.assertNotEqual(self.key, proposal_key("run_other", 1))

    def test_persist_first_candidate_once(self):
        repo = Repo()
        with patch("v2_m6.service.verified_document", return_value=self.draft) as gen:
            one = prepare_proposal(repo, self.ctx, object(), self.key)
            two = prepare_proposal(repo, self.ctx, object(), self.key)
        self.assertEqual(one, two)
        self.assertEqual(repo.stage_count, 1)
        gen.assert_called_once()

    def test_db_already_committed_skips_nondeterministic_writer(self):
        old = Repo()
        old.stage_m6_proposal(self.run, self.key, self.draft)
        with patch("v2_m6.service.verified_document",
                   side_effect=AssertionError("LLM was called again")):
            self.assertEqual(prepare_proposal(
                old, self.ctx, object(), self.key), old.value)
        self.assertEqual(old.stage_count, 1)

    def test_concurrent_writer_loser_reads_sql_winner_without_overwrite(self):
        repo = Repo(competing=True)
        with patch("v2_m6.service.verified_document",return_value=self.draft):
            result = prepare_proposal(repo, self.ctx, object(), self.key)
        self.assertTrue(result["document"]["different_writer"])
        self.assertEqual(repo.stage_count,1)

    def test_wrong_patient_run_cannot_reuse_candidate(self):
        bad = Repo(already={
            "run_id":"run_different", "proposal_key":self.key,
            "contract_revision":1, "status":"staged_untrusted",
            "classification":"SIM",
            "document":{"report_id":"m5_run_different","contract_revision":1},
        })
        with self.assertRaises(WorkflowConflict):
            prepare_proposal(bad, self.ctx, object(), self.key)

    def test_wrong_contract_or_publish_state_cannot_be_cached(self):
        for bad in (
            {"contract_revision":2},
            {"status":"clinical_signed"},
            {"classification":"REAL"},
        ):
            good={"run_id":self.run,"proposal_key":self.key,
                  "contract_revision":1,"status":"staged_untrusted",
                  "classification":"SIM","document":self.draft}
            good.update(bad)
            with self.subTest(change=bad), self.assertRaises(WorkflowConflict):
                _assert_staged(good,self.run,self.key,1)

    def test_no_unapproved_checkpointer_uri(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(WorkflowConflict):
                with checkpoint_backend(
                    str(Path(root)/"x.sqlite3"),"mysql://unsupported"):
                    pass

    def test_sqlite_saver_holds_only_structural_state(self):
        with tempfile.TemporaryDirectory() as root:
            with checkpoint_backend(str(Path(root)/"checkpoint.sqlite3")) as saver:
                self.assertIsNotNone(saver)


if __name__ == "__main__":
    import unittest
    unittest.main()
