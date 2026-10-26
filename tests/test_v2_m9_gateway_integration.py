#!/usr/bin/env python3
"""M9 genuine C++ JWT/Run-capability + MySQL8 dynamic candidate storage E2E.

Only P001 SYNTHETIC fixture. No LLM, no clinical database or report publishing.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from test_v2_m2_gateway_integration import (
    ACCESS_SECRET, CAPABILITY_SECRET, DOC, OTHER_DOC, PAT, OTHER_PAT,
    free_port, jwt_token, mysql, seed_fake_users, wait_healthy,
    expect_gateway_error,
)
from v2_m2.gateway_adapter import (
    GatewayArtifactRepository, GatewayDoctorClient, GatewayTransport,
)
from v2_m2.gateway_workflow import GatewayContractWorkflow
from v2_m8.fake_agent import ScriptedInvestigator
from v2_m8.workflow import M8GatewayWorkflow


class CrashAfterFirstIssueCommit:
    def __init__(self, actual):
        self.actual = actual
        self.failed = False

    def save(self, run_id, kind, key, payload):
        result = self.actual.save(run_id, kind, key, payload)
        if kind == "issue" and not self.failed:
            self.failed = True
            raise RuntimeError("M9 simulated process crash after issue commit")
        return result

    def get(self, *args):
        return self.actual.get(*args)


class ForbiddenModel:
    def plan_task(self, ctx):
        raise AssertionError("never regenerate persisted Agenda")

    def decide(self, ctx):
        raise AssertionError("never run a model for completed checkpoint")

    def consolidate(self, ctx):
        raise AssertionError("never regenerate committed IssueResult")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_v2_m9_gateway_integration.py SERVER")
    # This script also runs after M6 in the same integration CI database.
    if mysql("SELECT COUNT(*) FROM treesem_users "
             "WHERE user_id='m2_doc'") == "0":
        seed_fake_users()
    port = free_port()
    env = {
        **os.environ,
        "TREESEM_STORAGE_BACKEND": "mysql",
        "TREESEM_MODEL_BACKEND": "remote",
        "TREESEM_AGENT_ENABLED": "false",
        "TREESEM_AUTH_MODE": "required",
        "TREESEM_V2_SIM_ARTIFACT_API": "1",
        "TREESEM_ACCESS_JWT_SECRET": ACCESS_SECRET,
        "TREESEM_CAPABILITY_JWT_SECRET": CAPABILITY_SECRET,
        "TREESEM_AGENT_SERVICE_SECRET": "m9-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m9-knowledge-secret-at-least-32-bytes",
        "TREESEM_CAPABILITY_TOKEN_TTL_SECONDS": "120",
        "TREESEM_DB_POOL_SIZE": "3",
        "TREESEM_DB_HOST": "127.0.0.1",
        "TREESEM_DB_PORT": os.environ.get("TREESEM_DB_PORT", "3306"),
        "TREESEM_DB_NAME": os.environ.get("TREESEM_DB_NAME", "treesem_test"),
        "TREESEM_DB_USER": os.environ.get("TREESEM_DB_USER", "treesem_app"),
        "TREESEM_DB_PASSWORD": os.environ["TREESEM_DB_PASSWORD"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "server.log", "w") as logfile:
            process = subprocess.Popen(
                [str(Path(sys.argv[1]).resolve()), str(port)],
                stdout=logfile, stderr=subprocess.STDOUT, env=env,
            )
            try:
                wait_healthy(port, process)
                transport = GatewayTransport(f"http://127.0.0.1:{port}")
                doctor = GatewayDoctorClient(
                    transport, jwt_token(DOC, "doctor"), PAT)
                other = GatewayDoctorClient(
                    transport, jwt_token(OTHER_DOC, "doctor"), OTHER_PAT)
                m1 = GatewayContractWorkflow(tmp, doctor)

                def create_ready(tag):
                    run = m1.start(
                        "m9_" + uuid.uuid4().hex,
                        "P001 SIM：核查原始病历与转诊摘要的差异 " + tag,
                        "先确认再调查；不生成未经证据审查的报告",
                    )["run_id"]
                    expect_gateway_error(lambda: doctor.issue_capability(run), 409)
                    assert m1.resume_confirm(run, 0, 1)["business"] == "contract_confirmed"
                    token = doctor.issue_capability(run)
                    return run, GatewayArtifactRepository(transport, token)

                run, gateway = create_ready("happy")
                agent = ScriptedInvestigator()
                graph = M8GatewayWorkflow(tmp, gateway, agent=agent)
                out = graph.execute(run, 1)
                assert out["phase"] == "complete", out
                assert out["outcome"] == "restricted_unverified", out
                assert [x["issue_id"] for x in out["agenda"]] == [
                    "I-01", "I-02", "I-03",
                ], out
                assert out["seen_evidence"] == ["E-002", "E-003"]
                assert out["tool_calls"] == 2
                assert agent.consolidation_calls == 3
                result = graph.repo.get(run, "issue", "I-03:r1")
                assert result["status"] == "closed_unverified"
                assert result["semantic_validation"] == "not_evaluated"
                assert result["disposition"] == "unresolved"
                assert result["evidence_ids"] == ["E-002", "E-003"]
                seed = graph.repo.get(run, "agenda_seed", "contract-r1")
                assert seed["contract_summary"] == doctor.status(run)["summary"]
                assert seed["agenda"][0]["issue_id"] == "I-01"

                # M8 types are separate SQL kinds: no M3 fixed Issue or Draft.
                counts = mysql("SELECT kind,COUNT(*) FROM treesem_v2_artifacts "
                               f"WHERE run_id='{run}' GROUP BY kind")
                assert "m8_seed" in counts and "m8_issue" in counts, counts
                assert mysql("SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}' AND kind IN ('issue','draft')") == "0"
                assert mysql("SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}'") == "6"
                assert doctor.status(run)["status"] == "contract_confirmed"

                # True cross-process Resume: zero rerun and no duplicate SQL.
                reopened = M8GatewayWorkflow(tmp, gateway, agent=ForbiddenModel())
                assert reopened.execute(run, 1) == out
                assert mysql("SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}'") == "6"

                # Exact replay returns the idempotent journal receipt.
                same = graph.repo.save(run, "issue", "I-03:r1", result)
                assert same["kind"] == "m8_issue"
                expect_gateway_error(
                    lambda: graph.repo.save(
                        run, "issue", "I-03:r1",
                        {**result, "unresolved_questions": ["changed"]}), 409)

                # A capable model cannot bypass existing M5/M6 admission by
                # putting an unverified candidate into an old verified kind.
                expect_gateway_error(
                    lambda: transport.request(
                        "POST", gateway._prefix(run) + "/artifacts",
                        gateway.capability,
                        {"operation_id": "m9_fake_legacy_issue",
                         "kind": "issue", "artifact_key": "I-99:r1",
                         "payload": result}, run_id=run), 409)
                expect_gateway_error(
                    lambda: transport.request(
                        "POST", gateway._prefix(run) + "/artifacts",
                        gateway.capability,
                        {"operation_id": "m9_fake_promoted_issue",
                         "kind": "m8_issue", "artifact_key": "I-99:r1",
                         "payload": {**result, "issue_id": "I-99",
                                     "status": "verified"}}, run_id=run), 400)
                source = graph.repo.get(run, "evidence",
                                        "EMR-P001-DEL-20220508:v1")
                expect_gateway_error(
                    lambda: graph.repo.save(run, "evidence",
                        "EMR-P001-DEL-20220508:v2",
                        {**source, "record": {**source["record"],
                                              "blood_loss_ml": 999}}), 400)

                # Scoped capability and run-specific Business Authority.
                run2, gateway2 = create_ready("different")
                expect_gateway_error(
                    lambda: graph.repo.get(run2, "issue", "I-03:r1"), 403, 404)
                expect_gateway_error(
                    lambda: other.status(run), 403, 404)
                expect_gateway_error(
                    lambda: gateway2.transport.request(
                        "GET", gateway2._prefix(run) +
                        "/artifacts/m8_issue/I-03:r1",
                        gateway2.capability, run_id=run), 403, 404)

                # SQL commit happened; graph didn't checkpoint. Resume must
                # rehydrate immutable candidate instead of rerunning LLM.
                crash_run, crash_gateway = create_ready("crash")
                first_agent = ScriptedInvestigator()
                first = M8GatewayWorkflow(tmp, crash_gateway, agent=first_agent)
                first.repo = CrashAfterFirstIssueCommit(first.repo)
                try:
                    first.execute(crash_run, 1)
                except RuntimeError as exc:
                    assert "after issue commit" in str(exc)
                else:
                    raise AssertionError("M9 fault injection did not fire")
                restored_agent = ScriptedInvestigator()
                restored = M8GatewayWorkflow(
                    tmp, crash_gateway, agent=restored_agent)
                recovered = restored.execute(crash_run, 1)
                assert recovered["phase"] == "complete"
                assert recovered["tool_calls"] == 2
                assert restored_agent.consolidation_calls == 2
                assert mysql("SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{crash_run}'") == "6"
                assert mysql("SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{crash_run}' AND kind='draft'") == "0"

                print("M9 REAL C++ JWT/MySQL: dynamic Agenda, IssueResult, "
                      "cross-Issue, replay, SQL crash, isolation and no draft PASS")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


if __name__ == "__main__":
    main()
