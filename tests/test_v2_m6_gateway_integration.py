#!/usr/bin/env python3
"""M6 synthetic C++/MySQL source freshness + optional official PG Saver E2E."""
from __future__ import annotations
from copy import deepcopy
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
from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import (
    GatewayArtifactRepository, GatewayDoctorClient, GatewayTransport,
)
from v2_m2.gateway_workflow import GatewayContractWorkflow
from v2_m3.service import M3GatewayWorkflow
from v2_m4.service import M4GatewayWorkflow
from v2_m5.coverage import P001ReportContext
from v2_m5.service import M5DoctorSession, operation_id, verified_document
from v2_m5.writer import FakeParagraphWriter
from v2_m6.service import M6ReliableGatewayWorkflow, proposal_key


class NeverCalledWriter:
    def write(self, context):
        raise AssertionError("Writer was rerun after MySQL already had its output")


class CountingWriter(FakeParagraphWriter):
    def __init__(self): self.count = 0
    def write(self, context):
        self.count += 1
        if self.count > 1:
            raise AssertionError("Writer output would differ after retry")
        return super().write(context)


def main():
    assert len(sys.argv) == 2
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
        "TREESEM_AGENT_SERVICE_SECRET": "m6-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m6-knowledge-secret-at-least-32-bytes",
        "TREESEM_CAPABILITY_TOKEN_TTL_SECONDS": "120",
        "TREESEM_DB_POOL_SIZE": "3",
        "TREESEM_DB_HOST": "127.0.0.1",
        "TREESEM_DB_PORT": os.environ.get("TREESEM_DB_PORT", "3306"),
        "TREESEM_DB_NAME": os.environ.get("TREESEM_DB_NAME", "treesem_test"),
        "TREESEM_DB_USER": os.environ.get("TREESEM_DB_USER", "treesem_app"),
        "TREESEM_DB_PASSWORD": os.environ["TREESEM_DB_PASSWORD"],
    }
    postgres = os.environ.get("TREESEM_M6_POSTGRES_DSN")
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "server.log", "w") as log:
            server = subprocess.Popen(
                [str(Path(sys.argv[1]).resolve()), str(port)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            try:
                wait_healthy(port, server)
                transport = GatewayTransport(f"http://127.0.0.1:{port}")
                doctor = GatewayDoctorClient(
                    transport, jwt_token(DOC, "doctor"), PAT)
                other = GatewayDoctorClient(
                    transport, jwt_token(OTHER_DOC, "doctor"), OTHER_PAT)
                m1 = GatewayContractWorkflow(tmp, doctor)

                def ready_run():
                    run = m1.start(
                        "m6_" + uuid.uuid4().hex,
                        "P001 SIM 报告 Writer 宕机恢复与来源新鲜度",
                        "先确认、调查、接收 LIS 更正，再准备未签发受限报告",
                    )["run_id"]
                    assert m1.resume_confirm(run, 0, 1)["business"] == "contract_confirmed"
                    repo = GatewayArtifactRepository(
                        transport, doctor.issue_capability(run))
                    M3GatewayWorkflow(tmp, repo).execute(run, 1)
                    M4GatewayWorkflow(tmp, doctor).execute(
                        run, explicit_simulated_lis_event=True)
                    return run, repo

                # A: MySQL stage COMMIT → Graph has *no* checkpoint. A new
                # worker must consume persisted draft, not generate another.
                run, repo = ready_run()
                ctx = P001ReportContext(run, 1, repo, doctor).load()
                document = verified_document(ctx)
                key = proposal_key(run, 1)
                stage = repo.stage_m6_proposal(run, key, document)
                assert stage["document"] == document
                assert stage["status"] == "staged_untrusted"
                assert repo.get_m6_proposal(run, key) == stage
                malicious = deepcopy(document)
                malicious["sections"][1]["paragraphs"][0]["text"] = "未经证据确认的诊断。"
                expect_gateway_error(
                    lambda: repo.stage_m6_proposal(run, key, malicious), 409)
                workflow = M6ReliableGatewayWorkflow(
                    tmp, repo, doctor, writer=NeverCalledWriter(),
                    postgres_dsn=postgres)
                published = workflow.execute(run, 1)
                assert published["report_id"] == "m5_" + run
                assert published["revision"] == 1
                assert published["source_freshness"] == "current"
                assert published["signing_eligible"] is False
                assert published["document"] == document
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m6_writer_proposals "
                    f"WHERE run_id='{run}'") == "1"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                    f"WHERE run_id='{run}'") == "1"
                # Different local directory: only the DB proposal and shared
                # official PG checkpoint (when configured) are relied upon.
                with tempfile.TemporaryDirectory() as another:
                    again = M6ReliableGatewayWorkflow(
                        another, repo, doctor, writer=NeverCalledWriter(),
                        postgres_dsn=postgres)
                    assert again.execute(run, 1)["document"] == document
                # Read-only explanation and edit were valid before EV-002.
                doctor_session = M5DoctorSession(doctor)
                assert doctor_session.explain(
                    run, "m5_" + run, base_revision=1,
                    paragraph_id="P-02", conversation_id="cv_m6_before",
                )["created_revision"] is None
                # Late authorized SIM LIS update AFTER report delivery.
                late = doctor.m6_apply_late_sim_correction(run)
                assert late["event_id"] == "EV-002"
                assert doctor.m6_apply_late_sim_correction(run) == late
                assert late["previous_hb_g_l"] == 130
                assert late["current_hb_g_l"] == 125
                overlay = doctor.m4_state(run)
                sources = {x["evidence_id"]: x for x in overlay["source_versions"]}
                assert sources["E-004"]["validity"] == "stale"
                assert sources["E-005"]["validity"] == "stale"
                assert sources["E-006"]["validity"] == "current"
                assert sources["E-006"]["hb_g_l"] == 125
                statements = {x["statement_id"]: x for x in overlay["statements"]}
                assert statements["S-05"]["validity"] == "stale"
                assert statements["S-06"]["validity"] == "current"
                assert {x["issue_id"]:x["status"] for x in overlay["followup_issues"]}[
                    "I-06"] == "needs_review"
                assert doctor.validate_m4_current(
                    run, ["E-005"])["admissible"] is False
                assert doctor.validate_m4_current(
                    run, ["E-006"])["admissible"] is True
                assert doctor.validate_m4_current(
                    run, ["E-002","E-003"])["admissible"] is True
                history = doctor.m5_revision(run, "m5_" + run, 1)
                assert history["document"] == document  # immutable old Draft
                assert history["source_freshness"] == "stale_requires_investigation"
                assert history["signing_eligible"] is False
                try:
                    workflow.execute(run, 1)
                except WorkflowConflict:
                    pass
                else:
                    raise AssertionError("stale cached checkpoint delivered as current")
                expect_gateway_error(
                    lambda: doctor_session.explain(
                        run, "m5_" + run, base_revision=1,
                        paragraph_id="P-02", conversation_id="cv_m6_after"), 409)
                # Python Coverage may reject this *before* an HTTP write.
                # A direct doctor endpoint is independently protected by
                # C++ current-source SQL checks.
                try:
                    doctor_session.shorten(
                        run, "m5_" + run, base_revision=1,
                        paragraph_id="P-02", conversation_id="cv_m6_after",
                        request_key="m6_after")
                except WorkflowConflict:
                    pass
                else:
                    raise AssertionError("stale-source paragraph patch was permitted")
                # M6 does not rewrite a clinically obsolete draft with 125;
                # it blocks reuse and requests new authorized investigation.
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                    f"WHERE run_id='{run}'") == "1"

                # B: source changes between SQL proposal stage and publish.
                run2, repo2 = ready_run()
                cached = verified_document(
                    P001ReportContext(run2, 1, repo2, doctor).load())
                repo2.stage_m6_proposal(run2, proposal_key(run2,1), cached)
                doctor.m6_apply_late_sim_correction(run2)
                expect_gateway_error(
                    lambda: repo2.publish_m6_proposal(
                        run2, proposal_key(run2,1),
                        operation_id(run2,"m5","publish","1")), 409)
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_reports "
                    f"WHERE run_id='{run2}'") == "0"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m6_writer_proposals "
                    f"WHERE run_id='{run2}'") == "1"
                expect_gateway_error(
                    lambda: other.m6_apply_late_sim_correction(run2), 403, 404)

                # C: business publish COMMIT BEFORE Graph Checkpoint.
                run3, repo3 = ready_run()
                proposal = verified_document(
                    P001ReportContext(run3,1,repo3,doctor).load())
                key3 = proposal_key(run3, 1)
                repo3.stage_m6_proposal(run3,key3,proposal)
                receipt = repo3.publish_m6_proposal(
                    run3,key3,operation_id(run3,"m5","publish","1"))
                assert receipt["revision"] == 1
                with tempfile.TemporaryDirectory() as newest:
                    assert M6ReliableGatewayWorkflow(
                        newest,repo3,doctor,writer=NeverCalledWriter(),
                        postgres_dsn=postgres).execute(run3,1)["document"] == proposal
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_operations "
                    f"WHERE run_id='{run3}'") == "1"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                    f"WHERE run_id='{run3}'") == "1"
                # Actor is re-authorized even when the graph has fully finished.
                mysql("UPDATE treesem_doctor_patient_assignments "
                      "SET status='revoked',revoked_at=UTC_TIMESTAMP(6) "
                      "WHERE assignment_id='m2_asn'")
                expect_gateway_error(
                    lambda: doctor.m5_revision(run3,"m5_"+run3,1),403,404)
                print("M6 MySQL/Checkpoint integration passed: proposal recovery, "
                      "stale sources, old Draft audit, late source and replay")
            finally:
                if server.poll() is None:
                    server.terminate()
                    try: server.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=3)


if __name__ == "__main__":
    main()
