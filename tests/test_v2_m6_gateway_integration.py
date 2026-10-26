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
from v2_m5.catalog import SECTION_PLAN, paragraph_text
from v2_m5.grounding import review_document
from v2_m6.service import M6ReliableGatewayWorkflow, proposal_key
from v2_m7_trace import ReplayTrace


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
    if len(sys.argv) not in (2, 4) or (
        len(sys.argv) == 4 and sys.argv[2] != "--trace-out"
    ):
        raise SystemExit(
            "usage: test_v2_m6_gateway_integration.py SERVER "
            "[--trace-out PATH]"
        )
    trace_path = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    trace = ReplayTrace()
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

                def ready_run(scenario):
                    run = m1.start(
                        "m6_" + uuid.uuid4().hex,
                        "P001 SIM 报告 Writer 宕机恢复与来源新鲜度",
                        "先确认、调查、接收 LIS 更正，再准备未签发受限报告",
                    )["run_id"]
                    assert m1.resume_confirm(run, 0, 1)["business"] == "contract_confirmed"
                    trace.record(scenario, "contract_confirmed",
                                 metrics={"contract_revision": 1})
                    repo = GatewayArtifactRepository(
                        transport, doctor.issue_capability(run))
                    investigated = M3GatewayWorkflow(tmp, repo).execute(run, 1)
                    assert investigated["phase"] == "draft_ready"
                    assert {i["issue_id"] for i in investigated["agenda"]} == {
                        "I-01", "I-02", "I-03", "I-04"}
                    trace.record(scenario, "investigation_complete",
                                 refs=["I-01", "I-02", "I-03", "I-04",
                                       "E-002", "E-003"],
                                 metrics={"steps": investigated["steps"],
                                          "tool_calls": investigated["tool_calls"]})
                    correction = M4GatewayWorkflow(tmp, doctor).execute(
                        run, explicit_simulated_lis_event=True)
                    assert correction["correction_event_id"] == "EV-001"
                    assert correction["current_source"] == "E-005"
                    trace.record(scenario, "source_corrected",
                                 refs=["EV-001", "E-004", "E-005", "S-04",
                                       "S-05", "I-05"])
                    return run, repo

                # A: MySQL stage COMMIT → Graph has *no* checkpoint. A new
                # worker must consume persisted draft, not generate another.
                run, repo = ready_run("late_after_delivery")
                ctx = P001ReportContext(run, 1, repo, doctor).load()
                document = verified_document(ctx)
                key = proposal_key(run, 1)
                stage = repo.stage_m6_proposal(run, key, document)
                assert stage["document"] == document
                assert stage["status"] == "staged_untrusted"
                assert repo.get_m6_proposal(run, key) == stage
                trace.record("late_after_delivery", "proposal_staged",
                             metrics={"proposal_rows": 1})
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
                assert published["document"]["grounding"]["total_assertions"] == 10
                trace.record("late_after_delivery", "draft_delivered",
                             refs=["P-02", "E-002", "E-003", "E-005"],
                             metrics={"report_revision": 1, "assertions": 10,
                                      "sections": 6})
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
                explanation = doctor_session.explain(
                    run, "m5_" + run, base_revision=1,
                    paragraph_id="P-02", conversation_id="cv_m6_before",
                )
                assert explanation["created_revision"] is None
                assert len(explanation["assertions"]) == 3
                trace.record("late_after_delivery", "explain_readonly",
                             refs=["P-02"], metrics={"report_revision": 1})
                edit_one = doctor_session.shorten(
                    run, "m5_" + run, base_revision=1,
                    paragraph_id="P-02", conversation_id="cv_m6_before",
                    request_key="m7a_shorten_one")
                assert edit_one["revision"] == 2
                doc2 = doctor.m5_revision(run, "m5_" + run, 2)["document"]
                assert doc2["grounding"] == document["grounding"]
                trace.record("late_after_delivery", "draft_revised",
                             refs=["P-02"], metrics={"report_revision": 2})
                edit_two = doctor_session.shorten(
                    run, "m5_" + run, base_revision=2,
                    paragraph_id="P-05", conversation_id="cv_m6_before",
                    request_key="m7a_shorten_two")
                assert edit_two["revision"] == 3
                doc3 = doctor.m5_revision(run, "m5_" + run, 3)["document"]
                assert doc3["sections"][1] == doc2["sections"][1]
                assert doc3["sections"][4] != doc2["sections"][4]
                trace.record("late_after_delivery", "draft_revised",
                             refs=["P-05"], metrics={"report_revision": 3})
                # Build a valid revision-4 patch before EV-002, so rejecting
                # it after EV-002 proves source freshness (not invalid CAS).
                proposals = [{
                    "section_id": section["section_id"],
                    "paragraph_id": p["paragraph_id"],
                    "text": p["text"],
                    "fact_ids": list(p["fact_ids"]),
                    "candidate_evidence_refs": list(p["candidate_evidence_refs"]),
                } for section in doc3["sections"] for p in section["paragraphs"]]
                proposals[2]["text"] = paragraph_text(SECTION_PLAN[2][2], "short")
                reviewed, grounding = review_document(proposals, ctx)
                valid_patch = deepcopy(doc3)
                valid_patch.update({
                    "draft_revision": 4,
                    "writer_kind": "sim_deterministic_patch",
                    "sections": reviewed,
                    "grounding": grounding,
                })
                # Only this doctor's signed access JWT may trigger fixed EV-002.
                expect_gateway_error(lambda: transport.request(
                    "POST",doctor._prefix()+f"/{run}/m6/late-correction",
                    repo.capability, {}, run_id=run),401,403)
                expect_gateway_error(lambda: transport.request(
                    "POST",doctor._prefix()+f"/{run}/m6/late-correction",
                    doctor.doctor_jwt, {"hb_g_l":999}),400,422)
                # Late authorized SIM LIS update AFTER report delivery.
                late = doctor.m6_apply_late_sim_correction(run)
                assert late["event_id"] == "EV-002"
                assert doctor.m6_apply_late_sim_correction(run) == late
                assert late["previous_hb_g_l"] == 130
                assert late["current_hb_g_l"] == 125
                trace.record("late_after_delivery", "late_source_corrected",
                             refs=["EV-002", "E-005", "E-006", "S-05",
                                   "S-06", "I-06"])
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
                for revision, original in ((1, document), (2, doc2), (3, doc3)):
                    history = doctor.m5_revision(run, "m5_" + run, revision)
                    assert history["document"] == original
                    assert history["source_freshness"] == "stale_requires_investigation"
                    assert history["signing_eligible"] is False
                trace.record("late_after_delivery", "historical_stale",
                             refs=["E-005", "E-006", "I-06"],
                             metrics={"report_revision": 3})
                try:
                    workflow.execute(run, 1)
                except WorkflowConflict:
                    pass
                else:
                    raise AssertionError("stale cached checkpoint delivered as current")
                expect_gateway_error(
                    lambda: doctor_session.explain(
                        run, "m5_" + run, base_revision=3,
                        paragraph_id="P-02", conversation_id="cv_m6_after"), 409)
                expect_gateway_error(
                    lambda: doctor.m5_message(
                        run, "m5_" + run,
                        conversation_id="cv_m6_after", base_revision=3,
                        paragraph_id="P-03", intent="shorten",
                        operation_id=operation_id(run,"m6_stale_patch"),
                        document=valid_patch),409)
                # Python Coverage may reject this *before* an HTTP write.
                # A direct doctor endpoint is independently protected by
                # C++ current-source SQL checks.
                try:
                    doctor_session.shorten(
                        run, "m5_" + run, base_revision=3,
                        paragraph_id="P-03", conversation_id="cv_m6_after",
                        request_key="m6_after")
                except WorkflowConflict:
                    pass
                else:
                    raise AssertionError("stale-source paragraph patch was permitted")
                # M6 does not rewrite a clinically obsolete draft with 125;
                # it blocks reuse and requests new authorized investigation.
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                    f"WHERE run_id='{run}'") == "3"
                trace.record("late_after_delivery", "edit_rejected",
                             refs=["P-03", "E-005"],
                             metrics={"report_revision": 3})

                # B: source changes between SQL proposal stage and publish.
                run2, repo2 = ready_run("correction_before_publish")
                cached = verified_document(
                    P001ReportContext(run2, 1, repo2, doctor).load())
                repo2.stage_m6_proposal(run2, proposal_key(run2,1), cached)
                trace.record("correction_before_publish", "proposal_staged")
                late2 = doctor.m6_apply_late_sim_correction(run2)
                assert late2["event_id"] == "EV-002"
                trace.record("correction_before_publish", "late_source_corrected",
                             refs=["EV-002", "E-006"])
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
                trace.record("correction_before_publish", "publish_rejected",
                             refs=["E-005", "E-006"], metrics={"report_rows": 0})
                expect_gateway_error(
                    lambda: other.m6_apply_late_sim_correction(run2), 403, 404)

                # C: business publish COMMIT BEFORE Graph Checkpoint.
                run3, repo3 = ready_run("report_commit_before_checkpoint")
                proposal = verified_document(
                    P001ReportContext(run3,1,repo3,doctor).load())
                key3 = proposal_key(run3, 1)
                repo3.stage_m6_proposal(run3,key3,proposal)
                trace.record("report_commit_before_checkpoint", "proposal_staged")
                receipt = repo3.publish_m6_proposal(
                    run3,key3,operation_id(run3,"m5","publish","1"))
                assert receipt["revision"] == 1
                trace.record("report_commit_before_checkpoint", "draft_delivered",
                             metrics={"report_revision": 1})
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
                trace.record("report_commit_before_checkpoint",
                             "checkpoint_recovered", metrics={"report_rows": 1})
                # D: normal fresh end-to-end generation (no pre-staging),
                # followed by a new service instance with a Writer that
                # cannot be called successfully. Exercises the normal branch.
                run4, repo4 = ready_run("fresh_generation_replay")
                writer = CountingWriter()
                first_generated = M6ReliableGatewayWorkflow(
                    tmp, repo4, doctor, writer=writer,
                    postgres_dsn=postgres).execute(run4, 1)
                assert first_generated["source_freshness"] == "current"
                assert writer.count == 1
                trace.record("fresh_generation_replay", "draft_delivered",
                             metrics={"report_revision": 1, "writer_calls": 1})
                with tempfile.TemporaryDirectory() as other_directory:
                    replayed_from_ref = M6ReliableGatewayWorkflow(
                        other_directory, repo4, doctor,
                        writer=NeverCalledWriter(),
                        postgres_dsn=postgres).execute(run4,1)
                    assert replayed_from_ref["document"] == first_generated["document"]
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m6_writer_proposals "
                             f"WHERE run_id='{run4}'") == "1"
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                             f"WHERE run_id='{run4}'") == "1"
                trace.record("fresh_generation_replay", "checkpoint_recovered",
                             metrics={"writer_calls": 1, "report_rows": 1})
                # Actor is re-authorized even when the graph has fully finished.
                mysql("UPDATE treesem_doctor_patient_assignments "
                      "SET status='revoked',revoked_at=UTC_TIMESTAMP(6) "
                      "WHERE assignment_id='m2_asn'")
                expect_gateway_error(
                    lambda: doctor.m5_revision(run3,"m5_"+run3,1),403,404)
                trace.record("fresh_generation_replay", "access_revoked")
                trace.finish()
                if trace_path is not None:
                    trace.write(trace_path)
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
