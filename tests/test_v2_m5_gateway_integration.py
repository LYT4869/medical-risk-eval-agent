#!/usr/bin/env python3
"""Real C++/MySQL M5 report and doctor revision integration; synthetic P001."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from test_v2_m2_gateway_integration import (
    ACCESS_SECRET,CAPABILITY_SECRET,DOC,OTHER_DOC,PAT,OTHER_PAT,
    free_port,jwt_token,mysql,seed_fake_users,wait_healthy,expect_gateway_error,
)
from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import GatewayArtifactRepository,GatewayDoctorClient,GatewayTransport
from v2_m2.gateway_workflow import GatewayContractWorkflow
from v2_m3.service import M3GatewayWorkflow
from v2_m4.service import M4GatewayWorkflow
from v2_m5.coverage import P001ReportContext
from v2_m5.service import M5DoctorSession,M5GatewayWorkflow,operation_id,verified_document


def main():
    assert len(sys.argv)==2
    seed_fake_users()
    port=free_port()
    env={**os.environ,
        "TREESEM_STORAGE_BACKEND":"mysql","TREESEM_MODEL_BACKEND":"remote",
        "TREESEM_AGENT_ENABLED":"false","TREESEM_AUTH_MODE":"required",
        "TREESEM_V2_SIM_ARTIFACT_API":"1","TREESEM_ACCESS_JWT_SECRET":ACCESS_SECRET,
        "TREESEM_CAPABILITY_JWT_SECRET":CAPABILITY_SECRET,
        "TREESEM_AGENT_SERVICE_SECRET":"m5-test-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET":"m5-test-knowledge-secret-at-least-32-bytes",
        "TREESEM_CAPABILITY_TOKEN_TTL_SECONDS":"120","TREESEM_DB_POOL_SIZE":"3",
        "TREESEM_DB_HOST":"127.0.0.1",
        "TREESEM_DB_PORT":os.environ.get("TREESEM_DB_PORT","3306"),
        "TREESEM_DB_NAME":os.environ.get("TREESEM_DB_NAME","treesem_test"),
        "TREESEM_DB_USER":os.environ.get("TREESEM_DB_USER","treesem_app"),
        "TREESEM_DB_PASSWORD":os.environ["TREESEM_DB_PASSWORD"]}
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp)/"backend.log","w") as log:
            proc=subprocess.Popen([str(Path(sys.argv[1]).resolve()),str(port)],
                                  stdout=log,stderr=subprocess.STDOUT,env=env)
            try:
                wait_healthy(port,proc)
                transport=GatewayTransport(f"http://127.0.0.1:{port}")
                doctor=GatewayDoctorClient(transport,jwt_token(DOC,"doctor"),PAT)
                other=GatewayDoctorClient(transport,jwt_token(OTHER_DOC,"doctor"),OTHER_PAT)
                m1=GatewayContractWorkflow(tmp,doctor)
                run=m1.start("m5_"+uuid.uuid4().hex,"合成报告与来源核验",
                             "医生确认后动态调查，保留异源差异和LIS版本")["run_id"]
                assert m1.resume_confirm(run,0,1)["business"]=="contract_confirmed"
                repo=GatewayArtifactRepository(transport,doctor.issue_capability(run))
                workflow=M5GatewayWorkflow(tmp,repo,doctor)
                try: workflow.execute(run,1)
                except WorkflowConflict: pass
                else: raise AssertionError("M5 bypassed M3/M4 coverage")
                M3GatewayWorkflow(tmp,repo).execute(run,1)
                M4GatewayWorkflow(tmp,doctor).execute(
                    run,explicit_simulated_lis_event=True)
                first=workflow.execute(run,1)
                report_id="m5_"+run
                assert first["report_id"]==report_id
                assert first["revision"]==1
                doc1=first["document"]
                assert doc1["coverage"]["status"]=="restricted"
                assert doc1["grounding"]["total_sentences"]==6
                assert doc1["grounding"]["total_assertions"]==10
                assert doc1["grounding"]["unsupported_assertions"]==0
                assert len(doc1["sections"])==6
                assert doc1["not_signed"] and doc1["clinical_unverified"]
                assert doc1["model_score"] is None
                p02=doc1["sections"][1]["paragraphs"][0]
                assert len(p02["sentences"][0]["assertions"])==3
                assert [x["evidence_id"] for a in p02["sentences"][0]["assertions"]
                        for x in a["evidence_refs"]]==[
                            "E-002","E-003","E-002","E-003"]
                assert workflow.execute(run,1)["document"]==doc1
                session=M5DoctorSession(doctor)
                cv="cv_001"
                explanation=session.explain(
                    run,report_id,base_revision=1,paragraph_id="P-02",
                    conversation_id=cv)
                assert explanation["created_revision"] is None
                assert len(explanation["assertions"])==3
                changed=session.shorten(
                    run,report_id,base_revision=1,paragraph_id="P-02",
                    conversation_id=cv,request_key="shorten_one")
                assert changed["revision"]==2
                doc2=doctor.m5_revision(run,report_id,2)["document"]
                assert len(doc2["sections"][1]["paragraphs"][0]["text"])<len(p02["text"])
                assert len(doc2["sections"][1]["paragraphs"][0]
                           ["sentences"][0]["assertions"])==3
                assert doc2["grounding"]==doc1["grounding"]
                for i in (0,2,3,4,5):
                    assert doc1["sections"][i]==doc2["sections"][i]
                op=operation_id(run,report_id,cv,"1","P-02","shorten_one")
                assert doctor.m5_message(
                    run,report_id,conversation_id=cv,base_revision=1,
                    paragraph_id="P-02",intent="shorten",
                    operation_id=op,document=doc2)==changed
                expect_gateway_error(
                    lambda:doctor.m5_message(
                        run,report_id,conversation_id=cv,base_revision=1,
                        paragraph_id="P-02",intent="shorten",
                        operation_id=operation_id(run,"another"),document=doc2),409)
                revised=session.shorten(
                    run,report_id,base_revision=2,paragraph_id="P-05",
                    conversation_id=cv,request_key="shorten_two")
                assert revised["revision"]==3
                doc3=doctor.m5_revision(run,report_id,3)["document"]
                assert doc3["sections"][1]==doc2["sections"][1]
                assert doc3["sections"][4]!=doc2["sections"][4]
                assert session.explain(
                    run,report_id,base_revision=3,paragraph_id="P-05",
                    conversation_id=cv)["created_revision"] is None
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                             f"WHERE run_id='{run}'")=="3"
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m5_operations "
                             f"WHERE run_id='{run}'")=="3"
                expect_gateway_error(lambda:other.m5_revision(run,report_id,1),403,404)
                expect_gateway_error(lambda:doctor.m5_revision(run,"m5_wrong",1),404)
                false=deepcopy(doc1)
                false["sections"][1]["paragraphs"][0]["text"]="未被证据支持的诊断结论。"
                expect_gateway_error(
                    lambda:repo.publish_m5(run,"op_invalid_content",false),400,422)
                expect_gateway_error(lambda:transport.request(
                    "POST",repo._prefix(run)+"/artifacts/m5/publish",
                    doctor.doctor_jwt,{"operation_id":"op_wrong_token","document":doc1},
                    run_id=run),401,403)
                # Wrong token class cannot cross the internal/public boundary.
                expect_gateway_error(lambda: transport.request(
                    "POST",doctor._prefix()+f"/{run}/m5/reports/{report_id}/messages",
                    repo.capability,
                    {"intent":"explain","base_revision":3,"conversation_id":cv,
                     "paragraph_id":"P-02"},run_id=run),401,403)

                # MySQL COMMIT then no M5 LangGraph checkpoint: a *second*
                # explicitly confirmed Report/Run uses identical stable op.
                run2=m1.start(
                    "m5_"+uuid.uuid4().hex,"第二份独立P001合成报告",
                    "与第一份报告隔离，不自动复用报告身份"
                )["run_id"]
                m1.resume_confirm(run2,0,1)
                repo2=GatewayArtifactRepository(
                    transport,doctor.issue_capability(run2))
                M3GatewayWorkflow(tmp,repo2).execute(run2,1)
                M4GatewayWorkflow(tmp,doctor).execute(
                    run2,explicit_simulated_lis_event=True)
                staged=verified_document(
                    P001ReportContext(run2,1,repo2,doctor).load())
                precommit_op=operation_id(run2,"m5","publish","1")
                repo2.publish_m5(run2,precommit_op,staged)
                replayed=M5GatewayWorkflow(tmp,repo2,doctor).execute(run2,1)
                assert replayed["document"]==staged
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m5_report_revisions "
                             f"WHERE run_id='{run2}'")=="1"
                assert mysql("SELECT COUNT(*) FROM treesem_v2_m5_operations "
                             f"WHERE run_id='{run2}'")=="1"
                expect_gateway_error(
                    lambda:doctor.m5_revision(run2,report_id,1),404)
                # Live patient authorization rechecked on every subsequent
                # report read or explanation, even if the old token remains.
                mysql("UPDATE treesem_doctor_patient_assignments "
                      "SET status='revoked',revoked_at=UTC_TIMESTAMP(6) "
                      "WHERE assignment_id='m2_asn'")
                expect_gateway_error(
                    lambda:doctor.m5_revision(run,report_id,3),403,404)
                expect_gateway_error(
                    lambda:session.explain(run,report_id,base_revision=3,
                        paragraph_id="P-02",conversation_id=cv),403,404)
                print("M5 MySQL: Coverage/Writer/10 assertions and doctor v1-v3 CAS passed")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=3)


if __name__=="__main__":
    main()
