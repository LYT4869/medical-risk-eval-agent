"""M8 opt-in runtime acceptance. No real LLM, real HIS or clinical NLI."""
from __future__ import annotations
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from v2_m8.contracts import initial_agenda, apply_delta, check_close
from v2_m8.context import context_for_issue, get_issue_result, get_evidence
from v2_m8.fake_agent import ScriptedInvestigator
from v2_m8.workflow import M8LocalWorkflow

OBJECTIVES={"O-01","O-02"}

def planned(question="核查原始记录"):
    return {"question":question,"category":"original_record",
            "priority":"normal","objective_refs":["O-01"],
            "trigger_evidence_refs":[],"related_issue_ids":[]}

class BrokenDecisions(ScriptedInvestigator):
    def decide(self,ctx):
        return {"action":"delete_patient"}

class BrokenClose(ScriptedInvestigator):
    def consolidate(self,ctx):
        result=super().consolidate(ctx)
        result["statements"]=[{"text":"无证据结论","evidence_refs":["E-999"],
                               "qualifiers":{}}]
        return result

class AfterCommitCrash:
    def __init__(self,actual):
        self.actual=actual
        self.failed=False
    def save(self,run_id,kind,key,payload):
        result=self.actual.save(run_id,kind,key,payload)
        if kind=="issue" and not self.failed:
            self.failed=True
            raise RuntimeError("simulated crash after immutable artifact commit")
        return result
    def get(self,*args):
        return self.actual.get(*args)

class Contracts(unittest.TestCase):
    def test_initial_issues_are_proposed_not_fixed(self):
        one=planned()
        two={**planned("核查转诊摘要"),"objective_refs":["O-02"],
             "category":"referral_review"}
        out=initial_agenda({"issues":[two,one]},OBJECTIVES)
        self.assertEqual([x["issue_id"] for x in out],["I-01","I-02"])
        self.assertEqual(out[0]["question"],"核查转诊摘要")
        with self.assertRaises(WorkflowConflict):
            initial_agenda({"issues":[one]},OBJECTIVES)
        with self.assertRaises(WorkflowConflict):
            initial_agenda({"issues":[one,deepcopy(one)]},{"O-01"})

    def test_atomic_agenda_delta_guards_and_stable_ids(self):
        agenda=initial_agenda({"issues":[
            planned(),
            {**planned("核查转诊摘要"),"objective_refs":["O-02"],
             "category":"referral_review"}]},OBJECTIVES)
        operation={"op":"add_issue","question":"调查跨来源差异",
                   "category":"source_discrepancy","priority":"high",
                   "objective_refs":["O-02"],
                   "trigger_evidence_refs":["E-002","E-003"],
                   "related_issue_ids":["I-01","I-02"]}
        proposal={"base_agenda_revision":2,"operations":[operation]}
        result,rev=apply_delta(agenda,proposal,2,OBJECTIVES,{"E-002","E-003"})
        self.assertEqual(rev,3)
        self.assertEqual(result[-1]["issue_id"],"I-03")
        self.assertEqual(result[-1]["evidence_ids"],["E-002","E-003"])
        self.assertEqual(len(agenda),2)
        for bad in (
            {**proposal,"base_agenda_revision":1},
            {**proposal,"operations":[{**operation,
                 "trigger_evidence_refs":["E-999"]}]},
            {**proposal,"operations":[operation,operation]},
        ):
            with self.assertRaises(WorkflowConflict):
                apply_delta(agenda,bad,2,OBJECTIVES,{"E-002","E-003"})

    def test_edit_unclosed_issue_without_changing_confirmed_scope(self):
        agenda=initial_agenda({"issues":[planned()]},{"O-01"})
        delta={"base_agenda_revision":1,"operations":[{
            "op":"update_issue","issue_id":"I-01",
            "question":"核查原始记录的来源版本和估计值",
            "priority":"high"}]}
        edited,rev=apply_delta(agenda,delta,1,{"O-01"},set())
        self.assertEqual((edited[0]["question"],edited[0]["priority"]),
                         ("核查原始记录的来源版本和估计值","high"))
        self.assertEqual(edited[0]["objective_refs"],["O-01"])
        self.assertEqual(agenda[0]["question"],"核查原始记录")
        self.assertEqual(rev,2)
        with self.assertRaises(WorkflowConflict):
            apply_delta(edited,{"base_agenda_revision":2,"operations":[{
                "op":"reprioritize","issue_id":"I-01","priority":"high"}]},
                2,{"O-01"},set())
        edited[0]["status"]="done"
        with self.assertRaises(WorkflowConflict):
            apply_delta(edited,{"base_agenda_revision":2,"operations":[{
                "op":"update_issue","issue_id":"I-01",
                "question":"再写一个问题","priority":"high"}]},
                2,{"O-01"},set())

    def test_close_never_auto_promotes_candidate_to_medical_truth(self):
        issue={"issue_id":"I-01","evidence_ids":["E-002"]}
        proposal={"issue_id":"I-01","disposition":"limited",
                  "statements":[{"text":"原始病历记载250 mL",
                                  "evidence_refs":["E-002"],
                                  "qualifiers":{"source_scope":"记载值"}}],
                  "unresolved_questions":[]}
        self.assertEqual(check_close(proposal,issue,["E-002"],set())
                         ["disposition"],"limited")
        with self.assertRaises(WorkflowConflict):
            check_close({**proposal,"disposition":"supported"},
                        issue,["E-002"],set())
        with self.assertRaises(WorkflowConflict):
            check_close({**proposal,"statements":[{
                "text":"伪造值","evidence_refs":["E-003"],
                "qualifiers":{}}]},issue,["E-002"],set())

class GraphIntegration(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root=temp.name
        self.m1=DemoWorkflowService(self.root)

    def approved(self):
        task=self.m1.start()
        self.m1.resume(task["run_id"],"confirm",0,1)
        return task["run_id"]

    def test_full_multissue_fake_end_to_end_and_immutable_snapshots(self):
        run=self.approved()
        scripted=ScriptedInvestigator()
        workflow=M8LocalWorkflow(self.root,agent=scripted)
        out=workflow.execute(run)
        self.assertEqual(out["phase"],"complete")
        self.assertEqual(out["outcome"],"restricted_unverified")
        self.assertEqual([x["issue_id"] for x in out["agenda"]],
                         ["I-01","I-02","I-03"])
        self.assertEqual(out["seen_evidence"],["E-002","E-003"])
        self.assertEqual(out["tool_calls"],2)
        self.assertEqual(scripted.consolidation_calls,3)
        self.assertEqual(workflow.repo.count(run),6)
        self.assertEqual(out["next"],[])
        self.assertIn("close:I-03",out["trace"])
        third=workflow.repo.get(run,"issue","I-03:r1")
        self.assertEqual(third["disposition"],"unresolved")
        self.assertEqual(third["semantic_validation"],"not_evaluated")
        self.assertEqual(third["evidence_ids"],["E-002","E-003"])
        self.assertFalse("report" in out)
        assert any(c.get("prior_issue_results") for c in scripted.contexts)
        self.assertEqual(out["agenda_revision"],6)

    def test_restart_is_idempotent_and_run_isolation(self):
        rid=self.approved()
        first=M8LocalWorkflow(self.root).execute(rid)
        second=M8LocalWorkflow(self.root,agent=BrokenDecisions()).execute(rid)
        self.assertEqual(first,second)
        self.assertEqual(M8LocalWorkflow(self.root).repo.count(rid),6)
        other=self.approved()
        with self.assertRaises(WorkflowConflict):
            M8LocalWorkflow(self.root).repo.get(other,"issue","I-01:r1")

    def test_no_doctor_confirm_and_invalid_model_calls_fail_closed(self):
        unconfirmed=self.m1.start()["run_id"]
        with self.assertRaises(WorkflowConflict):
            M8LocalWorkflow(self.root).execute(unconfirmed)
        bad=self.approved()
        out=M8LocalWorkflow(self.root,agent=BrokenDecisions()).execute(bad)
        self.assertEqual(out["phase"],"blocked")
        self.assertEqual(out["tool_calls"],0)
        self.assertEqual(out["steps"],2)
        self.assertEqual(M8LocalWorkflow(self.root).repo.count(bad),1)

    def test_bad_close_does_not_persist_issue(self):
        rid=self.approved()
        with self.assertRaises(WorkflowConflict):
            M8LocalWorkflow(self.root,agent=BrokenClose()).execute(rid)
        repo=M8LocalWorkflow(self.root).repo
        self.assertEqual(repo.count(rid),2) # seed + source only

    def test_recovery_after_close_sql_commit_without_checkpoint(self):
        rid=self.approved()
        first_agent=ScriptedInvestigator()
        first=M8LocalWorkflow(self.root,agent=first_agent)
        first.repo=AfterCommitCrash(first.repo)
        with self.assertRaisesRegex(RuntimeError,"simulated crash"):
            first.execute(rid)
        second_agent=ScriptedInvestigator()
        recovered=M8LocalWorkflow(self.root,agent=second_agent)
        final=recovered.execute(rid)
        self.assertEqual(final["phase"],"complete")
        self.assertEqual(final["tool_calls"],2)
        self.assertEqual(recovered.repo.count(rid),6)
        self.assertEqual(second_agent.consolidation_calls,2)

    def test_confirmed_summary_is_source_of_runtime_goal(self):
        rid=self.approved()
        scripted=ScriptedInvestigator()
        M8LocalWorkflow(self.root,agent=scripted).execute(rid)
        confirmed=self.m1.store.contract(rid,"DOC-001")["summary"]
        self.assertEqual(scripted.contexts[0]["contract_goal"],confirmed)
        self.assertIn("P001",confirmed)

    def test_completed_checkpoint_cannot_be_read_with_changed_objectives(self):
        rid=self.approved()
        M8LocalWorkflow(self.root).execute(rid)
        changed={"goal":"ignored local override",
            "objectives":[{"id":"O-01","text":"a new set of objectives"}],
            "skill":{"id":"perinatal-investigation","version":"SIM-v1"},
            "risk_profile":{"id":"PPH-SIM","version":1}}
        with self.assertRaisesRegex(WorkflowConflict,"Contract binding"):
            M8LocalWorkflow(self.root,contract=changed).execute(rid)

    def test_alternate_plan_is_not_hardcoded_to_p001_issue_list(self):
        rid=self.approved()
        only={"question":"核查模型是否适合此次任务",
              "category":"model_capability","priority":"high",
              "objective_refs":["O-01"],"trigger_evidence_refs":[],
              "related_issue_ids":[]}
        contract={"goal":"合成模型资格调查",
            "objectives":[{"id":"O-01","text":"核查资格"}],
            "skill":{"id":"perinatal-investigation","version":"SIM-v1"},
            "risk_profile":{"id":"PPH-SIM","version":1}}
        out=M8LocalWorkflow(self.root,agent=ScriptedInvestigator([only]),
                            contract=contract).execute(rid)
        self.assertEqual(len(out["agenda"]),1)
        self.assertEqual(out["phase"],"complete")
        self.assertEqual(out["seen_evidence"],[])
        self.assertEqual(out["agenda"][0]["disposition"],"not_applicable")

if __name__=="__main__":
    unittest.main()
