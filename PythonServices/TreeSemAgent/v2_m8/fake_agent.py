"""Deterministic semantic-script test double for M8.

All semantic branching is *inside the replaceable Agent*, not Runtime.
This deliberately does not establish real Qwen investigation quality.
"""
from __future__ import annotations
from copy import deepcopy

class ScriptedInvestigator:
    def __init__(self, plan=None):
        self.plan=plan or [
            {"question":"核查原始分娩资料中的失血量记载",
             "category":"original_record","priority":"normal",
             "objective_refs":["O-01"],"trigger_evidence_refs":[],
             "related_issue_ids":[]},
            {"question":"核查转诊摘要并识别来源差异",
             "category":"referral_review","priority":"normal",
             "objective_refs":["O-02"],"trigger_evidence_refs":[],
             "related_issue_ids":[]},
        ]
        self.consolidation_calls=0
        self.contexts=[]

    def plan_task(self, context):
        self.contexts.append(deepcopy(context))
        return {"issues":deepcopy(self.plan)}

    def decide(self, context):
        self.contexts.append(deepcopy(context))
        item=context["issue"]
        expected={"original_record":"E-002",
                  "referral_review":"E-003"}.get(item["category"])
        if expected and expected not in context["seen_evidence_ids"]:
            return {"action":"read","record_id":expected}
        return {"action":"close_issue","issue_id":item["issue_id"]}

    def consolidate(self, context):
        self.consolidation_calls+=1
        self.contexts.append(deepcopy(context))
        issue=context["issue"]
        category=issue["category"]
        if category=="original_record":
            return {"issue_id":issue["issue_id"],"disposition":"limited",
                    "statements":[{"text":"原始记录记载估计失血量250 mL",
                                   "evidence_refs":["E-002"],
                                   "qualifiers":{"source_scope":"原始记录中的估计值"}}],
                    "unresolved_questions":[]}
        if category=="referral_review":
            prior=context["prior_issue_results"]
            has_old=any(x["issue_id"]=="I-01" and
                        any("E-002" in s["evidence_refs"]
                            for s in x["statements"]) for x in prior)
            assert has_old, "Fake requires actual cross-Issue Result memory"
            return {"issue_id":issue["issue_id"],"disposition":"unresolved",
                    "statements":[{"text":"转诊摘要记载约800 mL，尚未核实差异",
                                   "evidence_refs":["E-003"],
                                   "qualifiers":{"uncertainty":"未确认与原始记录是否可比"}}],
                    "unresolved_questions":["核查250与约800mL是否属于同一分娩事件"],
                    "agenda_delta":{"base_agenda_revision":context["agenda_revision"],
                        "operations":[{"op":"add_issue",
                            "question":"核对原始记录与转诊摘要的估计失血量差异",
                            "category":"source_discrepancy",
                            "priority":"high","objective_refs":["O-02"],
                            "trigger_evidence_refs":["E-002","E-003"],
                            "related_issue_ids":["I-01","I-02"]}]}}
        if category=="source_discrepancy":
            return {"issue_id":issue["issue_id"],"disposition":"unresolved",
                    "statements":[{"text":"两份不同来源记载的数值有差异，原因未核实",
                                   "evidence_refs":["E-002","E-003"],
                                   "qualifiers":{"uncertainty":"不能确认实际失血量"}}],
                    "unresolved_questions":["需要医生结合原始档案核对"],
                    }
        return {"issue_id":issue["issue_id"],
                "disposition":"not_applicable","statements":[],
                "unresolved_questions":["该合成调查项暂无已验证的医学推断"]}
