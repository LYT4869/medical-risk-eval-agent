"""Per-turn bounded cross-Issue context from immutable, Run-scoped artifacts."""
from __future__ import annotations
from v2_m1.run_store import WorkflowConflict

def get_evidence(state, repo, ref):
    key=state.get("source_keys",{}).get(ref)
    if not key or ref not in state.get("seen_evidence",[]):
        raise WorkflowConflict("unknown or unobserved Evidence reference")
    item=repo.get(state["run_id"],"evidence",key)
    if (item.get("run_id")!=state["run_id"] or
        item.get("classification")!="SIM" or item.get("evidence_id")!=ref):
        raise WorkflowConflict("invalid Evidence identity")
    return item

def get_issue_result(state, repo, issue_id):
    item=next((x for x in state["agenda"] if x["issue_id"]==issue_id),None)
    if not item or not item.get("result_key"):
        raise WorkflowConflict("unknown or unclosed Issue Result")
    result=repo.get(state["run_id"],"issue",item["result_key"])
    if (result.get("run_id")!=state["run_id"] or
        result.get("classification")!="SIM" or
        result.get("issue_id")!=issue_id):
        raise WorkflowConflict("Issue Result identity mismatch")
    return result

def context_for_issue(state, issue, repo, contract, *,
                      max_history=4, max_evidence=5):
    """Related results first; also include unresolved history when available.

    No free-form patient-wide retrieval, no raw records, no cross-Run search.
    The returned data is a prompt projection, not the authoritative memory.
    """
    closed=[x for x in state["agenda"] if x.get("result_key") and
            x["issue_id"]!=issue["issue_id"]]
    indexed={x["issue_id"]:x for x in closed}
    related=[i for i in issue["related_issue_ids"] if i in indexed]
    unresolved=[x["issue_id"] for x in closed if x.get("disposition") in
                ("unresolved","needs_clinician_review") and
                x["issue_id"] not in related]
    remaining=[x["issue_id"] for x in closed if
               x["issue_id"] not in related+unresolved]
    history=[]
    for issue_id in (related+unresolved+remaining)[:max_history]:
        result=get_issue_result(state,repo,issue_id)
        history.append({
            "issue_id":issue_id,
            "disposition":result["disposition"],
            "semantic_validation":result.get("semantic_validation"),
            "statements":[{"text":s["text"][:340],
                           "evidence_refs":list(s["evidence_refs"]),
                           "qualifiers":s["qualifiers"]}
                          for s in result["statements"][:3]],
            "unresolved_questions":result["unresolved_questions"][:3],
        })
    references=list(dict.fromkeys(issue["evidence_ids"] + [
        ref for past in history for s in past["statements"]
        for ref in s["evidence_refs"]
    ]))[:max_evidence]
    evidence=[]
    for ref in references:
        record=get_evidence(state,repo,ref)["record"]
        evidence.append({
            "evidence_id":ref,"kind":record.get("kind"),
            "source_record_id":record.get("logical_id"),
            "source_version":record.get("version"),
            "blood_loss_ml":record.get("blood_loss_ml"),
            "approximate":record.get("approximate",False),
        })
    return {
        "contract_goal":contract["goal"][:360],
        "confirmed_objectives":contract["objectives"],
        "skill":contract["skill"],
        "risk_profile":contract["risk_profile"],
        "issue":issue.copy(),
        "agenda_revision":state["agenda_revision"],
        "prior_issue_results":history,
        "evidence_summaries":evidence,
        "seen_evidence_ids":list(state["seen_evidence"])[:12],
        "pending_issue_ids":[x["issue_id"] for x in state["agenda"]
                             if x["status"] in ("todo","in_progress")],
        "remaining_tool_calls":state["max_tool_calls"]-state["tool_calls"],
        "available_tools":["read_patient_record","get_issue_result"],
    }

def lookup_result(state, repo, issue_id):
    """Explicit bounded get_issue_result. No untrusted patient/run arguments."""
    return get_issue_result(state,repo,issue_id)
