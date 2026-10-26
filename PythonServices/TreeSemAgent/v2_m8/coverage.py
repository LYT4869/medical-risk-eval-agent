"""Read-only M9 structural Coverage preview; no clinical report admission.

M9 objectives are a synthetic declared test profile, not yet an independent
semantic interpretation of the doctor-confirmed Task Contract.
"""
from __future__ import annotations
from v2_m1.run_store import WorkflowConflict
from .context import get_issue_result

def structural_coverage(state, repo, contract):
    objectives=contract["objectives"]
    if not isinstance(objectives,list) or not objectives:
        raise WorkflowConflict("synthetic profile objectives missing")
    collected=[]
    incomplete=[]
    unreviewed=0
    seen=set()
    for objective in objectives:
        ident=objective["id"]
        if ident in seen:
            raise WorkflowConflict("duplicate synthetic objective")
        seen.add(ident)
        linked=[x for x in state["agenda"] if ident in x["objective_refs"]]
        dispositions=[]
        if not linked:
            incomplete.append(ident)
        for issue in linked:
            if not issue.get("result_key"):
                incomplete.append(ident)
                continue
            result=get_issue_result(state,repo,issue["issue_id"])
            if (result.get("semantic_validation")!="not_evaluated" or
                    result.get("status")!="closed_unverified"):
                raise WorkflowConflict("unverified dynamic result claimed to be verified")
            unreviewed+=len(result["statements"])
            dispositions.append(result["disposition"])
        collected.append({"objective_id":ident,
                          "issue_ids":[x["issue_id"] for x in linked],
                          "dispositions":dispositions})
    return {"classification":"SIM",
            "scope":"structural_coverage_only_not_report_admission",
            "objective_source":"synthetic_profile_not_semantically_confirmed",
            "unreviewed_statement_count":unreviewed,
            "incomplete_objective_ids":sorted(set(incomplete)),
            "objectives":collected,
            "source_freshness":"not_evaluated",
            "clinical_grounding":"not_evaluated",
            "publish_allowed":False,
            "reason":"candidate assertions need independent source/clinical grounding"}
