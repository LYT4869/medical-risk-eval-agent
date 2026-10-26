"""Composed close_issue workflow: load -> consolidate -> validate -> commit.

The committed candidate result and approved Agenda Delta share one immutable
artifact. Replay reads that artifact before ever regenerating model text.
"""
from __future__ import annotations
from copy import deepcopy
from v2_m1.run_store import WorkflowConflict
from .contracts import apply_delta, check_close
from .context import context_for_issue, get_issue_result

def close_issue(state, issue, repo, contract, agent):
    """Callable Workflow-as-Tool, opted into by an Agent close_issue decision.

    This is *not* a semantic/clinical Reviewer. The resulting statements
    are unverified synthetic proposals, never a publishable M5 draft.
    """
    artifact_key=issue["issue_id"]+":r1"
    cached=None
    try:
        cached=repo.get(state["run_id"],"issue",artifact_key)
    except WorkflowConflict as exc:
        if "artifact missing" not in str(exc):
            raise

    staged=deepcopy(state["agenda"])
    target=next(x for x in staged if x["issue_id"]==issue["issue_id"])
    target["status"]="done"
    target["result_key"]=artifact_key

    if cached is None:
        context=context_for_issue(state,issue,repo,contract)
        related_refs={ref for item in context["prior_issue_results"]
                      for s in item["statements"] for ref in s["evidence_refs"]}
        proposal=agent.consolidate(context)
        checked=check_close(proposal,issue,state["seen_evidence"],related_refs)
        target["disposition"]=checked["disposition"]
        delta=checked["agenda_delta"]
        if delta is not None:
            # Validate entire proposed batch before any SQL write.
            after,next_revision=apply_delta(
                staged,delta,state["agenda_revision"],
                {x["id"] for x in contract["objectives"]},
                state["seen_evidence"])
        else:
            after,next_revision=staged,state["agenda_revision"]+1
        statements=[
            {"statement_id":f"{issue['issue_id']}-S{idx:02}",**s,
             "semantic_validation":"not_evaluated"}
            for idx,s in enumerate(checked["statements"],1)]
        cached={"classification":"SIM","run_id":state["run_id"],
                "issue_id":issue["issue_id"],"result_revision":1,
                "status":"closed_unverified",
                "disposition":checked["disposition"],
                "statements":statements,
                "evidence_ids":list(dict.fromkeys(
                    ref for s in statements for ref in s["evidence_refs"])),
                "unresolved_questions":checked["unresolved_questions"],
                "agenda_delta":delta,
                "semantic_validation":"not_evaluated",
                "no_diagnosis":True}
        repo.save(state["run_id"],"issue",artifact_key,cached)
    else:
        if (cached.get("run_id")!=state["run_id"] or
                cached.get("classification")!="SIM" or
                cached.get("issue_id")!=issue["issue_id"] or
                cached.get("semantic_validation")!="not_evaluated"):
            raise WorkflowConflict("existing Close artifact identity mismatch")
        target["disposition"]=cached["disposition"]
        delta=cached.get("agenda_delta")
        if delta is not None:
            after,next_revision=apply_delta(
                staged,delta,state["agenda_revision"],
                {x["id"] for x in contract["objectives"]},
                state["seen_evidence"])
        else:
            after,next_revision=staged,state["agenda_revision"]+1
    return {"agenda":after,"agenda_revision":next_revision,
            "result_key":artifact_key,"disposition":cached["disposition"]}
