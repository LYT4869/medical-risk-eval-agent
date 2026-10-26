"""Typed proposal validation for the opt-in synthetic M8 agent runtime.

Machine checking covers IDs, state transitions, scopes and versions.
It never claims to verify arbitrary natural language statements.
"""
from __future__ import annotations
from copy import deepcopy
from v2_m1.run_store import WorkflowConflict

PRIORITIES = {"low": 0, "normal": 1, "high": 2}
DISPOSITIONS = {"limited", "unresolved", "not_applicable", "needs_clinician_review"}
MAX_ISSUES = 8

def check(condition, reason):
    if not condition:
        raise WorkflowConflict("M8 proposal rejected: " + reason)

def fields(data, required, optional=()):
    check(isinstance(data, dict), "expected object")
    check(set(required) <= set(data) and set(data) <= set(required) | set(optional),
          "invalid fields")

def label(value, limit=240):
    check(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
          "invalid bounded label")
    return value.strip()

def refs(values, available, max_count=8):
    check(isinstance(values, list) and len(values) <= max_count, "invalid refs")
    check(all(isinstance(x, str) and x in available for x in values),
          "unknown or unauthorized reference")
    check(len(set(values)) == len(values), "duplicate refs")
    return list(values)

def issue_spec(item, objectives, known_evidence, known_issues):
    names=("question", "category", "priority", "objective_refs",
           "trigger_evidence_refs", "related_issue_ids")
    fields(item, names, ("op",))
    check(isinstance(item["priority"], str) and item["priority"] in PRIORITIES, "priority")
    objective_refs = refs(item["objective_refs"], objectives)
    check(bool(objective_refs), "issue must serve a confirmed objective")
    return {"question": label(item["question"]),
            "category": label(item["category"], 64),
            "priority": item["priority"],
            "objective_refs": objective_refs,
            "trigger_evidence_refs": refs(item["trigger_evidence_refs"], known_evidence),
            "related_issue_ids": refs(item["related_issue_ids"], known_issues)}

def initial_agenda(proposal, objectives):
    fields(proposal, ("issues",))
    items=proposal["issues"]
    check(isinstance(items,list) and 1 <= len(items) <= MAX_ISSUES, "issue count")
    agenda, covered, names=[],set(),set()
    for idx,item in enumerate(items,1):
        spec=issue_spec(item, set(objectives), set(), set())
        name=spec["question"].casefold()
        check(name not in names, "duplicate question")
        names.add(name)
        covered.update(spec["objective_refs"])
        agenda.append({"issue_id":f"I-{idx:02}", **spec, "status":"todo",
                       "disposition":None,"evidence_ids":[],
                       "result_key":None})
    check(set(objectives) <= covered,"uncovered confirmed objective")
    return agenda

def apply_delta(agenda, delta, revision, objectives, seen):
    fields(delta, ("base_agenda_revision","operations"))
    check(type(delta["base_agenda_revision"]) is int and
          delta["base_agenda_revision"] == revision, "stale agenda revision")
    operations=delta["operations"]
    check(isinstance(operations,list) and 1<=len(operations)<=3,"delta size")
    after=deepcopy(agenda)
    for op in operations:
        fields(op, ("op",), ("question","category","priority",
             "objective_refs","trigger_evidence_refs","related_issue_ids","issue_id"))
        if op["op"]=="add_issue":
            check(len(after)<MAX_ISSUES,"agenda capacity")
            spec=issue_spec(op,set(objectives),set(seen),
                            {x["issue_id"] for x in after})
            check(not any(x["question"].casefold()==spec["question"].casefold()
                          for x in after),"duplicate question")
            after.append({"issue_id":f"I-{len(after)+1:02}", **spec,
                          "status":"todo","disposition":None,
                          "evidence_ids":list(spec["trigger_evidence_refs"]),
                          "result_key":None})
        elif op["op"]=="update_issue":
            # The core category/objective bindings stay unchanged. A semantic
            # change of task scope requires re-confirmation, not an Agenda edit.
            fields(op, ("op","issue_id","question","priority"))
            check(isinstance(op["priority"],str) and
                  op["priority"] in PRIORITIES,"bad priority")
            target=next((x for x in after if x["issue_id"]==op["issue_id"]),None)
            check(target is not None and target["status"] in ("todo","in_progress"),
                  "cannot edit completed Issue")
            changed=label(op["question"])
            check(not any(x["issue_id"]!=target["issue_id"] and
                          x["question"].casefold()==changed.casefold()
                          for x in after),"duplicate question")
            check(changed!=target["question"] or
                  op["priority"]!=target["priority"],"no-progress Issue update")
            target["question"]=changed
            target["priority"]=op["priority"]
        elif op["op"]=="reprioritize":
            fields(op, ("op","issue_id","priority"))
            check(isinstance(op["priority"],str) and op["priority"] in PRIORITIES,"bad priority")
            target=next((x for x in after if x["issue_id"]==op["issue_id"]),None)
            check(target is not None and target["status"] in ("todo","in_progress"),
                  "cannot reprioritize completed Issue")
            check(target["priority"]!=op["priority"],"no-progress priority update")
            target["priority"]=op["priority"]
        else:
            check(False,"unsupported operation")
    return after, revision+1

def check_decision(proposal, issue, state, available_records):
    fields(proposal, ("action",), ("record_id","issue_id","delta"))
    action=proposal["action"]
    if action=="read":
        fields(proposal, ("action","record_id"))
        check(proposal["record_id"] in available_records and
              proposal["record_id"] not in state["seen_evidence"] and
              state["tool_calls"] < state["max_tool_calls"], "invalid source read")
    elif action=="close_issue":
        fields(proposal, ("action","issue_id"))
        check(proposal["issue_id"]==issue["issue_id"],"wrong active Issue")
    elif action=="agenda_delta":
        fields(proposal, ("action","delta"))
        check(isinstance(proposal["delta"],dict) and
              proposal["delta"].get("base_agenda_revision") ==
              state["agenda_revision"],"stale agenda delta")
    else:
        check(False,"unsupported action")
    return deepcopy(proposal)

def check_close(proposal, issue, seen, related_refs):
    fields(proposal, ("issue_id","disposition","statements",
                      "unresolved_questions"), ("agenda_delta",))
    check(proposal["issue_id"]==issue["issue_id"],"wrong close Issue")
    check(proposal["disposition"] in DISPOSITIONS,"unsupported disposition")
    statements=proposal["statements"]
    check(isinstance(statements,list) and len(statements)<=6,"too many statements")
    valid=set(issue["evidence_ids"]) | set(related_refs)
    check(valid <= set(seen),"out-of-run source in Close")
    out=[]
    for s in statements:
        fields(s,("text","evidence_refs","qualifiers"))
        check(isinstance(s["qualifiers"],dict) and
              set(s["qualifiers"]) <= {"source_scope","uncertainty"},
              "invalid qualifiers")
        qualifier={key:label(value,120)
                   for key,value in s["qualifiers"].items()}
        evidence=refs(s["evidence_refs"],valid)
        check(bool(evidence) or proposal["disposition"]=="not_applicable",
              "uncited candidate assertion")
        out.append({"text":label(s["text"],340),"evidence_refs":evidence,
                    "qualifiers":qualifier})
    u=proposal["unresolved_questions"]
    check(isinstance(u,list) and len(u)<=5,"unresolved count")
    unresolved=[label(item,180) for item in u]
    check(bool(out) or bool(unresolved) or
          proposal["disposition"]=="not_applicable","empty Close")
    return {"issue_id":issue["issue_id"],"disposition":proposal["disposition"],
            "statements":out,"unresolved_questions":unresolved,
            "agenda_delta":deepcopy(proposal.get("agenda_delta"))}
