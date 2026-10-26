"""Opt-in LangGraph M8: dynamic planning, guarded tools, cross-Issue Close.

Independent of the fixed M3/M5 P001 report protocol. No real LLM, hospital
connection, clinical sentence verification, or report publish is performed.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from v2_m3.policy import AS_OF, M3FixtureTools
from v2_m3.repository import LocalM3Repository

from .contracts import PRIORITIES, apply_delta, check, check_decision, initial_agenda
from .context import context_for_issue, get_evidence, get_issue_result
from .close import close_issue
from .fake_agent import ScriptedInvestigator
from .coverage import structural_coverage

class RuntimeState(TypedDict, total=False):
    run_id: str
    contract_revision: int
    agenda: list[dict]
    agenda_revision: int
    phase: str
    seen_evidence: list[str]
    source_keys: dict[str,str]
    tool_calls: int
    max_tool_calls: int
    steps: int
    no_progress: int
    pending: dict
    trace: list[str]
    outcome: str
    contract_digest: str

def select_issue(agenda):
    """Resume in-progress first; else high-priority todo with stable order."""
    for item in agenda:
        if item["status"]=="in_progress":
            return item
    todo=[x for x in agenda if x["status"]=="todo"]
    if not todo:
        return None
    return max(todo,key=lambda x:PRIORITIES[x["priority"]])

def build_graph(repo, authorize, agent, contract, saver, *,
                max_steps=24, max_tool_calls=6):
    if not 1<=max_steps<=40 or not 1<=max_tool_calls<=8:
        raise ValueError("invalid runtime budget")
    objective_ids={x["id"] for x in contract["objectives"]}
    check(len(objective_ids)==len(contract["objectives"]) and bool(objective_ids),
          "confirmed objective IDs required")
    tools=M3FixtureTools()
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,
                     ensure_ascii=False).encode("utf-8")).hexdigest()

    def authorized(state):
        authorize(state["run_id"],state["contract_revision"])
        if state.get("contract_digest") and state["contract_digest"]!=digest:
            raise WorkflowConflict("confirmed Contract binding changed")

    def seed(state):
        authorized(state)
        if state.get("agenda"):
            raise WorkflowConflict("initial plan already created")
        # No graph checkpoint is needed to recover exactly the same LLM proposal.
        key=f"contract-r{state['contract_revision']}"
        try:
            cached=repo.get(state["run_id"],"agenda_seed",key)
        except WorkflowConflict as exc:
            if "artifact missing" not in str(exc) and getattr(exc, "status", None) != 404:
                raise
            cached=None
        if cached is None:
            proposed=agent.plan_task({
                "contract_goal":contract["goal"][:360],
                "confirmed_objectives":contract["objectives"],
                "skill":contract["skill"],"risk_profile":contract["risk_profile"],
                "initial_evidence_refs":[],
            })
            agenda=initial_agenda(proposed,objective_ids)
            cached={"classification":"SIM","run_id":state["run_id"],
                    "contract_revision":state["contract_revision"],
                    "agenda":agenda,"contract_digest":digest,
                    "contract_summary":contract["goal"][:360]}
            repo.save(state["run_id"],"agenda_seed",key,cached)
        else:
            if (cached.get("run_id")!=state["run_id"] or
                    cached.get("contract_revision")!=state["contract_revision"] or
                    cached.get("classification")!="SIM" or
                    cached.get("contract_digest")!=digest):
                raise WorkflowConflict("Agenda seed or approved Contract changed")
        return {"phase":"investigating","agenda":cached["agenda"],
                "contract_digest":digest,
                "agenda_revision":1,"source_keys":{},"seen_evidence":[],
                "steps":0,"tool_calls":0,"max_tool_calls":max_tool_calls,
                "pending":{},"no_progress":0,"trace":["agenda:seed"],
                "outcome":"not_evaluated"}

    def decide(state):
        authorized(state)
        issue=select_issue(state["agenda"])
        if issue is None:
            return {"phase":"complete","outcome":"restricted_unverified",
                    "pending":{}}
        if state["steps"]>=max_steps:
            return {"phase":"blocked","outcome":"budget_exhausted",
                    "pending":{},"trace":[*state["trace"],"budget:max_steps"]}
        # Work is not declared complete simply because a tool succeeded.
        agenda=deepcopy(state["agenda"])
        active=next(x for x in agenda if x["issue_id"]==issue["issue_id"])
        changed=active["status"]=="todo"
        active["status"]="in_progress"
        proposed=agent.decide(context_for_issue(state,active,repo,contract))
        try:
            approved=check_decision(proposed,active,state,tools.SOURCES)
            if approved["action"]=="agenda_delta":
                # Validate BEFORE executing any observable state update.
                apply_delta(agenda,approved["delta"],state["agenda_revision"],
                            objective_ids,state["seen_evidence"])
        except (WorkflowConflict,KeyError,ValueError,TypeError):
            no_progress=state["no_progress"]+1
            return {"agenda":agenda,
                    "agenda_revision":state["agenda_revision"]+(1 if changed else 0),
                    "phase":"blocked" if no_progress>=2 else "retry",
                    "no_progress":no_progress,"steps":state["steps"]+1,
                    "pending":{},"trace":[*state["trace"],"guard:rejected"]}
        # Entering the Issue is an explicit, versioned transition.
        # For an agenda_delta the proposal must use the incoming revision;
        # hence do NOT increment for merely marking in_progress here.
        return {"phase":"execute","pending":approved,"agenda":agenda,
                "no_progress":0,"trace":[*state["trace"],
                   f"agent:{issue['issue_id']}:{approved['action']}"]}

    def execute(state):
        authorized(state)
        issue=select_issue(state["agenda"])
        if issue is None or state["phase"]!="execute":
            raise WorkflowConflict("no active Issue for M8 action")
        decision=check_decision(state["pending"],issue,state,tools.SOURCES)
        if decision["action"]=="read":
            ref=decision["record_id"]
            record=tools.call("read_patient_record",actor_id="DOC-001",
                              patient_id="P001",as_of=AS_OF,record_id=ref)
            if record.get("id")!=ref:
                raise WorkflowConflict("wrong source returned")
            key=f"{record['logical_id']}:v{record['version']}"
            repo.save(state["run_id"],"evidence",key,{
                "classification":"SIM","run_id":state["run_id"],
                "evidence_id":ref,"record":record,
            })
            agenda=deepcopy(state["agenda"])
            target=next(x for x in agenda if x["issue_id"]==issue["issue_id"])
            target["evidence_ids"].append(ref)
            return {"phase":"investigating","agenda":agenda,
                    "agenda_revision":state["agenda_revision"]+1,
                    "source_keys":{**state["source_keys"],ref:key},
                    "seen_evidence":[*state["seen_evidence"],ref],
                    "tool_calls":state["tool_calls"]+1,
                    "steps":state["steps"]+1,"pending":{},
                    "trace":[*state["trace"],f"tool:{issue['issue_id']}:{ref}"]}
        if decision["action"]=="close_issue":
            changed=close_issue(state,issue,repo,contract,agent)
            return {"phase":"investigating","agenda":changed["agenda"],
                    "agenda_revision":changed["agenda_revision"],
                    "steps":state["steps"]+1,"pending":{},
                    "trace":[*state["trace"],f"close:{issue['issue_id']}"]}
        agenda,next_rev=apply_delta(
            state["agenda"],decision["delta"],state["agenda_revision"],
            objective_ids,state["seen_evidence"])
        return {"phase":"investigating","agenda":agenda,
                "agenda_revision":next_rev,"steps":state["steps"]+1,
                "pending":{},"trace":[*state["trace"],"agenda:delta"]}

    def route(state):
        if state["phase"]=="execute":
            return "execute"
        if state["phase"] in ("complete","blocked"):
            return END
        return "decide"

    graph=StateGraph(RuntimeState)
    graph.add_node("seed",seed)
    graph.add_node("decide",decide)
    graph.add_node("execute",execute)
    graph.add_edge(START,"seed")
    graph.add_edge("seed","decide")
    graph.add_conditional_edges("decide",route)
    graph.add_edge("execute","decide")
    return graph.compile(checkpointer=saver)

class M8LocalWorkflow:
    """P001-authorized local integration harness; M3/M5/MySQL remain unchanged."""
    def __init__(self,directory,*,agent=None,contract=None):
        root=Path(directory)
        root.mkdir(parents=True,exist_ok=True)
        self.m1=DemoWorkflowService(root)
        self.repo=LocalM3Repository(root/"m8_business.sqlite3")
        self.checkpoints=str(root/"m8_graph_checkpoints.sqlite3")
        self.agent=agent or ScriptedInvestigator()
        self.contract=contract or {
            "goal":"P001 SIM：核查两个来源并保留未决差异",
            "objectives":[{"id":"O-01","text":"核查原始资料"},
                          {"id":"O-02","text":"核查转诊资料"}],
            "skill":{"id":"perinatal-investigation","version":"SIM-v1"},
            "risk_profile":{"id":"PPH-SIM","version":1},
        }

    def authorized(self,run_id,contract_revision):
        item=self.m1.store.get(run_id,"DOC-001")
        if (item["status"]!="contract_confirmed" or
                item["subject_id"]!="P001" or
                item["contract_revision"]!=contract_revision):
            raise WorkflowConflict("M8 needs current confirmed SIM Contract")

    def execute(self,run_id,contract_revision=1,*,max_steps=24,max_tool_calls=6):
        self.authorized(run_id,contract_revision)
        config={"configurable":{"thread_id":f"treesem-v2:m8:{run_id}"}}
        approved=self.m1.store.contract(run_id,"DOC-001")
        # The summary is authoritative in this SIM-only M1 workflow.
        # Objective IDs remain a declared synthetic test profile, not parsed clinical goals.
        bound_contract={**self.contract,"goal":approved["summary"]}
        expected=hashlib.sha256(json.dumps(bound_contract,sort_keys=True,
                          ensure_ascii=False).encode("utf-8")).hexdigest()
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph=build_graph(self.repo,self.authorized,self.agent,
                              bound_contract,saver,max_steps=max_steps,
                              max_tool_calls=max_tool_calls)
            previous=graph.get_state(config)
            if not previous.values:
                graph.invoke({"run_id":run_id,"contract_revision":contract_revision},
                             config=config,durability="sync")
            elif previous.next:
                graph.invoke(None,config=config,durability="sync")
            current=graph.get_state(config)
        values=current.values
        if values and values.get("contract_digest")!=expected:
            raise WorkflowConflict("resumed M8 Graph has a different Contract binding")
        return {"run_id":run_id,"phase":values.get("phase"),
                "outcome":values.get("outcome"),"agenda":values.get("agenda",[]),
                "agenda_revision":values.get("agenda_revision"),
                "trace":values.get("trace",[]),
                "seen_evidence":values.get("seen_evidence",[]),
                "tool_calls":values.get("tool_calls",0),
                "steps":values.get("steps",0),"next":list(current.next)}


class M8GatewayWorkflow:
    """M9 isolated M8 dynamic graph with authoritative C++/MySQL SIM Run.

    The capability-scoped Gateway only persists dynamic artifacts in m8_* kinds.
    M5/M6 cannot consume these candidate Statements as report-worthy assertions.
    """
    def __init__(self, directory, gateway, *, agent=None, contract=None):
        from .gateway_repository import GatewayM8Repository
        root=Path(directory)
        root.mkdir(parents=True,exist_ok=True)
        self.gateway=gateway
        self.repo=GatewayM8Repository(gateway)
        self.checkpoints=str(root/"m9_graph_checkpoints.sqlite3")
        self.agent=agent or ScriptedInvestigator()
        self.contract=contract or {
            "goal":"SIM P001 goal injected from C++",
            "objectives":[{"id":"O-01","text":"核查原始资料"},
                          {"id":"O-02","text":"核查转诊资料"}],
            "skill":{"id":"perinatal-investigation","version":"SIM-v1"},
            "risk_profile":{"id":"PPH-SIM","version":1},
        }

    def authorized(self,run_id,contract_revision):
        status=self.gateway.status(run_id)
        if (status.get("run_id")!=run_id or
                status.get("classification")!="SIM" or
                status.get("fixture_id")!="P001" or
                status.get("status")!="contract_confirmed" or
                status.get("contract_revision")!=contract_revision):
            raise WorkflowConflict("M9 requires current authoritative confirmed Run")

    def execute(self,run_id,contract_revision=1,*,max_steps=24,max_tool_calls=6):
        self.authorized(run_id,contract_revision)
        status=self.gateway.status(run_id)
        if not isinstance(status.get("summary"),str):
            raise WorkflowConflict("authoritative confirmed goal missing")
        contract={**self.contract,"goal":status["summary"]}
        digest=hashlib.sha256(json.dumps(contract,sort_keys=True,
                    ensure_ascii=False).encode("utf-8")).hexdigest()
        config={"configurable":{"thread_id":f"treesem-v2:m9:{run_id}"}}
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph=build_graph(self.repo,self.authorized,self.agent,contract,
                              saver,max_steps=max_steps,max_tool_calls=max_tool_calls)
            previous=graph.get_state(config)
            if not previous.values:
                graph.invoke({"run_id":run_id,"contract_revision":contract_revision},
                             config=config,durability="sync")
            elif previous.next:
                graph.invoke(None,config=config,durability="sync")
            current=graph.get_state(config)
        values=current.values
        if values and values.get("contract_digest")!=digest:
            raise WorkflowConflict("M9 checkpoint does not match confirmed Contract")
        return {"run_id":run_id,"phase":values.get("phase"),
                "outcome":values.get("outcome"),"agenda":values.get("agenda",[]),
                "agenda_revision":values.get("agenda_revision"),
                "trace":values.get("trace",[]),
                "seen_evidence":values.get("seen_evidence",[]),
                "tool_calls":values.get("tool_calls",0),
                "steps":values.get("steps",0),
                "coverage":structural_coverage(values,self.repo,contract),
                "next":list(current.next)}
