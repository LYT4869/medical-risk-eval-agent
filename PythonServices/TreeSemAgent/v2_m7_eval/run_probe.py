"""M7-B bounded local LLM component probe; synthetic P001 only, no DB writes."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from v2_m1.run_store import WorkflowConflict
from v2_m3.agent import FakeInvestigationAgent, OpenAICompatibleInvestigationAgent
from v2_m3.policy import M3FixtureTools, checked_decision, initial_agenda
from v2_m5.catalog import FACTS, plan
from v2_m5.grounding import review_document
from v2_m5.writer import FakeParagraphWriter, LocalParagraphWriter

def local_endpoint(endpoint):
    parts = urlsplit(endpoint)
    if (parts.scheme not in ("http", "https") or
        parts.hostname not in ("127.0.0.1", "localhost", "::1") or
        not parts.port or parts.path != "/v1/chat/completions" or
        parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("explicit loopback OpenAI-compatible URL and port required")
    return endpoint

@dataclass
class Completion:
    raw: str
    model: str
    latency_ms: float
    usage: dict

class LocalChat:
    """Mirror the existing M3/M5 system/user JSON request boundary."""
    def __init__(self, endpoint, model, timeout=30):
        self.endpoint = local_endpoint(endpoint)
        if not isinstance(model, str) or not model.strip() or len(model)>128 or not 0<timeout<=45:
            raise ValueError("invalid model or timeout")
        self.model, self.timeout = model, timeout

    def complete(self, system, context):
        req = Request(self.endpoint,
            data=json.dumps({"model":self.model,"temperature":0,
                "messages":[{"role":"system","content":system},
                            {"role":"user","content":json.dumps(context,ensure_ascii=False)}]
            },ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type":"application/json"},method="POST")
        start=time.perf_counter()
        with urlopen(req,timeout=self.timeout) as response:
            body=response.read(131073)
        elapsed=round((time.perf_counter()-start)*1000,3)
        if len(body)>131072:
            raise ValueError("provider response too large")
        result=json.loads(body)
        raw=result["choices"][0]["message"]["content"]
        if not isinstance(raw,str) or len(raw)>30000:
            raise ValueError("invalid provider content")
        usage=result.get("usage") or {}
        if not isinstance(usage,dict):
            usage={}
        return Completion(raw,str(result.get("model") or self.model)[:128],elapsed,
            {k:usage.get(k) if type(usage.get(k)) is int and usage[k]>=0 else None
             for k in ("prompt_tokens","completion_tokens","total_tokens")})

class FixtureChat:
    """CI scripted transport; never identified as a real LLM."""
    def __init__(self, replies):
        self.replies=iter(replies)
    def complete(self, system, context):
        return Completion(next(self.replies),"fixture-not-model",0.0,
            {"prompt_tokens":None,"completion_tokens":None,"total_tokens":None})

def build_case(case_id, issue_id, timeline, observed=(), seen=(), calls=0):
    agenda=initial_agenda()
    if issue_id=="I-04":
        agenda.append({"issue_id":"I-04","topic":"source_discrepancy",
                       "status":"todo","evidence_ids":[],"result_key":None})
    issue=next(x for x in agenda if x["issue_id"]==issue_id)
    issue["evidence_ids"]=list(observed)
    for entry in agenda:
        if entry["issue_id"]!=issue_id:
            entry["status"]="done"
    state={"timeline_loaded":timeline,"tool_calls":calls,
           "seen_evidence":list(seen),"agenda":agenda}
    tools=M3FixtureTools()
    summaries=[]
    for ref in observed:
        record=tools.records[ref]
        summaries.append({"evidence_id":ref,"source_type":record.get("kind"),
                          "version":record.get("version"),
                          "blood_loss_ml":record.get("blood_loss_ml"),
                          "approximate":record.get("approximate",False)})
    context={"contract_goal":"P001 SIM：核查既往分娩记录，不编造诊断",
        "skill":{"id":"perinatal-investigation","version":"SIM-v1",
                 "constraints":["preserve_source_discrepancy",
                                "never_infer_prior_pph_diagnosis",
                                "no_unverified_prenatal_model"]},
        "risk_profile":{"id":"PPH-SIM","version":1},
        "issue":deepcopy(issue),"evidence_summaries":summaries,
        "timeline_loaded":timeline,
        "available_tools":["get_patient_timeline","read_patient_record"],
        "remaining_tool_calls":5-calls}
    return {"id":case_id,"issue":issue,"state":state,"prompt":context,
            "golden":FakeInvestigationAgent().decide(context)}

def cases():
    return [
        build_case("I01-timeline","I-01",False),
        build_case("I01-read","I-01",True,calls=1),
        build_case("I02-missing-source","I-02",True,("E-002",),
                   ("E-001","E-002"),3),
        build_case("I02-discrepancy","I-02",True,("E-002","E-003"),
                   ("E-001","E-002","E-003"),4),
        build_case("I03-block-model","I-03",True,(),
                   ("E-001","E-002","E-003"),4),
        build_case("I04-unresolved","I-04",True,("E-002","E-003"),
                   ("E-001","E-002","E-003"),4),
    ]

def writer_prompt():
    return {"classification":"SIM",
        "confirmed_goal":"P001 SIM：既往记录、LIS 更正、来源分歧",
        "contract_revision":1,"section_plan":plan(),
        "available_claims":[{"fact_id":fid,"statement":info["full"],
             "candidate_evidence_refs":info["evidence_ids"]}
             for fid,info in FACTS.items()],
        "source_snapshot":["E-001:v1","E-002:v1","E-003:v1",
                           "E-004:v1:historical_stale","E-005:v2"],
        "constraints":{"no_diagnosis":True,"model_score":None,
                       "only_historical_E004":True}}

def synthetic_review_context():
    tools=M3FixtureTools()
    return SimpleNamespace(
        evidence={ref:tools.records[ref] for ref in ("E-001","E-002","E-003")},
        sources={"E-004":{"logical_id":"LIS-P001-HB-20261010","version":1,
                          "hb_g_l":110,"validity":"stale"},
                 "E-005":{"logical_id":"LIS-P001-HB-20261010","version":2,
                          "hb_g_l":130,"validity":"current"}},
        statements={"S-04":{"validity":"stale"},"S-05":{"validity":"current"}},
        issues={"I-01":{"no_diagnosis":True},"I-03":{"status":"blocked"},
                "I-04":{"status":"closed_unresolved"}},
        overlay={"model_capability":"not_verified_for_prenatal"})

def metadata(response, save_raw):
    return {"provider_model":response.model,"latency_ms":response.latency_ms,
            "usage":response.usage,"raw_output":response.raw if save_raw else None}

def evaluate_agent(case,response,save_raw=True):
    row={"case_id":case["id"],"stage":"agent",**metadata(response,save_raw),
         "outcome":None,"guard_pass":False,"golden_match":False}
    try:
        value=json.loads(response.raw)
    except (ValueError,TypeError):
        row["outcome"]="invalid_json"
        return row
    if not isinstance(value,dict):
        row["outcome"]="schema_mismatch"
        return row
    try:
        checked=checked_decision(value,case["issue"],case["state"])
    except (WorkflowConflict,KeyError,TypeError,ValueError):
        row["outcome"]="guard_rejected"
        return row
    row["guard_pass"]=True
    row["golden_match"]=checked==case["golden"]
    row["outcome"]="golden_match" if row["golden_match"] else "golden_mismatch"
    return row

def evaluate_writer(response,save_raw=True):
    row={"case_id":"P001-six-sections","stage":"writer",
         **metadata(response,save_raw),"outcome":None,
         "schema_pass":False,"catalog_pass":False,
         "semantic_correctness":"not_evaluated"}
    try:
        proposals=json.loads(response.raw)
    except (ValueError,TypeError):
        row["outcome"]="invalid_json"
        return row
    columns={"section_id","paragraph_id","text",
             "candidate_evidence_refs","fact_ids"}
    if (not isinstance(proposals,list) or len(proposals)!=6 or
        any(not isinstance(x,dict) or set(x)!=columns or
            not isinstance(x["text"],str) for x in proposals)):
        row["outcome"]="schema_mismatch"
        return row
    row["schema_pass"]=True
    try:
        _,result=review_document(proposals,synthetic_review_context())
    except (WorkflowConflict,KeyError,TypeError,ValueError):
        row["outcome"]="catalog_rejected"
        return row
    row["catalog_pass"]=result["total_assertions"]==10
    row["outcome"]="catalog_pass" if row["catalog_pass"] else "catalog_rejected"
    return row

def probe(client,mode,save_raw=True):
    if mode not in ("live","fixture"):
        raise ValueError("unknown evaluation mode")
    rows=[]
    selected=cases()
    for case in selected:
        try:
            result=client.complete(OpenAICompatibleInvestigationAgent.SYSTEM,
                                   case["prompt"])
            rows.append(evaluate_agent(case,result,save_raw))
        except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration):
            rows.append({"case_id":case["id"],"stage":"agent",
                         "outcome":"provider_error","guard_pass":False,
                         "golden_match":False})
    try:
        result=client.complete(LocalParagraphWriter.SYSTEM,writer_prompt())
        rows.append(evaluate_writer(result,save_raw))
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration):
        rows.append({"case_id":"P001-six-sections","stage":"writer",
                     "outcome":"provider_error","schema_pass":False,
                     "catalog_pass":False,"semantic_correctness":"not_evaluated"})
    return {"schema":"treesem.v2.m7b.probe.v1","classification":"SIM",
        "fixture":"P001","mode":mode,"real_llm_called":mode=="live",
        "scope":"isolated_components_not_e2e_or_clinical_nli",
        "cases":len(rows),"agent_cases":len(selected),
        "golden_matches":sum(x.get("golden_match",False) for x in rows),
        "writer_catalog_pass":any(x.get("catalog_pass",False) for x in rows),
        "records":rows}

def fixture_client():
    replies=[json.dumps(c["golden"],ensure_ascii=False) for c in cases()]
    replies.append(json.dumps(FakeParagraphWriter().write(writer_prompt()),
                              ensure_ascii=False))
    return FixtureChat(replies)

def write_report(path,report):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=None
    try:
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",
             prefix=".m7b-",suffix=".tmp",dir=path.parent,delete=False) as f:
            tmp=f.name
            json.dump(report,f,ensure_ascii=False,indent=2)
            f.write("\n")
        os.chmod(tmp,0o600)
        os.replace(tmp,path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

def main():
    parser=argparse.ArgumentParser(description="M7-B P001 synthetic LLM probe")
    parser.add_argument("--mode",choices=("fixture","live"),required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--out",required=True)
    parser.add_argument("--redact-raw",action="store_true")
    args=parser.parse_args()
    if args.mode=="live":
        if not args.endpoint or not args.model:
            parser.error("live mode requires explicit --endpoint and --model")
        client=LocalChat(args.endpoint,args.model)
    else:
        if args.endpoint or args.model:
            parser.error("fixture mode does not accept a model or endpoint")
        client=fixture_client()
    result=probe(client,args.mode,save_raw=not args.redact_raw)
    write_report(args.out,result)
    print(json.dumps({"mode":result["mode"],"cases":result["cases"],
         "golden_matches":result["golden_matches"],
         "writer_catalog_pass":result["writer_catalog_pass"],
         "report_path":str(args.out)},ensure_ascii=False))

if __name__=="__main__":
    main()
