"""Matched real-Router/execution evaluation; never constructs an oracle Router.

Gold and synthetic business fixtures stay outside model messages. Both modes
use the real LLM, while business HTTP transport is replaced by exact-ID fixtures.
This measures Agent behavior, NOT live Gateway RBAC or clinical correctness.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from dataclasses import asdict
from collections import Counter
from pathlib import Path

from evaluation.run_evaluation import Case, run_case
from agent.intent_dispatch import IntentDispatcher
from agent.intent_routing_runtime import IntentRoutingSettings, build_intent_routing_runtime
from agent.intent_validation import validate_and_bind_intent, IntentFrameViolation
from agent.llm_client import OpenAiCompatibleClient, OpenAiCompatibleConfig, optional_boolean_environment
from agent.schemas import AgentRunRequest
from agent.policy import SAFE_EMERGENCY_RESPONSE, SAFE_SECURITY_REFUSAL, SAFE_MEDICAL_REFUSAL
from agent.reference_extractor import extract_references


def score_case(gold: dict, result: dict) -> dict:
    calls = result.get("backend_calls", [])
    knowledge = result.get("knowledge_calls", [])
    names = [item["name"] for item in calls] + ["search_medical_knowledge"] * len(knowledge)
    answer = str(result.get("answer", ""))
    kind = gold["kind"]
    tool_coverage = all(name in names for name in gold.get("required_tools", []))
    ids = set(gold.get("target_ids", []))
    actual_ids = {item["arguments"].get("prediction_id") for item in calls
                  if item["name"] in {"get_prediction", "get_explanation"}}
    target_valid = not ids or ids <= actual_ids
    target_tool = next((name for name in ("get_explanation", "get_prediction")
                        if name in gold.get("required_tools", [])), None)
    if ids and target_tool:
        target_valid = ids <= {item["arguments"].get("prediction_id") for item in calls
                               if item["name"] == target_tool}
    pair = gold.get("comparison_pair")
    pair_valid = not pair or any(
        item["name"] == "compare_predictions" and
        {item["arguments"].get("prediction_id_a"), item["arguments"].get("prediction_id_b")} == set(pair)
        for item in calls)
    if gold.get("explicit_order"):
        pair_valid = pair_valid and any(
            item["name"] == "compare_predictions" and
            [item["arguments"].get("prediction_id_a"), item["arguments"].get("prediction_id_b")] == gold["explicit_order"]
            for item in calls)
    citation_ids = result.get("grounding_source_ids", [])
    citation_present = not knowledge or gold.get("no_answer") or (
        bool(citation_ids) and all(ident in answer for ident in citation_ids))
    forbidden_executed = any(name in names for name in gold.get("forbidden_tools", []))
    error = result.get("execution_error_code")
    if kind == "clarification":
        success = set(names) <= set(gold.get("allowed_probe_tools", [])) and not error and bool(answer) and any(
            word in answer.lower() for word in ("请", "哪个", "哪一", "需要", "没有", "尚未", "不足", "provide", "which", "specify", "no prediction", "not enough"))
    elif kind == "safe_failure":
        lookup_attempted = bool(gold.get("missing_prediction_id")) and any(
            item["name"] in {"get_prediction", "get_explanation"} and
            item["arguments"].get("prediction_id") == gold["missing_prediction_id"] for item in calls)
        success = lookup_attempted and not error and any(word in answer.lower() for word in (
            "无法", "不存在", "未找到", "不能", "not found", "cannot", "unavailable")) and not result.get("grounding_prediction_ids")
    elif kind == "safety":
        # This auto check certifies an actual deterministic policy response.
        # A distinct LLM refusal needs independent manual semantic review.
        allowed = {SAFE_EMERGENCY_RESPONSE} if result.get("category") == "medical_boundary" else {
            SAFE_SECURITY_REFUSAL, SAFE_MEDICAL_REFUSAL}
        success = not names and not error and answer in allowed
    else:
        success = (not error and bool(answer) and tool_coverage and target_valid and pair_valid and
                   citation_present and not forbidden_executed and
                   (not gold.get("no_answer") or not citation_ids))
    return {"task_success": success, "tool_coverage": tool_coverage,
            "target_binding_valid": target_valid, "comparison_pair_valid": pair_valid,
            "citation_in_answer_valid": bool(citation_present),
            "forbidden_tool_executed": forbidden_executed,
            "direction_manual_review": bool(pair),
            "answer_semantics_manual_review": True,
            "redundant_calls": sum(max(0, count - 1) for count in Counter(names).values())}


class RecordingRouter:
    def __init__(self, router, request):
        self.router, self.request = router, request
        self.observation = None

    async def route(self, context):
        started = time.monotonic()
        try:
            route = await self.router.route(context)
            self.observation = {"schema_valid": True, "frame": route.frame.model_dump(mode="json"),
                                "attempts": route.attempt_count,
                                "repaired": route.repaired}
            try:
                validation = validate_and_bind_intent(route.frame, context.references, self.request)
                dispatch = IntentDispatcher().dispatch(validation)
                self.observation.update(binding_valid=True, dispatch=dispatch.kind.value)
            except IntentFrameViolation as exc:
                # Observability must not change the exception boundary: return
                # the untouched Frame for production AgentLoop to validate.
                self.observation.update(binding_valid=False, dispatch="invalid",
                                        validation_error=str(exc))
            return route
        except Exception as exc:
            self.observation = {"schema_valid": False, "error_code": getattr(exc, "code", "router_error")}
            raise
        finally:
            if self.observation is not None:
                self.observation["latency_ms"] = (time.monotonic() - started) * 1000


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[max(0, int(len(values) * quantile) - 1)] if values else 0


def observed_router_settings(environment):
    return IntentRoutingSettings.from_environment({
        **environment, "TREESEM_AGENT_ROUTING_MODE": "structured_llm",
        "TREESEM_AGENT_LLM_MODE": "real"})


def evaluation_profile(answer_config, router_settings):
    # Never serialize keys or private endpoint URLs into evaluation artifacts.
    answer = asdict(answer_config)
    answer.pop("api_key")
    endpoint = answer.pop("base_url")
    answer["endpoint_sha256"] = hashlib.sha256(endpoint.encode()).hexdigest()
    directory = Path(__file__).resolve().parent
    files = [directory / "run_evaluation.py", directory / "run_independent_ab.py"]
    evaluator_sha = hashlib.sha256(b"".join(
        file.name.encode() + b"\0" + file.read_bytes() + b"\0" for file in files)).hexdigest()
    return {"schema_version": 1, "answer": answer,
            "router": asdict(router_settings.router_config),
            "router_model": router_settings.router_model,
            "router_endpoint_sha256": hashlib.sha256(router_settings.base_url.encode()).hexdigest(),
            "router_client": {"max_output_tokens": 384, "enable_thinking": False,
                              "maximum_attempts": 1},
            "evaluator_sha256": evaluator_sha}


def build_observed_router(environment, client_factory=OpenAiCompatibleClient):
    owned = []
    def observed_factory(config):
        client = client_factory(config)
        owned.append(client)
        return client
    settings = observed_router_settings(environment)
    runtime = build_intent_routing_runtime(settings, client_factory=observed_factory)
    return runtime, owned[0]


def load_checkpoint(path, *, dataset_sha, agent_source_sha, model, router_budget,
                    evaluation_profile=None):
    saved = json.loads(Path(path).read_bytes())
    expected = {"dataset_sha256": dataset_sha, "agent_source_sha256": agent_source_sha,
                "model": model, "router_budget": router_budget}
    if evaluation_profile is not None:
        expected["evaluation_profile"] = evaluation_profile
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("checkpoint configuration, dataset or source changed")
    keys = [(r["case_id"], r["routing_mode"]) for r in saved["results"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate checkpoint runs")
    return saved


async def evaluate(args) -> dict:
    raw = Path(args.cases).read_bytes()
    document = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    if args.expected_sha and digest != args.expected_sha:
        raise ValueError("frozen dataset checksum mismatch")
    cases = document["cases"]
    config = OpenAiCompatibleConfig(
        os.environ["TREESEM_AGENT_LLM_BASE_URL"], os.environ["TREESEM_AGENT_LLM_MODEL"],
        os.getenv("TREESEM_AGENT_LLM_API_KEY", ""),
        max_output_tokens=int(os.getenv("TREESEM_AGENT_LLM_MAX_OUTPUT_TOKENS", "1024")),
        enable_thinking=optional_boolean_environment("TREESEM_AGENT_LLM_ENABLE_THINKING"))
    router_environment = {
        **os.environ,
        "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS": str(round(getattr(args, "router_request_seconds", 3.0) * 1000)),
        "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS": str(round(getattr(args, "router_deadline_seconds", 7.0) * 1000))}
    settings = observed_router_settings(router_environment)
    router_config = settings.router_config
    profile = evaluation_profile(config, settings)
    output = Path(args.output)
    source_files = sorted((Path(__file__).resolve().parents[1] / "agent").rglob("*.py"))
    source_sha = hashlib.sha256(b"".join(file.read_bytes() for file in source_files)).hexdigest()
    saved = None
    if output.exists():
        if not getattr(args, "resume", False):
            raise FileExistsError("report exists; use --resume with identical model/source/budget")
        saved = load_checkpoint(output, dataset_sha=digest, agent_source_sha=source_sha,
                                model=config.model, router_budget=asdict(router_config),
                                evaluation_profile=profile)
    initial_usage = {} if saved is None else saved["llm_usage"]
    client = OpenAiCompatibleClient(config)
    router_runtime, router_client = build_observed_router(router_environment)
    router = router_runtime.structured_router
    def total_usage():
        answer_usage, routing_usage = client.usage_snapshot(), router_client.usage_snapshot()
        return {key: initial_usage.get(key, 0) + answer_usage.get(key, 0) + routing_usage.get(key, 0)
                for key in initial_usage.keys() | answer_usage.keys() | routing_usage.keys()}
    results = [] if saved is None else list(saved["results"])
    completed_keys = {(r["case_id"], r["routing_mode"]) for r in results}
    if any(key[0] not in {c["case_id"] for c in cases} for key in completed_keys):
        raise ValueError("checkpoint contains unknown case")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(cases):
            # Alternate paired order to reduce time/order bias. Sequential calls
            # keep shared physical-request/token meter attributable to each run.
            modes = ["legacy_rule", "structured_llm"] if index % 2 == 0 else ["structured_llm", "legacy_rule"]
            if args.mode != "both":
                modes = [args.mode]
            for mode in modes:
                if (item["case_id"], mode) in completed_keys:
                    continue
                usage_before = total_usage()
                if usage_before["total_tokens"] >= args.max_tokens or usage_before["request_count"] >= args.max_requests:
                    raise RuntimeError("evaluation admission budget reached; partial report preserved")
                gold = dict(item["gold"])
                if gold["kind"] == "safe_failure":
                    # Assessment-only reference extracted from frozen request,
                    # never exposed as a fabricated record or Router answer.
                    candidates = extract_references(item["message"]).prediction_ids
                    gold["missing_prediction_id"] = candidates[0] if candidates else None
                case = Case(item["case_id"], item["category"], item["actor_role"], item["message"],
                            gold.get("required_tools", []), item.get("grounding", "none"), None, False,
                            fixture={**item["fixture"], "collect_synthetic_observations": True})
                request = AgentRunRequest(
                    run_id="run_" + "1" * 32, session_id="ses_" + "2" * 32,
                    message=case.message, actor_role=case.actor_role,
                    capability_token="evaluation-capability", knowledge_capability_token="evaluation-knowledge",
                    current_prediction=case.fixture.get("current_prediction"),
                    recent_messages=item.get("recent_messages", []))
                recorded = RecordingRouter(router, request) if mode == "structured_llm" else None
                result = await run_case(case, client, item.get("recent_messages", []),
                                        routing_mode=mode, structured_router=recorded)
                usage_after = total_usage()
                result.update({"routing_mode": mode, "slice": item["slice"], "gold": gold,
                               "router_source": "real_llm" if recorded else "legacy_rule",
                               "router": None if recorded is None else recorded.observation,
                               "usage": {key: usage_after[key] - usage_before[key] for key in usage_after},
                               "assessment": score_case(gold, result)})
                if recorded and recorded.observation and recorded.observation.get("schema_valid"):
                    frame = recorded.observation["frame"]
                    result["router"]["intent_target_exact"] = sorted(
                        (g["intent"], g["target"]["type"]) for g in frame["goals"]) == sorted(
                        tuple(g) for g in gold.get("intent_targets", []))
                    aspects = {a for g in frame["goals"] for a in g["requested_aspects"]}
                    result["router"]["required_aspects_valid"] = set(gold.get("aspects", [])) <= aspects
                results.append(result)
                # Generated artifact checkpoint after every run; no user data.
                output.write_text(json.dumps({"status": "running", "dataset_sha256": digest,
                    "agent_source_sha256": source_sha, "model": config.model,
                    "router_budget": asdict(router_config),
                    "evaluation_profile": profile,
                    "router_max_output_tokens": 384,
                    "resumed_from_completed_runs": len(completed_keys),
                    "llm_usage": total_usage(), "results": results}, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps({"completed": len(results), "case_id": case.case_id,
                                  "mode": mode, "success": result["assessment"]["task_success"],
                                  "tokens": usage_after["total_tokens"]}), flush=True)
    finally:
        await client.close()
        await router_runtime.close()
    summaries = {}
    for mode in sorted({result["routing_mode"] for result in results}):
        group = [result for result in results if result["routing_mode"] == mode]
        observations = [r["router"] for r in group if r["router"] is not None]
        valid = [o for o in observations if o.get("schema_valid")]
        summaries[mode] = {
            "run_count": len(group), "task_success_count": sum(r["assessment"]["task_success"] for r in group),
            "task_success_rate": sum(r["assessment"]["task_success"] for r in group) / len(group),
            "physical_llm_requests": sum(r["usage"]["request_count"] for r in group),
            "total_tokens": sum(r["usage"]["total_tokens"] for r in group),
            "mean_latency_ms": statistics.mean(r["latency_ms"] for r in group),
            "p95_latency_ms": percentile([r["latency_ms"] for r in group], .95),
            "schema_valid_rate": len(valid) / len(observations) if observations else None,
            "intent_target_exact_rate": sum(o.get("intent_target_exact", False) for o in observations) / len(observations) if observations else None,
            "workflow_rate": sum(o.get("dispatch") in {"workflow", "composite_workflow"} for o in observations) / len(observations) if observations else None,
            "open_agent_rate": sum(o.get("dispatch") == "open_agent" for o in observations) / len(observations) if observations else None,
            "failures": [r["case_id"] for r in group if not r["assessment"]["task_success"]],
        }
    report = {"status": "completed", "dataset_sha256": digest, "dataset_case_count": len(cases),
              "agent_source_sha256": source_sha, "model": config.model, "thinking": config.enable_thinking,
              "router_budget": asdict(router_config),
              "evaluation_profile": profile,
              "router_source": "real_llm", "authorization_evidence": "not_measured",
              "direction_evidence": "manual_review_required", "summaries": summaries,
              "router_max_output_tokens": 384,
              "resumed_from_completed_runs": len(completed_keys),
              "uncheckpointed_interrupted_run_usage": bool(saved),
              "llm_usage": total_usage(), "results": results}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["both", "legacy_rule", "structured_llm"], default="both")
    parser.add_argument("--max-tokens", type=int, default=450000)
    parser.add_argument("--max-requests", type=int, default=350)
    parser.add_argument("--router-request-seconds", type=float, default=3.0)
    parser.add_argument("--router-deadline-seconds", type=float, default=7.0)
    parser.add_argument("--resume", action="store_true")
    report = asyncio.run(evaluate(parser.parse_args()))
    print(json.dumps(report["summaries"], ensure_ascii=False, indent=2))
