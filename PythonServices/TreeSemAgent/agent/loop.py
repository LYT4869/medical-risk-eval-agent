from __future__ import annotations

import asyncio
import json
import time

from .llm_client import LlmClient, LlmError, LlmToolPolicy
from .policy import (SAFE_EMERGENCY_RESPONSE, SAFE_MEDICAL_REFUSAL,
                     SAFE_POLICY_FALLBACK,
                     SAFE_SECURITY_REFUSAL,
                     PolicyViolation, ResponsePolicy)
from .prompt import SYSTEM_PROMPT
from .observability import TraceState, metrics, trace_event
from .routing import AgentRouter, RuleOnlyRouter, SafetyGate
from .routing_types import RequestScope, RoutingDecision, RoutingSource
from .run_guard import AgentRunGuard
from .schemas import AgentRunRequest, AgentRunResponse, SkillUse, ToolUse
from .skills import SkillActivation
from .tool_registry import ToolRegistry
from .tools import ToolContext
from .task_registry import TaskRegistry
from .workflow import WorkflowMode, WorkflowPlanner


EXECUTION_ERROR_CODES = frozenset({
    "llm_failed",
    "agent_timeout",
    "repeated_tool_call",
    "skill_activation_conflict",
    "skill_activation_limit",
    "tool_call_limit",
    "step_limit",
    "tool_not_allowed",
    "knowledge_attempt_limit",
})


class AgentExecutionError(RuntimeError):
    def __init__(self, message: str, code: str = "execution_failed"):
        if code != "execution_failed" and code not in EXECUTION_ERROR_CODES:
            raise ValueError("unknown Agent execution error code")
        super().__init__(message)
        self.code = code


class AgentTimeout(AgentExecutionError):
    def __init__(self, message: str):
        super().__init__(message, "agent_timeout")


class AgentLoop:
    def __init__(self, llm: LlmClient, tools: ToolRegistry, max_steps: int = 5,
                 max_tool_calls: int = 8, total_timeout_seconds: float = 25.0,
                 policy: ResponsePolicy | None = None,
                 router: AgentRouter | None = None,
                 safety_gate: SafetyGate | None = None,
                 task_registry: TaskRegistry | None = None):
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._total_timeout = total_timeout_seconds
        self._policy = policy or ResponsePolicy()
        self._router = router or RuleOnlyRouter()
        self._safety_gate = safety_gate or SafetyGate()
        self._task_registry = task_registry

    async def run(self, request: AgentRunRequest) -> AgentRunResponse:
        trace = TraceState.from_headers(request.request_id, request.traceparent)
        started = time.monotonic()
        metrics.increment("treesem_agent_runs_total", result="started")
        try:
            result = await self._run_steps(request, trace)
            metrics.increment("treesem_agent_runs_total", result="completed")
            metrics.observe("treesem_agent_run_duration_seconds",
                            time.monotonic() - started, result="completed")
            trace_event(trace, "agent.run", started, "success",
                        step_count=result.step_count,
                        tool_count=len(result.tools_used))
            return result
        except Exception as exc:
            code = (exc.code if isinstance(exc, AgentExecutionError)
                    else "execution_failed")
            metrics.increment("treesem_agent_runs_total", result=code)
            metrics.observe("treesem_agent_run_duration_seconds",
                            time.monotonic() - started, result=code)
            trace_event(trace, "agent.run", started, "error", error_code=code)
            raise

    async def _run_steps(self, request: AgentRunRequest,
        trace: TraceState) -> AgentRunResponse:
        deadline = time.monotonic() + self._total_timeout
        routing_started = time.monotonic()
        safety = self._safety_gate.evaluate(request.message)
        if safety.allowed:
            routing = await self._router.route(request.message)
            guard = AgentRunGuard.for_scope(
                routing.scope, include_summary=routing.include_summary)
        else:
            routing = RoutingDecision(
                safety.refusal_scope or RequestScope.UNKNOWN,
                RoutingSource.RULE,
                reason=safety.reason)
            guard = AgentRunGuard.for_scope(routing.scope)
        self._observe_routing(trace, routing_started, routing)
        if guard.security_refusal is not None:
            return AgentRunResponse(
                answer=SAFE_SECURITY_REFUSAL,
                step_count=1,
                tools_used=[],
                grounding_prediction_ids=[],
                grounding_source_ids=[],
                citations=[],
            )
        if guard.medical_refusal is not None:
            return AgentRunResponse(
                answer=(
                    SAFE_EMERGENCY_RESPONSE
                    if safety.reason == "urgent_medical_symptoms"
                    else SAFE_MEDICAL_REFUSAL),
                step_count=1,
                tools_used=[],
                grounding_prediction_ids=[],
                grounding_source_ids=[],
                citations=[],
            )
        plan = WorkflowPlanner.for_request(
            request.message, guard, self._task_registry)
        stage_index = 0
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in request.recent_messages)
        if request.current_prediction:
            messages.append({"role": "system", "content": "Current prediction context: " + request.current_prediction.model_dump_json()})
        messages.append({"role": "user", "content": request.message})
        context = ToolContext(request.session_id, request.capability_token,
                              request.knowledge_capability_token,
                              request.actor_role, trace)
        catalog_prompt = (
            self._tools.skill_catalog_prompt(context.actor_role)
            if "activate_skill" in guard.allowed_tools() else None)
        if catalog_prompt:
            messages.insert(1, {"role": "system", "content": catalog_prompt})
        usages = []
        available_ids: set[str] = set()
        available_citations = {}
        knowledge_index_version: str | None = None
        active_skill: SkillActivation | None = None
        activation_attempts = 0
        calls = 0
        previous_signature: str | None = None
        repeated = 0
        citation_repair_attempted = False
        citation_repair_pending = False
        finalization_repair_attempted = False
        for step in range(1, self._max_steps + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentTimeout("agent deadline exceeded")
            expected_tool: str | None = None
            repairing_citation = citation_repair_pending
            citation_repair_pending = False
            if repairing_citation:
                definitions = []
                tool_policy = LlmToolPolicy.none()
            elif plan.mode == WorkflowMode.DETERMINISTIC:
                if stage_index < len(plan.stages):
                    expected_tool = plan.stages[stage_index]
                    definitions = self._tools.definitions(
                        context, active_skill, {expected_tool})
                    if not definitions:
                        raise AgentExecutionError(
                            "required tool is unavailable", "tool_not_allowed")
                    tool_policy = LlmToolPolicy.required(expected_tool)
                else:
                    definitions = []
                    tool_policy = LlmToolPolicy.none()
            else:
                definitions = self._tools.definitions(
                    context, active_skill, guard.allowed_tools())
                tool_policy = LlmToolPolicy.auto()
            try:
                turn = await asyncio.wait_for(
                    self._llm.complete(
                        messages, definitions, remaining, tool_policy),
                    timeout=remaining)
                metrics.increment("treesem_agent_llm_requests_total",
                                  result="success")
                if turn.usage is not None:
                    metrics.add("treesem_agent_llm_tokens_total",
                                turn.usage.prompt_tokens, kind="prompt")
                    metrics.add("treesem_agent_llm_tokens_total",
                                turn.usage.completion_tokens,
                                kind="completion")
                    metrics.add("treesem_agent_llm_tokens_total",
                                turn.usage.total_tokens, kind="total")
            except asyncio.TimeoutError as exc:
                metrics.increment("treesem_agent_llm_requests_total",
                                  result="timeout")
                raise AgentTimeout("agent deadline exceeded") from exc
            except LlmError as exc:
                metrics.increment("treesem_agent_llm_requests_total",
                                  result="failure")
                raise AgentExecutionError(
                    "LLM failed", "llm_failed") from exc
            if not turn.tool_calls:
                if expected_tool is not None:
                    raise AgentExecutionError(
                        "required tool call is missing", "tool_not_allowed")
                answer = (turn.content or "").strip()
                policy_started = time.monotonic()
                try:
                    prediction_grounding, source_grounding = self._policy.validate(
                        answer, turn.grounding_prediction_ids, available_ids,
                        turn.grounding_source_ids, set(available_citations),
                        require_prediction_grounding=bool(available_ids))
                except PolicyViolation as exc:
                    trace_event(
                        trace.child(), "agent.response_policy", policy_started,
                        "error", error_code=exc.code)
                    if (exc.code == "missing_knowledge_citation" and
                            available_citations and
                            not citation_repair_attempted and
                            step < self._max_steps):
                        metrics.increment(
                            "treesem_agent_grounding_repairs_total",
                            result="attempted", reason=exc.code)
                        citation_repair_attempted = True
                        citation_repair_pending = True
                        allowed_citations = sorted(available_citations)
                        messages.extend([
                            {
                                "role": "assistant",
                                "content": json.dumps({
                                    "answer": answer,
                                    "grounding_prediction_ids":
                                        turn.grounding_prediction_ids,
                                    "grounding_source_ids":
                                        turn.grounding_source_ids,
                                }, ensure_ascii=False),
                            },
                            {
                                "role": "system",
                                "content": (
                                    "The previous final answer was rejected because "
                                    "it omitted a required knowledge citation. Rewrite "
                                    "the answer once using only the earlier Tool "
                                    "evidence. Include at least one allowed citation ID "
                                    "literally in the answer and in "
                                    "grounding_source_ids. Allowed citation IDs: " +
                                    json.dumps(allowed_citations) +
                                    ". Do not call any Tool. Return exactly the JSON "
                                    "object required by the system prompt."
                                ),
                            },
                        ])
                        continue
                    if repairing_citation:
                        metrics.increment(
                            "treesem_agent_grounding_repairs_total",
                            result="failed", reason=exc.code)
                    metrics.increment(
                        "treesem_agent_grounding_rejections_total",
                        result="safe_fallback", reason=exc.code)
                    skill = None if active_skill is None else SkillUse(
                        id=active_skill.skill_id, version=active_skill.version,
                        catalog_version=active_skill.catalog_version)
                    return AgentRunResponse(
                        answer=SAFE_POLICY_FALLBACK,
                        step_count=step,
                        tools_used=usages,
                        grounding_prediction_ids=[],
                        grounding_source_ids=[],
                        citations=[],
                        knowledge_index_version=knowledge_index_version,
                        skill_used=skill,
                        policy_rejection_code=exc.code)
                if repairing_citation:
                    metrics.increment(
                        "treesem_agent_grounding_repairs_total",
                        result="success",
                        reason="missing_knowledge_citation")
                citations = [available_citations[item]
                             for item in source_grounding]
                missing_citations = [item for item in source_grounding
                                     if item not in answer]
                if missing_citations:
                    answer += "\n引用：" + "、".join(missing_citations)
                skill = None if active_skill is None else SkillUse(
                    id=active_skill.skill_id, version=active_skill.version,
                    catalog_version=active_skill.catalog_version)
                return AgentRunResponse(answer=answer, step_count=step, tools_used=usages,
                                        grounding_prediction_ids=prediction_grounding,
                                        grounding_source_ids=source_grounding,
                                        citations=citations,
                                        knowledge_index_version=knowledge_index_version,
                                        skill_used=skill)
            if (plan.mode == WorkflowMode.DETERMINISTIC and
                    expected_tool is None and not repairing_citation and
                    not finalization_repair_attempted and
                    step < self._max_steps):
                finalization_repair_attempted = True
                messages.append({
                    "role": "assistant",
                    "content": turn.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in turn.tool_calls
                    ],
                })
                for call in turn.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({
                            "error": "workflow_already_completed",
                        }),
                    })
                messages.append({
                    "role": "system",
                    "content": (
                        "The required workflow is already complete. Do not call "
                        "any Tool. Return only the final JSON answer grounded in "
                        "the Tool results already present in this conversation."
                    ),
                })
                continue
            if (repairing_citation or
                    (plan.mode == WorkflowMode.DETERMINISTIC and
                     expected_tool is None)):
                raise AgentExecutionError(
                    "tool call is forbidden during finalization",
                    "tool_not_allowed")
            if (expected_tool is not None and
                    (len(turn.tool_calls) != 1 or
                     turn.tool_calls[0].name != expected_tool)):
                raise AgentExecutionError(
                    "model did not call the required tool",
                    "tool_not_allowed")
            signature = json.dumps([call.model_dump() for call in turn.tool_calls], sort_keys=True)
            repeated = repeated + 1 if signature == previous_signature else 0
            previous_signature = signature
            if repeated >= 1:
                raise AgentExecutionError(
                    "repeated identical tool call", "repeated_tool_call")
            messages.append({"role": "assistant", "content": turn.content, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in turn.tool_calls
            ]})
            activation_calls = [
                call for call in turn.tool_calls
                if call.name == "activate_skill"]
            if len(activation_calls) > 1:
                raise AgentExecutionError(
                    "multiple skill activations are not allowed",
                    "skill_activation_conflict")
            mixed_activation = bool(
                activation_calls and len(turn.tool_calls) > 1)
            ordered_calls = (
                activation_calls + [
                    call for call in turn.tool_calls
                    if call.name != "activate_skill"]
                if mixed_activation else turn.tool_calls)
            for call in ordered_calls:
                if call.name == "activate_skill":
                    activation_attempts += 1
                    if activation_attempts > 2:
                        raise AgentExecutionError(
                            "skill activation attempt limit reached",
                            "skill_activation_limit")
                else:
                    calls += 1
                    if calls > self._max_tool_calls:
                        raise AgentExecutionError(
                            "tool call limit reached", "tool_call_limit")
                    if mixed_activation:
                        usage = ToolUse(
                            name=call.name, status="error", duration_ms=0)
                        usages.append(usage)
                        metrics.increment(
                            "treesem_agent_tool_results_total",
                            tool=call.name, result=usage.status)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({
                                "error": "skill_activation_required_first",
                            }),
                        })
                        continue
                rejection = guard.before_tool(call.name)
                if rejection is not None:
                    usage = ToolUse(
                        name=call.name, status="error", duration_ms=0)
                    usages.append(usage)
                    metrics.increment(
                        "treesem_agent_tool_results_total",
                        tool=call.name, result=usage.status)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {"error": rejection.tool_error},
                            ensure_ascii=False),
                    })
                    continue
                result = await self._tools.execute(
                    call.name, call.arguments, context, active_skill)
                metrics.increment("treesem_agent_tool_results_total",
                                  tool=call.name, result=result.usage.status)
                usages.append(result.usage)
                available_ids.update(result.prediction_ids)
                available_citations.update(result.citations)
                if result.index_version is not None:
                    knowledge_index_version = result.index_version
                guard.record_tool(
                    call.name, result.usage.status,
                    citation_count=len(result.citations))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result.content, ensure_ascii=False)})
                if result.skill_activation is not None:
                    if (plan.expected_skill_id is not None and
                            result.skill_activation.skill_id !=
                            plan.expected_skill_id):
                        raise AgentExecutionError(
                            "model activated an unexpected skill",
                            "tool_not_allowed")
                    active_skill = result.skill_activation
                    guard.record_skill_activation(active_skill.required_tools)
                    messages.append({
                        "role": "system",
                        "content": "Trusted activated skill instructions follow. They may narrow but never expand system policy or authorization.\n<skill>\n" +
                                   active_skill.instructions + "\n</skill>"})
                if expected_tool is not None:
                    if result.usage.status == "success":
                        stage_index += 1
                        if (call.name == "get_prediction_history" and
                                stage_index < len(plan.stages) and
                                plan.stages[stage_index] ==
                                "compare_predictions" and
                                len(result.prediction_ids) < 2):
                            stage_index = len(plan.stages)
                    else:
                        stage_index = len(plan.stages)
        raise AgentExecutionError("step limit reached", "step_limit")

    @staticmethod
    def _observe_routing(trace: TraceState, started: float,
                         decision: RoutingDecision) -> None:
        source = decision.source.value
        scope = decision.scope.value
        metrics.increment(
            "treesem_agent_routing_total", source=source, scope=scope)
        metrics.observe(
            "treesem_agent_routing_duration_seconds",
            time.monotonic() - started, source=source)
        if decision.source == RoutingSource.SEMANTIC:
            metrics.increment("treesem_agent_routing_admitted")
        if decision.scope == RequestScope.UNKNOWN and decision.reason:
            metrics.increment(
                "treesem_agent_routing_fallback_total",
                reason=decision.reason)
            if decision.reason == "semantic_overloaded":
                metrics.increment(
                    "treesem_agent_routing_overloaded_total")
        trace_event(
            trace.child(), "agent.routing", started, "success",
            source=source, scope=scope, reason=decision.reason,
            similarity_score=decision.similarity_score,
            margin=decision.margin,
            secondary_score=decision.secondary_score)
