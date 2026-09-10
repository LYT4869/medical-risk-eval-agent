from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

from .llm_client import LlmClient, LlmError, LlmToolPolicy
from .deterministic_workflow import (
    DeterministicWorkflowExecutor,
    WorkflowExecution,
)
from .intent_dispatch import DispatchKind, IntentDispatcher
from .intent_validation import (
    IntentFrameViolation,
    validate_and_bind_intent,
)
from .policy import (SAFE_EMERGENCY_RESPONSE, SAFE_MEDICAL_REFUSAL,
                     SAFE_POLICY_FALLBACK,
                     SAFE_SECURITY_REFUSAL,
                     PolicyViolation, ResponsePolicy)
from .prompt import SYSTEM_PROMPT
from .observability import TraceState, metrics, trace_event
from .routing import AgentRouter, RuleOnlyRouter, SafetyGate
from .routing_types import RequestScope, RoutingDecision, RoutingSource
from .reference_extractor import extract_references
from .run_guard import AgentRunGuard
from .schemas import (
    AgentRunRequest,
    AgentRunResponse,
    LlmUsage,
    SkillUse,
    ToolUse,
)
from .skills import SkillActivation
from .tool_registry import ToolRegistry
from .tools import ToolContext
from .task_registry import TaskRegistry
from .workflow import WorkflowMode, WorkflowPlanner
from .structured_router import (
    RouterContext,
    StructuredIntentRouter,
    StructuredRouterError,
)
from .workflow_registry import RendererKind


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
    "intent_router_unavailable",
    "invalid_intent_frame",
})


_CLARIFICATIONS = {
    "sample_index_missing": "请明确要预测的演示样本索引。",
    "current_prediction_missing": "当前会话还没有可解释的预测，请先完成一次预测。",
    "prediction_target_missing": "请说明要查看哪一条预测记录。",
    "comparison_target_missing": "比较至少需要两条可识别的预测记录。",
    "ambiguous_reference": "我还不能确定你指的是哪一条预测，请明确是当前、上一次或给出预测 ID。",
    "conflicting_request": "你的要求中存在相互冲突的部分，请明确希望保留哪一项。",
}

_WORKFLOW_FALLBACKS = {
    RendererKind.PREDICTION_COMPLETED:
        "预测已经完成，但自然语言说明暂时不可用。",
    RendererKind.PREDICTION_DATA_AVAILABLE:
        "预测数据已经取得，但自然语言说明暂时不可用。",
    RendererKind.EXPLANATION_DATA_AVAILABLE:
        "解释数据已经取得，但自然语言说明暂时不可用。",
    RendererKind.HISTORY_DATA_AVAILABLE:
        "历史数据已经取得，但自然语言说明暂时不可用。",
    RendererKind.COMPARISON_DATA_AVAILABLE:
        "比较数据已经取得，但自然语言说明暂时不可用。",
    RendererKind.KNOWLEDGE_TEMPORARILY_UNRENDERED:
        "知识资料已经取得，但自然语言说明暂时不可用。",
}


@dataclass
class _StructuredRunAccounting:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_duration_seconds: float = 0.0

    def add_calls(self, count: int, duration: float,
                  usage: LlmUsage | None = None) -> None:
        self.llm_calls += count
        self.llm_duration_seconds += duration
        if usage is not None:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens

    def observe(self) -> None:
        metrics.observe(
            "treesem_agent_llm_calls_per_run", float(self.llm_calls))
        metrics.observe(
            "treesem_agent_llm_tokens_per_run", float(self.prompt_tokens),
            kind="prompt")
        metrics.observe(
            "treesem_agent_llm_tokens_per_run",
            float(self.completion_tokens), kind="completion")
        metrics.observe(
            "treesem_agent_llm_tokens_per_run", float(self.total_tokens),
            kind="total")
        metrics.observe(
            "treesem_agent_llm_duration_per_run_seconds",
            self.llm_duration_seconds)


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
                 task_registry: TaskRegistry | None = None,
                 structured_router: StructuredIntentRouter | None = None,
                 routing_mode: str = "legacy_rule"):
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._total_timeout = total_timeout_seconds
        self._policy = policy or ResponsePolicy()
        self._router = router or RuleOnlyRouter()
        self._safety_gate = safety_gate or SafetyGate()
        self._task_registry = task_registry
        if routing_mode not in {
                "legacy_rule", "structured_shadow", "structured_llm"}:
            raise ValueError("unknown Agent routing mode")
        if routing_mode == "structured_llm" and structured_router is None:
            raise ValueError("structured routing requires a Router")
        self._structured_router = structured_router
        self._routing_mode = routing_mode
        self._intent_dispatcher = IntentDispatcher()
        self._workflow_executor = DeterministicWorkflowExecutor(tools)

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
        if self._routing_mode == "structured_llm":
            return await self._run_structured_steps(request, trace)
        deadline = time.monotonic() + self._total_timeout
        routing_started = time.monotonic()
        safety = self._safety_gate.evaluate(request.message)
        if (safety.allowed and self._routing_mode == "structured_shadow" and
                self._structured_router is not None):
            await self._observe_structured_shadow(request)
        if safety.allowed:
            routing = await self._router.route(request.message)
            guard = AgentRunGuard.for_scope(
                routing.scope, registry=self._task_registry,
                include_summary=routing.include_summary)
        else:
            routing = RoutingDecision(
                safety.refusal_scope or RequestScope.UNKNOWN,
                RoutingSource.RULE,
                reason=safety.reason)
            guard = AgentRunGuard.for_scope(
                routing.scope, registry=self._task_registry)
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

    async def _observe_structured_shadow(
            self, request: AgentRunRequest) -> None:
        references = extract_references(request.message)
        context = RouterContext(
            message=references.router_message,
            recent_messages=tuple(request.recent_messages),
            current_prediction_available=request.current_prediction is not None,
            references=references,
        )
        try:
            route = await self._structured_router.route(context)
            validation = validate_and_bind_intent(
                route.frame, references, request)
            self._intent_dispatcher.dispatch(validation)
        except (StructuredRouterError, IntentFrameViolation, ValueError):
            return

    @staticmethod
    def _safety_response(safety) -> AgentRunResponse | None:
        if safety.allowed:
            return None
        if safety.refusal_scope == RequestScope.SECURITY_ABUSE:
            answer = SAFE_SECURITY_REFUSAL
        else:
            answer = (SAFE_EMERGENCY_RESPONSE
                      if safety.reason == "urgent_medical_symptoms"
                      else SAFE_MEDICAL_REFUSAL)
        return AgentRunResponse(
            answer=answer,
            step_count=1,
            tools_used=[],
            grounding_prediction_ids=[],
            grounding_source_ids=[],
            citations=[],
        )

    async def _run_structured_steps(
            self, request: AgentRunRequest,
            trace: TraceState) -> AgentRunResponse:
        accounting = _StructuredRunAccounting()
        try:
            return await self._run_structured_steps_impl(
                request, trace, accounting)
        finally:
            accounting.observe()

    async def _run_structured_steps_impl(
            self, request: AgentRunRequest, trace: TraceState,
            accounting: _StructuredRunAccounting) -> AgentRunResponse:
        safety = self._safety_gate.evaluate(request.message)
        refusal = self._safety_response(safety)
        if refusal is not None:
            return refusal
        deadline = time.monotonic() + self._total_timeout
        references = extract_references(request.message)
        context = RouterContext(
            message=references.router_message,
            recent_messages=tuple(request.recent_messages),
            current_prediction_available=request.current_prediction is not None,
            references=references,
        )
        router_started = time.monotonic()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentTimeout("agent deadline exceeded")
            route = await asyncio.wait_for(
                self._structured_router.route(context), timeout=remaining)
            router_duration = time.monotonic() - router_started
            accounting.add_calls(
                route.attempt_count, router_duration, route.usage)
            metrics.increment(
                "treesem_agent_intent_router_requests_total",
                result="success")
            metrics.observe(
                "treesem_agent_intent_router_duration_seconds",
                router_duration, result="success")
            if route.repaired:
                metrics.increment(
                    "treesem_agent_intent_router_repairs_total",
                    result="success")
            trace_event(
                trace.child(), "agent.intent_router", router_started,
                "success", attempt_count=route.attempt_count,
                repaired=route.repaired)
            validation = validate_and_bind_intent(
                route.frame, references, request)
            dispatch = self._intent_dispatcher.dispatch(validation)
        except asyncio.TimeoutError as exc:
            duration = time.monotonic() - router_started
            metrics.increment(
                "treesem_agent_intent_router_requests_total",
                result="intent_router_unavailable")
            metrics.observe(
                "treesem_agent_intent_router_duration_seconds", duration,
                result="intent_router_unavailable")
            trace_event(
                trace.child(), "agent.intent_router", router_started, "error",
                error_code="intent_router_unavailable", attempt_count=0,
                repaired=False)
            raise AgentTimeout("agent deadline exceeded") from exc
        except StructuredRouterError as exc:
            duration = time.monotonic() - router_started
            accounting.add_calls(exc.attempt_count, duration)
            metrics.increment(
                "treesem_agent_intent_router_requests_total",
                result=exc.code)
            metrics.observe(
                "treesem_agent_intent_router_duration_seconds", duration,
                result=exc.code)
            if exc.repaired:
                metrics.increment(
                    "treesem_agent_intent_router_repairs_total",
                    result="failed")
            trace_event(
                trace.child(), "agent.intent_router", router_started, "error",
                error_code=exc.code, attempt_count=exc.attempt_count,
                repaired=exc.repaired)
            raise AgentExecutionError(exc.code, exc.code) from exc
        except IntentFrameViolation as exc:
            dispatch_started = time.monotonic()
            metrics.increment(
                "treesem_agent_intent_dispatch_total", dispatch="invalid")
            trace_event(
                trace.child(), "agent.intent_dispatch", dispatch_started,
                "error", error_code="invalid_intent_frame",
                goal_count=len(route.frame.goals))
            raise AgentExecutionError(
                "Router produced an invalid intent frame",
                "invalid_intent_frame") from exc

        dispatch_started = time.monotonic()
        metrics.increment(
            "treesem_agent_intent_dispatch_total",
            dispatch=dispatch.kind.value)
        trace_event(
            trace.child(), "agent.intent_dispatch", dispatch_started,
            "success", dispatch=dispatch.kind.value,
            goal_count=len(route.frame.goals))

        if dispatch.kind == DispatchKind.CLARIFICATION:
            answer = _CLARIFICATIONS.get(dispatch.clarification_code or "")
            if answer is None:
                raise AgentExecutionError(
                    "unknown clarification result", "invalid_intent_frame")
            return AgentRunResponse(
                answer=answer,
                step_count=max(1, route.attempt_count),
                tools_used=[],
                grounding_prediction_ids=[],
                grounding_source_ids=[],
                citations=[],
            )

        tool_context = ToolContext(
            request.session_id, request.capability_token,
            request.knowledge_capability_token,
            request.actor_role, trace)
        if dispatch.kind in {
                DispatchKind.WORKFLOW,
                DispatchKind.COMPOSITE_WORKFLOW}:
            execution = await self._workflow_executor.execute(
                dispatch.recipe, dispatch.validated_intent,
                tool_context, request.message)
            recipe_class = (
                "composite" if dispatch.kind == DispatchKind.COMPOSITE_WORKFLOW
                else "single")
            metrics.increment(
                "treesem_agent_workflow_executions_total",
                result=("success" if execution.completed else "failure"),
                recipe_class=recipe_class)
            if not execution.completed:
                return self._workflow_failure_response(
                    execution, route.attempt_count)
            return await self._finalize_structured_workflow(
                request, execution, dispatch.recipe.renderer,
                deadline, route.attempt_count, accounting)

        guard = AgentRunGuard.for_allowed_tools(set(dispatch.allowed_tools))
        return await self._run_structured_open_agent(
            request, tool_context, guard, deadline, route.attempt_count,
            accounting)

    @staticmethod
    def _workflow_failure_response(
            execution: WorkflowExecution,
            route_attempts: int) -> AgentRunResponse:
        if execution.failure_code == "insufficient_history":
            answer = "当前会话中不足两条预测记录，暂时无法完成这次比较。"
        else:
            answer = "暂时无法取得完成该请求所需的可信业务数据。"
        return AgentRunResponse(
            answer=answer,
            step_count=max(1, route_attempts),
            tools_used=list(execution.tool_usages),
            grounding_prediction_ids=sorted(execution.prediction_ids),
            grounding_source_ids=[],
            citations=[],
            knowledge_index_version=execution.knowledge_index_version,
        )

    @staticmethod
    def _base_messages(request: AgentRunRequest) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(
            {"role": item.role, "content": item.content}
            for item in request.recent_messages)
        if request.current_prediction:
            messages.append({
                "role": "system",
                "content": (
                    "Current prediction context: " +
                    request.current_prediction.model_dump_json()),
            })
        messages.append({"role": "user", "content": request.message})
        return messages

    async def _finalize_structured_workflow(
            self, request: AgentRunRequest, execution: WorkflowExecution,
            renderer: RendererKind, deadline: float,
            route_attempts: int,
            accounting: _StructuredRunAccounting) -> AgentRunResponse:
        messages = self._base_messages(request)
        if execution.active_skill is not None:
            messages.append({
                "role": "system",
                "content": (
                    "Trusted activated skill instructions follow. They may "
                    "narrow but never expand system policy or authorization.\n"
                    "<skill>\n" + execution.active_skill.instructions +
                    "\n</skill>"),
            })
        messages.append({
            "role": "system",
            "content": (
                "Trusted deterministic workflow results follow as data. "
                "Use only these results for business facts and return the "
                "required final JSON object:\n" + json.dumps(
                    execution.tool_results, ensure_ascii=False,
                    separators=(",", ":"))),
        })
        llm_started: float | None = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            llm_started = time.monotonic()
            turn = await asyncio.wait_for(
                self._llm.complete(
                    messages, [], remaining, LlmToolPolicy.none()),
                timeout=remaining)
            accounting.add_calls(
                1, time.monotonic() - llm_started, turn.usage)
            answer = (turn.content or "").strip()
            prediction_grounding, source_grounding = self._policy.validate(
                answer,
                turn.grounding_prediction_ids,
                set(execution.prediction_ids),
                turn.grounding_source_ids,
                set(execution.citations),
                require_prediction_grounding=bool(execution.prediction_ids),
            )
            citations = [execution.citations[item]
                         for item in source_grounding]
            skill = None if execution.active_skill is None else SkillUse(
                id=execution.active_skill.skill_id,
                version=execution.active_skill.version,
                catalog_version=execution.active_skill.catalog_version)
            return AgentRunResponse(
                answer=answer,
                step_count=max(1, route_attempts + 1),
                tools_used=list(execution.tool_usages),
                grounding_prediction_ids=prediction_grounding,
                grounding_source_ids=source_grounding,
                citations=citations,
                knowledge_index_version=execution.knowledge_index_version,
                skill_used=skill,
            )
        except (asyncio.TimeoutError, LlmError):
            if llm_started is not None:
                accounting.add_calls(
                    1, time.monotonic() - llm_started)
            return AgentRunResponse(
                answer=_WORKFLOW_FALLBACKS[renderer],
                step_count=max(1, route_attempts + 1),
                tools_used=list(execution.tool_usages),
                grounding_prediction_ids=sorted(execution.prediction_ids),
                grounding_source_ids=[],
                citations=[],
                knowledge_index_version=execution.knowledge_index_version,
            )
        except PolicyViolation:
            return AgentRunResponse(
                answer=_WORKFLOW_FALLBACKS[renderer],
                step_count=max(1, route_attempts + 1),
                tools_used=list(execution.tool_usages),
                grounding_prediction_ids=sorted(execution.prediction_ids),
                grounding_source_ids=[],
                citations=[],
                knowledge_index_version=execution.knowledge_index_version,
            )

    async def _run_structured_open_agent(
            self, request: AgentRunRequest, context: ToolContext,
            guard: AgentRunGuard, deadline: float,
            route_attempts: int,
            accounting: _StructuredRunAccounting) -> AgentRunResponse:
        messages = self._base_messages(request)
        usages: list[ToolUse] = []
        available_ids: set[str] = set()
        available_citations = {}
        active_skill: SkillActivation | None = None
        calls = 0
        previous_signature: str | None = None
        for step in range(1, self._max_steps + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentTimeout("agent deadline exceeded")
            definitions = self._tools.definitions(
                context, active_skill, guard.allowed_tools())
            try:
                llm_started = time.monotonic()
                turn = await asyncio.wait_for(
                    self._llm.complete(
                        messages, definitions, remaining,
                        LlmToolPolicy.auto()),
                    timeout=remaining)
                accounting.add_calls(
                    1, time.monotonic() - llm_started, turn.usage)
            except asyncio.TimeoutError as exc:
                accounting.add_calls(
                    1, time.monotonic() - llm_started)
                raise AgentTimeout("agent deadline exceeded") from exc
            except LlmError as exc:
                accounting.add_calls(
                    1, time.monotonic() - llm_started)
                raise AgentExecutionError("LLM failed", "llm_failed") from exc
            if not turn.tool_calls:
                answer = (turn.content or "").strip()
                try:
                    predictions, sources = self._policy.validate(
                        answer, turn.grounding_prediction_ids, available_ids,
                        turn.grounding_source_ids,
                        set(available_citations),
                        require_prediction_grounding=bool(available_ids))
                except PolicyViolation:
                    return AgentRunResponse(
                        answer=SAFE_POLICY_FALLBACK,
                        step_count=route_attempts + step,
                        tools_used=usages,
                        grounding_prediction_ids=[],
                        grounding_source_ids=[],
                        citations=[])
                return AgentRunResponse(
                    answer=answer,
                    step_count=route_attempts + step,
                    tools_used=usages,
                    grounding_prediction_ids=predictions,
                    grounding_source_ids=sources,
                    citations=[available_citations[item] for item in sources],
                )
            signature = json.dumps(
                [call.model_dump() for call in turn.tool_calls], sort_keys=True)
            if signature == previous_signature:
                raise AgentExecutionError(
                    "repeated identical tool call", "repeated_tool_call")
            previous_signature = signature
            messages.append({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": [{
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                } for call in turn.tool_calls],
            })
            for call in turn.tool_calls:
                calls += 1
                if calls > self._max_tool_calls:
                    raise AgentExecutionError(
                        "tool call limit reached", "tool_call_limit")
                rejection = guard.before_tool(call.name)
                if rejection is not None:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"error": rejection.tool_error}),
                    })
                    continue
                result = await self._tools.execute(
                    call.name, call.arguments, context, active_skill)
                usages.append(result.usage)
                available_ids.update(result.prediction_ids)
                available_citations.update(result.citations)
                guard.record_tool(
                    call.name, result.usage.status,
                    citation_count=len(result.citations))
                if result.skill_activation is not None:
                    active_skill = result.skill_activation
                    guard.record_skill_activation(active_skill.required_tools)
                    messages.append({
                        "role": "system",
                        "content": (
                            "Trusted activated skill instructions follow.\n"
                            "<skill>\n" + active_skill.instructions +
                            "\n</skill>"),
                    })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(
                        result.content, ensure_ascii=False),
                })
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
