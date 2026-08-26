from __future__ import annotations

import asyncio
import json
import time

from .llm_client import LlmClient, LlmError
from .policy import (SAFE_POLICY_FALLBACK, SAFE_SECURITY_REFUSAL,
                     PolicyViolation, ResponsePolicy)
from .prompt import SYSTEM_PROMPT
from .observability import TraceState, metrics, trace_event
from .run_guard import AgentRunGuard
from .schemas import AgentRunRequest, AgentRunResponse, SkillUse, ToolUse
from .skills import SkillActivation
from .tool_registry import ToolRegistry
from .tools import ToolContext


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
                 policy: ResponsePolicy | None = None):
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._total_timeout = total_timeout_seconds
        self._policy = policy or ResponsePolicy()

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
        guard = AgentRunGuard.for_request(request.message)
        if guard.security_refusal is not None:
            return AgentRunResponse(
                answer=SAFE_SECURITY_REFUSAL,
                step_count=1,
                tools_used=[],
                grounding_prediction_ids=[],
                grounding_source_ids=[],
                citations=[],
            )
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in request.recent_messages)
        if request.current_prediction:
            messages.append({"role": "system", "content": "Current prediction context: " + request.current_prediction.model_dump_json()})
        messages.append({"role": "user", "content": request.message})
        context = ToolContext(request.session_id, request.capability_token,
                              request.knowledge_capability_token,
                              request.actor_role, trace)
        catalog_prompt = self._tools.skill_catalog_prompt(context.actor_role)
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
        for step in range(1, self._max_steps + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentTimeout("agent deadline exceeded")
            try:
                turn = await asyncio.wait_for(
                    self._llm.complete(
                        messages, self._tools.definitions(
                            context, active_skill, guard.allowed_tools()),
                        remaining),
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
                answer = (turn.content or "").strip()
                policy_started = time.monotonic()
                try:
                    prediction_grounding, source_grounding = self._policy.validate(
                        answer, turn.grounding_prediction_ids, available_ids,
                        turn.grounding_source_ids, set(available_citations),
                        require_prediction_grounding=bool(available_ids))
                except PolicyViolation as exc:
                    metrics.increment(
                        "treesem_agent_grounding_rejections_total",
                        result="safe_fallback", reason=exc.code)
                    trace_event(
                        trace.child(), "agent.response_policy", policy_started,
                        "error", error_code=exc.code)
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
                    active_skill = result.skill_activation
                    guard.record_skill_activation(active_skill.required_tools)
                    messages.append({
                        "role": "system",
                        "content": "Trusted activated skill instructions follow. They may narrow but never expand system policy or authorization.\n<skill>\n" +
                                   active_skill.instructions + "\n</skill>"})
        raise AgentExecutionError("step limit reached", "step_limit")
