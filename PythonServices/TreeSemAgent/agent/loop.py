from __future__ import annotations

import asyncio
import json
import time

from .llm_client import LlmClient, LlmError
from .policy import PolicyViolation, ResponsePolicy
from .prompt import SYSTEM_PROMPT
from .schemas import AgentRunRequest, AgentRunResponse
from .tool_registry import ToolRegistry
from .tools import ToolContext


class AgentExecutionError(RuntimeError):
    pass


class AgentTimeout(AgentExecutionError):
    pass


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
        deadline = time.monotonic() + self._total_timeout
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in request.recent_messages)
        if request.current_prediction:
            messages.append({"role": "system", "content": "Current prediction context: " + request.current_prediction.model_dump_json()})
        messages.append({"role": "user", "content": request.message})
        context = ToolContext(request.session_id, request.capability_token)
        usages = []
        available_ids: set[str] = set()
        calls = 0
        previous_signature: str | None = None
        repeated = 0
        for step in range(1, self._max_steps + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AgentTimeout("agent deadline exceeded")
            try:
                turn = await asyncio.wait_for(
                    self._llm.complete(messages, self._tools.definitions(), remaining), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise AgentTimeout("agent deadline exceeded") from exc
            except LlmError as exc:
                raise AgentExecutionError("LLM failed") from exc
            if not turn.tool_calls:
                answer = (turn.content or "").strip()
                try:
                    self._policy.validate(answer, turn.grounding_prediction_ids, available_ids)
                except PolicyViolation as exc:
                    raise AgentExecutionError("final response failed grounding policy") from exc
                return AgentRunResponse(answer=answer, step_count=step, tools_used=usages,
                                        grounding_prediction_ids=turn.grounding_prediction_ids)
            signature = json.dumps([call.model_dump() for call in turn.tool_calls], sort_keys=True)
            repeated = repeated + 1 if signature == previous_signature else 0
            previous_signature = signature
            if repeated >= 1:
                raise AgentExecutionError("repeated identical tool call")
            messages.append({"role": "assistant", "content": turn.content, "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}
                for call in turn.tool_calls
            ]})
            for call in turn.tool_calls:
                calls += 1
                if calls > self._max_tool_calls:
                    raise AgentExecutionError("tool call limit reached")
                result = await self._tools.execute(call.name, call.arguments, context)
                usages.append(result.usage)
                available_ids.update(result.prediction_ids)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result.content, ensure_ascii=False)})
        raise AgentExecutionError("step limit reached")
