from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

from ..observability import TraceState, metrics, trace_event


class KnowledgeToolError(RuntimeError):
    pass


class KnowledgeClient(Protocol):
    async def search(self, token: str, query: str, scope: str,
                     top_k: int, trace: TraceState | None = None) -> dict[str, Any]: ...

    async def ready(self) -> bool: ...

    async def close(self) -> None: ...


class McpKnowledgeClient:
    def __init__(self, url: str, timeout_seconds: float = 3.0,
                 maximum_response_bytes: int = 65_536):
        self._url = url
        self._timeout = timeout_seconds
        self._maximum = maximum_response_bytes

    async def _call(self, token: str, query: str, scope: str,
                    top_k: int, trace: TraceState | None = None) -> dict[str, Any]:
        try:
            import httpx
            from mcp import ClientSession  # type: ignore[import-not-found]
            from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise KnowledgeToolError("official MCP SDK is unavailable") from exc
        headers = {"Authorization": f"Bearer {token}"}
        if trace:
            headers.update({"X-Request-Id": trace.request_id,
                            "traceparent": trace.traceparent})
        async with httpx.AsyncClient(headers=headers, timeout=self._timeout) as client:
            async with streamable_http_client(self._url, http_client=client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await session.call_tool("search_medical_knowledge", {
                        "query": query, "scope": scope, "top_k": top_k})
                    if result.isError:
                        raise KnowledgeToolError("knowledge MCP rejected the request")
                    structured = getattr(result, "structuredContent", None)
                    if structured is not None:
                        body = structured
                    elif result.content and hasattr(result.content[0], "text"):
                        body = json.loads(result.content[0].text)
                    else:
                        raise KnowledgeToolError("knowledge MCP returned no structured result")
                    encoded = json.dumps(body, ensure_ascii=False).encode()
                    if len(encoded) > self._maximum or not isinstance(body, dict):
                        raise KnowledgeToolError("knowledge response exceeded its contract")
                    return body

    async def search(self, token: str, query: str, scope: str,
                     top_k: int, trace: TraceState | None = None) -> dict[str, Any]:
        error: Exception | None = None
        for attempt in range(2):
            child = trace.child() if trace else None
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    self._call(token, query, scope, top_k, child), self._timeout)
                mode = str(result.get("retrieval_mode", "unknown"))
                metrics.increment("treesem_agent_knowledge_calls_total", result="success", mode=mode)
                metrics.observe("treesem_agent_knowledge_duration_seconds",
                                time.monotonic() - started, mode=mode)
                if child:
                    trace_event(child, "agent.mcp.search_medical_knowledge",
                                started, "success", retrieval_mode=mode)
                return result
            except (OSError, asyncio.TimeoutError, KnowledgeToolError) as exc:
                error = exc
                if child:
                    trace_event(child, "agent.mcp.search_medical_knowledge",
                                started, "error", error_code="knowledge_unavailable")
                if attempt == 0:
                    await asyncio.sleep(0.03)
        metrics.increment("treesem_agent_knowledge_calls_total",
                          result="error", mode="unavailable")
        raise KnowledgeToolError("knowledge MCP is unavailable") from error

    async def ready(self) -> bool:
        try:
            import httpx
            from mcp import ClientSession  # type: ignore[import-not-found]
            from mcp.client.streamable_http import streamable_http_client  # type: ignore[import-not-found]
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with streamable_http_client(self._url, http_client=client) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        return any(tool.name == "search_medical_knowledge" for tool in tools.tools)
        except Exception:
            return False

    async def close(self) -> None:
        return None
