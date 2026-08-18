from __future__ import annotations

import os
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException

from agent.llm_client import OpenAiCompatibleClient, OpenAiCompatibleConfig
from agent.loop import AgentExecutionError, AgentLoop, AgentTimeout
from agent.schemas import AgentRunRequest, AgentRunResponse
from agent.skills import SkillCatalog
from agent.tool_registry import ToolRegistry
from agent.tools import BackendToolClient, McpKnowledgeClient


def _integer(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name, "true" if default else "false").lower()
    if value not in {"true", "false"}:
        raise RuntimeError(f"{name} must be true or false")
    return value == "true"


def create_app(loop: AgentLoop | None = None) -> FastAPI:
    backend_url = os.getenv("TREESEM_AGENT_BACKEND_URL", "http://127.0.0.1:8080")
    service_secret = os.getenv("TREESEM_AGENT_SERVICE_SECRET", "")
    owned = loop is None
    llm = None
    backend = None
    knowledge = None
    skills = None
    if loop is None:
        if len(service_secret) < 32:
            raise RuntimeError("TREESEM_AGENT_SERVICE_SECRET must contain at least 32 characters")
        base_url = os.getenv("TREESEM_AGENT_LLM_BASE_URL", "").strip()
        model = os.getenv("TREESEM_AGENT_LLM_MODEL", "").strip()
        if not base_url or not model:
            raise RuntimeError("TREESEM_AGENT_LLM_BASE_URL and TREESEM_AGENT_LLM_MODEL are required")
        llm = OpenAiCompatibleClient(OpenAiCompatibleConfig(
            base_url=base_url, model=model,
            api_key=os.getenv("TREESEM_AGENT_LLM_API_KEY", ""),
            connect_timeout_seconds=_integer("TREESEM_AGENT_LLM_CONNECT_TIMEOUT_MS", 1000) / 1000,
            request_timeout_seconds=_integer("TREESEM_AGENT_LLM_REQUEST_TIMEOUT_MS", 20000) / 1000,
        ))
        backend = BackendToolClient(backend_url)
        knowledge_enabled = _boolean("TREESEM_KNOWLEDGE_ENABLED", True)
        if knowledge_enabled:
            knowledge = McpKnowledgeClient(
                os.getenv("TREESEM_KNOWLEDGE_MCP_URL", "http://127.0.0.1:8092/mcp"),
                _integer("TREESEM_KNOWLEDGE_REQUEST_TIMEOUT_MS", 3000) / 1000,
                _integer("TREESEM_KNOWLEDGE_MAX_RESPONSE_BYTES", 65536))
        skill_enabled = _boolean("TREESEM_AGENT_SKILLS_ENABLED", True)
        available_tools = ToolRegistry.native_tool_names()
        if knowledge is not None:
            available_tools.add("search_medical_knowledge")
        if skill_enabled:
            default_skills = Path(__file__).resolve().parent / "skills"
            skills = SkillCatalog(Path(os.getenv(
                "TREESEM_AGENT_SKILLS_DIR", str(default_skills))), available_tools)
        loop = AgentLoop(llm, ToolRegistry(backend, knowledge, skills),
                         _integer("TREESEM_AGENT_MAX_STEPS", 5),
                         _integer("TREESEM_AGENT_MAX_TOOL_CALLS", 8),
                         _integer("TREESEM_AGENT_TOTAL_TIMEOUT_MS", 25000) / 1000)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owned:
            if llm is not None:
                await llm.close()
            if backend is not None:
                await backend.close()
            if knowledge is not None:
                await knowledge.close()

    app = FastAPI(title="treeSem Agent", version="1.0", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        result = {"status": "ok", "service": "treeSem-agent"}
        if skills is not None:
            result.update({"skill_count": skills.count,
                           "skill_catalog_version": skills.version})
        return result

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.get(backend_url.rstrip("/") + "/health")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503, detail="tool backend unavailable") from exc
        if knowledge is not None and not await knowledge.ready():
            raise HTTPException(status_code=503, detail="knowledge MCP unavailable")
        return {"status": "ready"}

    @app.post("/v1/agent/runs", response_model=AgentRunResponse)
    async def run_agent(request: AgentRunRequest,
                        x_treesem_agent_token: str = Header(default="")) -> AgentRunResponse:
        if not service_secret or not hmac.compare_digest(
                x_treesem_agent_token.encode(), service_secret.encode()):
            raise HTTPException(status_code=401, detail="invalid service credential")
        try:
            return await loop.run(request)
        except AgentTimeout as exc:
            raise HTTPException(status_code=504, detail="agent timeout") from exc
        except AgentExecutionError as exc:
            raise HTTPException(status_code=502, detail="agent execution failed") from exc

    return app


app = create_app()
