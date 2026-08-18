import os
from typing import Annotated, Literal

from pydantic import Field

from .retrieval.index import (CrossEncoderReranker, HybridRetriever, KnowledgeIndex,
                              SentenceTransformerQueryEmbedding)
from .service import KnowledgeOverloaded, KnowledgeService


def _positive(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def create_service() -> KnowledgeService:
    directory = os.getenv("TREESEM_KNOWLEDGE_INDEX_DIR", "").strip()
    secret = os.getenv("TREESEM_KNOWLEDGE_JWT_SECRET", "")
    if not directory or len(secret) < 32:
        raise RuntimeError("knowledge index and JWT secret are required")
    # Serving must be deterministic and must never update model files over the
    # network. The offline index build is the only place allowed to download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    index = KnowledgeIndex(__import__("pathlib").Path(directory))
    embedding_config = index.manifest["embedding"]
    embedding = SentenceTransformerQueryEmbedding(
        embedding_config["model"], embedding_config["revision"])
    manifest_reranker = index.manifest.get("reranker", {})
    reranker_model = os.getenv(
        "TREESEM_KNOWLEDGE_RERANK_MODEL", manifest_reranker.get("model", ""))
    reranker_revision = os.getenv(
        "TREESEM_KNOWLEDGE_RERANK_REVISION",
        manifest_reranker.get("revision", "")).strip()
    if (not reranker_model or not reranker_revision or
            reranker_model != manifest_reranker.get("model") or
            reranker_revision != manifest_reranker.get("revision")):
        raise RuntimeError("reranker model and revision do not match index manifest")
    reranker = CrossEncoderReranker(reranker_model, reranker_revision)
    thresholds = index.manifest.get("thresholds", {})
    if set(thresholds) != {"reranker_min_score",
                           "all_scope_reranker_min_score", "rrf_min_score"}:
        raise RuntimeError("calibrated retrieval thresholds are missing")
    return KnowledgeService(HybridRetriever(
        index, embedding, reranker,
        reranker_min_score=float(thresholds["reranker_min_score"]),
        all_scope_reranker_min_score=float(
            thresholds["all_scope_reranker_min_score"]),
        rrf_min_score=float(thresholds["rrf_min_score"])), secret,
                            _positive("TREESEM_KNOWLEDGE_WORKERS", 1),
                            _positive("TREESEM_KNOWLEDGE_QUEUE_CAPACITY", 16))


def create_mcp(service: KnowledgeService | None = None):
    try:
        from mcp.server.fastmcp import Context, FastMCP  # type: ignore[import-not-found]
        from starlette.requests import Request  # type: ignore[import-not-found]
        from starlette.responses import JSONResponse  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("the official MCP Python SDK is required") from exc
    owned = service is None
    service = service or create_service()
    mcp = FastMCP(
        "treeSem Medical Knowledge", stateless_http=True, json_response=True,
        host=os.getenv("TREESEM_KNOWLEDGE_HOST", "127.0.0.1"),
        port=_positive("TREESEM_KNOWLEDGE_PORT", 8092),
        max_request_body_size=65_536)

    def bearer(ctx: Context) -> str:
        header = ctx.request_context.request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise ValueError("knowledge authentication required")
        return header[7:]

    @mcp.tool(name="search_medical_knowledge")
    async def search_medical_knowledge(
            query: Annotated[str, Field(min_length=1, max_length=500)],
            ctx: Context,
            scope: Literal["model", "clinical", "all"] = "all",
            top_k: Annotated[int, Field(ge=1, le=6)] = 5) -> dict:
        """Search curated treeSem and authoritative PPH knowledge with citations."""
        try:
            return await service.search(bearer(ctx), query, scope, top_k)
        except KnowledgeOverloaded as exc:
            raise RuntimeError("knowledge_overloaded") from exc

    # The pinned SDK builds a Pydantic argument model with extra="ignore".
    # Tighten that generated model so MCP rejects fields outside our contract.
    registered = mcp._tool_manager.get_tool("search_medical_knowledge")
    if registered is None:
        raise RuntimeError("knowledge MCP tool registration failed")
    registered.fn_metadata.arg_model.model_config["extra"] = "forbid"
    registered.fn_metadata.arg_model.model_rebuild(force=True)
    registered.parameters = registered.fn_metadata.arg_model.model_json_schema()
    if registered.parameters.get("additionalProperties") is not False:
        raise RuntimeError("official MCP SDK did not preserve strict tool arguments")

    @mcp.resource("treesem://knowledge/index-manifest")
    def index_manifest() -> dict:
        manifest = service.retriever.index.manifest
        return {"index_version": manifest["index_version"],
                "chunk_count": manifest["chunk_count"],
                "source_count": len(manifest["sources"]),
                "schema_version": manifest["schema_version"]}

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request):
        return JSONResponse({"status": "ok", "service": "treeSem-knowledge"})

    @mcp.custom_route("/ready", methods=["GET"])
    async def ready(_: Request):
        return JSONResponse({"status": "ready",
                             "index_version": service.retriever.index.version})

    if owned:
        original_close = service.close
        mcp._treesem_close = original_close  # type: ignore[attr-defined]
    return mcp


def main() -> None:
    create_mcp().run(transport="streamable-http")


if __name__ == "__main__":
    main()
