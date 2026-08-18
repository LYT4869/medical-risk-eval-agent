from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from .auth import verify_knowledge_token
from .retrieval import HybridRetriever
from .observability import metrics


class KnowledgeOverloaded(RuntimeError):
    pass


class KnowledgeService:
    def __init__(self, retriever: HybridRetriever, secret: str,
                 workers: int = 1, queue_capacity: int = 16):
        if workers <= 0 or queue_capacity <= 0:
            raise ValueError("knowledge execution limits must be positive")
        self.retriever = retriever
        self._secret = secret
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="treesem-knowledge")
        self._slots = asyncio.Semaphore(workers + queue_capacity)

    async def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def search(self, token: str, query: str, scope: str = "all",
                     top_k: int = 5) -> dict:
        started = time.monotonic()
        claims = verify_knowledge_token(token, self._secret)
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout=0.05)
        except asyncio.TimeoutError as exc:
            metrics.increment("treesem_knowledge_requests_total", result="overloaded")
            raise KnowledgeOverloaded("knowledge queue is full") from exc
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: self.retriever.search(
                    query, claims.actor_role, set(claims.scopes), scope, top_k))
            body = result.json()
            mode = str(body.get("retrieval_mode", "unknown"))
            metrics.increment("treesem_knowledge_requests_total",
                              result="success", mode=mode)
            metrics.observe("treesem_knowledge_duration_seconds",
                            time.monotonic() - started, mode=mode)
            return body
        except Exception:
            metrics.increment("treesem_knowledge_requests_total", result="error")
            raise
        finally:
            self._slots.release()
