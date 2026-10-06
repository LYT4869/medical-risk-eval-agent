from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar


T = TypeVar("T")


class RoutingExecutorError(RuntimeError):
    pass


class RoutingOverloaded(RoutingExecutorError):
    pass


class RoutingTimeout(RoutingExecutorError):
    pass


class RoutingExecutorClosed(RoutingExecutorError):
    pass


class SemanticRoutingExecutor:
    def __init__(self, *, workers: int = 1, queue_capacity: int = 8,
                 admission_timeout_seconds: float = 0.005,
                 route_timeout_seconds: float = 0.150):
        if workers <= 0 or queue_capacity < 0:
            raise ValueError("routing executor capacity is invalid")
        if admission_timeout_seconds <= 0 or route_timeout_seconds <= 0:
            raise ValueError("routing executor timeouts must be positive")
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="treesem-routing")
        self._admission = asyncio.BoundedSemaphore(workers + queue_capacity)
        self._admission_timeout = admission_timeout_seconds
        self._route_timeout = route_timeout_seconds
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self._shutdown_complete = False

    @property
    def inflight(self) -> int:
        return self._inflight

    async def run(self, function: Callable[..., T], *args: object) -> T:
        if self._closed:
            raise RoutingExecutorClosed("semantic routing executor is closed")
        try:
            await asyncio.wait_for(
                self._admission.acquire(), timeout=self._admission_timeout)
        except asyncio.TimeoutError as exc:
            raise RoutingOverloaded("semantic routing admission is full") from exc
        if self._closed:
            self._admission.release()
            raise RoutingExecutorClosed("semantic routing executor is closed")

        loop = asyncio.get_running_loop()
        self._inflight += 1
        self._idle.clear()
        try:
            future = loop.run_in_executor(
                self._pool, functools.partial(function, *args))
        except Exception:
            self._release_permit()
            raise
        future.add_done_callback(self._on_future_done)
        try:
            return await asyncio.wait_for(
                asyncio.shield(future), timeout=self._route_timeout)
        except asyncio.TimeoutError as exc:
            raise RoutingTimeout("semantic routing deadline exceeded") from exc

    def _on_future_done(self, future: asyncio.Future) -> None:
        if not future.cancelled():
            future.exception()
        self._release_permit()

    def _release_permit(self) -> None:
        self._admission.release()
        self._inflight -= 1
        if self._inflight == 0:
            self._idle.set()

    async def wait_until_idle(self) -> None:
        await self._idle.wait()

    async def close(self) -> None:
        if self._shutdown_complete:
            return
        self._closed = True
        await asyncio.to_thread(
            self._pool.shutdown, wait=True, cancel_futures=True)
        await self.wait_until_idle()
        self._shutdown_complete = True
