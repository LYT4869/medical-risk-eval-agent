import asyncio
import threading
import time
import unittest

from agent.routing_executor import (
    RoutingExecutorClosed,
    RoutingOverloaded,
    RoutingTimeout,
    SemanticRoutingExecutor,
)


class SemanticRoutingExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        executor = getattr(self, "executor", None)
        release = getattr(self, "release", None)
        if release is not None:
            release.set()
        if executor is not None:
            await executor.close()

    async def test_blocking_encoder_does_not_block_event_loop(self):
        self.release = threading.Event()
        self.executor = SemanticRoutingExecutor(
            workers=1, queue_capacity=0,
            admission_timeout_seconds=0.01,
            route_timeout_seconds=0.2)
        task = asyncio.create_task(
            self.executor.run(self.release.wait, 0.15))
        started = time.monotonic()
        await asyncio.sleep(0.01)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.08)
        self.release.set()
        await task

    async def test_admission_is_bounded(self):
        self.release = threading.Event()
        self.executor = SemanticRoutingExecutor(
            workers=1, queue_capacity=1,
            admission_timeout_seconds=0.01,
            route_timeout_seconds=0.5)
        first = asyncio.create_task(self.executor.run(self.release.wait, 0.4))
        second = asyncio.create_task(self.executor.run(self.release.wait, 0.4))
        await asyncio.sleep(0.02)
        self.assertEqual(self.executor.inflight, 2)
        with self.assertRaises(RoutingOverloaded):
            await self.executor.run(lambda: None)
        self.release.set()
        await asyncio.gather(first, second)

    async def test_timeout_keeps_permit_until_worker_finishes(self):
        self.release = threading.Event()
        self.executor = SemanticRoutingExecutor(
            workers=1, queue_capacity=0,
            admission_timeout_seconds=0.01,
            route_timeout_seconds=0.01)
        with self.assertRaises(RoutingTimeout):
            await self.executor.run(self.release.wait, 0.3)
        self.assertEqual(self.executor.inflight, 1)
        with self.assertRaises(RoutingOverloaded):
            await self.executor.run(lambda: "hidden overcommit")
        self.release.set()
        await self.executor.wait_until_idle()
        self.assertEqual(await self.executor.run(lambda: "ok"), "ok")

    async def test_shutdown_rejects_new_work(self):
        self.executor = SemanticRoutingExecutor(
            workers=1, queue_capacity=0,
            admission_timeout_seconds=0.01,
            route_timeout_seconds=0.1)
        await self.executor.close()
        with self.assertRaises(RoutingExecutorClosed):
            await self.executor.run(lambda: None)

    async def test_worker_exception_is_propagated(self):
        self.executor = SemanticRoutingExecutor(
            workers=1, queue_capacity=0,
            admission_timeout_seconds=0.01,
            route_timeout_seconds=0.1)

        def fail():
            raise ValueError("bad embedding")

        with self.assertRaisesRegex(ValueError, "bad embedding"):
            await self.executor.run(fail)
        await self.executor.wait_until_idle()
        self.assertEqual(self.executor.inflight, 0)


if __name__ == "__main__":
    unittest.main()
