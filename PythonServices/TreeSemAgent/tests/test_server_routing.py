import asyncio
import importlib
import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.routing_config import (
    RoutingEmbeddingBackend,
    RoutingMode,
    RoutingSettings,
    build_routing_runtime,
)
from agent.routing_artifact import RoutingArtifactError
from agent.routing import RuleOnlyRouter
from agent.task_registry import load_default_registry


ROOT = Path(__file__).resolve().parents[1]
HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
SERVER_ENVIRONMENT = {
    "TREESEM_AGENT_SERVICE_SECRET": "s" * 32,
    "TREESEM_AGENT_LLM_MODE": "scripted_demo",
    "TREESEM_DEPLOYMENT_ENV": "local",
    "TREESEM_KNOWLEDGE_ENABLED": "false",
    "TREESEM_AGENT_SKILLS_ENABLED": "false",
    "TREESEM_AGENT_ROUTING_MODE": "rule",
}


class FakeRoutingRuntime:
    def __init__(self):
        self.mode = RoutingMode.HYBRID_OPTIONAL
        self.router = RuleOnlyRouter()
        self.registry = load_default_registry()
        self.semantic_router = object()
        self.degradation_reason = None
        self.close_calls = 0

    @property
    def semantic_available(self):
        return self.semantic_router is not None

    async def close(self):
        self.close_calls += 1
        self.semantic_router = None


class StubLoop:
    async def run(self, request):
        raise AssertionError("run should not be called by health checks")


class HealthyResponse:
    def raise_for_status(self):
        return None


class HealthyAsyncClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def get(self, url):
        del url
        return HealthyResponse()


class ServerRoutingTest(unittest.TestCase):
    def settings(self, mode: RoutingMode, *,
                 artifact_dir: Path | None = None) -> RoutingSettings:
        return RoutingSettings(
            mode=mode,
            task_registry=ROOT / "config" / "tasks.yaml",
            thresholds=ROOT / "config" / "routing_thresholds.json",
            model="intfloat/multilingual-e5-small",
            revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_backend=RoutingEmbeddingBackend.ONNX_FP32,
            artifact_dir=artifact_dir or ROOT / "missing-routing-artifact",
            workers=1,
            queue_capacity=8,
            admission_timeout_seconds=0.005,
            route_timeout_seconds=0.150,
        )

    def test_rule_mode_never_constructs_embedding_provider(self):
        calls = []
        runtime = build_routing_runtime(
            self.settings(RoutingMode.RULE),
            provider_factory=lambda *args: calls.append(args))
        self.assertEqual(calls, [])
        self.assertFalse(runtime.semantic_available)
        self.assertIsNone(runtime.degradation_reason)

    def test_optional_mode_degrades_when_provider_is_unavailable(self):
        def fail(*args):
            raise RoutingArtifactError("model unavailable")

        runtime = build_routing_runtime(
            self.settings(RoutingMode.HYBRID_OPTIONAL),
            provider_factory=fail)
        self.assertFalse(runtime.semantic_available)
        self.assertEqual(runtime.degradation_reason,
                         "semantic_initialization_failed")

    def test_required_mode_fails_when_provider_is_unavailable(self):
        def fail(*args):
            raise RoutingArtifactError("model unavailable")

        with self.assertRaisesRegex(RoutingArtifactError, "model unavailable"):
            build_routing_runtime(
                self.settings(RoutingMode.HYBRID_REQUIRED),
                provider_factory=fail)

    def test_provider_factory_receives_settings_and_registry(self):
        calls = []

        class Provider:
            def encode_examples(self, texts):
                return [[1.0, 0.0] for _ in texts]

            def encode_query(self, text):
                del text
                return [1.0, 0.0]

        settings = self.settings(RoutingMode.HYBRID_REQUIRED)
        runtime = build_routing_runtime(
            settings,
            provider_factory=lambda received_settings, registry: (
                calls.append((received_settings, registry)) or Provider()))

        self.assertTrue(runtime.semantic_available)
        self.assertEqual(calls[0][0], settings)
        self.assertEqual(len(calls[0][1].business_definitions), 6)

    def test_environment_validation_rejects_invalid_capacity(self):
        with self.assertRaisesRegex(RuntimeError, "QUEUE_CAPACITY"):
            RoutingSettings.from_environment({
                "TREESEM_AGENT_ROUTING_MODE": "rule",
                "TREESEM_AGENT_ROUTING_QUEUE_CAPACITY": "-1",
            }, root=ROOT)

    def test_environment_parses_onnx_backend_and_artifact(self):
        settings = RoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "hybrid_optional",
            "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND": "onnx_fp32",
            "TREESEM_AGENT_ROUTING_ARTIFACT_DIR": "/routing/artifact",
        }, root=ROOT)

        self.assertEqual(settings.embedding_backend,
                         RoutingEmbeddingBackend.ONNX_FP32)
        self.assertEqual(settings.artifact_dir, Path("/routing/artifact"))

    def test_default_candidate_backend_is_validated_fp32(self):
        settings = RoutingSettings.from_environment({
            "TREESEM_AGENT_ROUTING_MODE": "rule",
        }, root=ROOT)

        self.assertEqual(settings.embedding_backend,
                         RoutingEmbeddingBackend.ONNX_FP32)

    def test_semantic_environment_requires_artifact_directory(self):
        with self.assertRaisesRegex(RuntimeError, "ARTIFACT_DIR"):
            RoutingSettings.from_environment({
                "TREESEM_AGENT_ROUTING_MODE": "hybrid_optional",
            }, root=ROOT)

    def test_environment_rejects_unknown_embedding_backend(self):
        with self.assertRaisesRegex(RuntimeError, "EMBEDDING_BACKEND"):
            RoutingSettings.from_environment({
                "TREESEM_AGENT_ROUTING_MODE": "rule",
                "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND": "pytorch",
            }, root=ROOT)

    def test_runtime_close_is_safe_in_rule_mode(self):
        runtime = build_routing_runtime(self.settings(RoutingMode.RULE))
        asyncio.run(runtime.close())
        asyncio.run(runtime.close())

    @staticmethod
    def _load_server():
        with patch.dict(os.environ, SERVER_ENVIRONMENT, clear=False):
            return importlib.import_module("server")

    @unittest.skipUnless(HAS_FASTAPI, "FastAPI service dependency is absent")
    def test_owned_app_closes_routing_runtime_on_shutdown(self):
        from fastapi.testclient import TestClient

        server = self._load_server()
        runtime = FakeRoutingRuntime()
        with patch.dict(os.environ, SERVER_ENVIRONMENT, clear=False), \
                patch.object(server, "build_routing_runtime",
                             return_value=runtime):
            app = server.create_app()
            with TestClient(app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["routing_mode"],
                                 "hybrid_optional")
                self.assertTrue(
                    health.json()["semantic_routing_available"])
            self.assertEqual(runtime.close_calls, 1)

    @unittest.skipUnless(HAS_FASTAPI, "FastAPI service dependency is absent")
    def test_injected_loop_never_builds_routing_runtime(self):
        from fastapi.testclient import TestClient

        server = self._load_server()
        with patch.object(
                server, "build_routing_runtime",
                side_effect=AssertionError("routing runtime must not be built")):
            app = server.create_app(loop=StubLoop())
        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routing_mode"], "injected")
        self.assertFalse(response.json()["semantic_routing_available"])

    @unittest.skipUnless(HAS_FASTAPI, "FastAPI service dependency is absent")
    def test_ready_reports_routing_state_without_embedding_request(self):
        from fastapi.testclient import TestClient

        server = self._load_server()
        runtime = FakeRoutingRuntime()
        with patch.dict(os.environ, SERVER_ENVIRONMENT, clear=False), \
                patch.object(server, "build_routing_runtime",
                             return_value=runtime):
            app = server.create_app()
        with patch.object(server.httpx, "AsyncClient",
                          return_value=HealthyAsyncClient()), \
                TestClient(app) as client:
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["routing_mode"],
                         "hybrid_optional")
        self.assertTrue(response.json()["semantic_routing_available"])


if __name__ == "__main__":
    unittest.main()
