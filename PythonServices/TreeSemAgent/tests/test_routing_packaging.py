from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
import subprocess
import shutil
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def load_prepare_demo_module():
    path = ROOT / "scripts" / "prepare_demo.py"
    spec = importlib.util.spec_from_file_location("prepare_demo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RoutingPackagingTest(unittest.TestCase):
    def test_agent_image_keeps_onnx_dependencies_optional(self):
        dockerfile = (ROOT / "deploy/docker/agent.Dockerfile").read_text(
            encoding="utf-8")
        self.assertIn("ARG TREESEM_INSTALL_SEMANTIC_ROUTING=false", dockerfile)
        self.assertIn("requirements-routing.txt", dockerfile)
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING", dockerfile)
        self.assertNotIn("snapshot_download", dockerfile)
        requirements = (
            ROOT / "PythonServices/TreeSemAgent/requirements-routing.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("onnxruntime==1.20.1", requirements)
        self.assertIn("numpy==2.0.1", requirements)
        self.assertIn("tokenizers==0.22.2", requirements)
        self.assertNotIn("torch", requirements)
        self.assertNotIn("sentence-transformers", requirements)
        self.assertNotIn("transformers==", requirements)

    def test_compose_uses_pinned_read_only_routing_artifact(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING", compose)
        self.assertIn("TREESEM_AGENT_ROUTING_MODE", compose)
        self.assertIn("TREESEM_AGENT_ROUTING_MODEL_REVISION", compose)
        self.assertIn("TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND", compose)
        self.assertIn(
            "TREESEM_AGENT_ROUTING_ARTIFACT_DIR: /routing/artifact", compose)
        self.assertIn("/routing/artifact:ro", compose)
        agent_section = compose.split("\n  agent:\n", 1)[1].split(
            "\n  demo-web:\n", 1)[0]
        self.assertNotIn("HF_HUB_OFFLINE", agent_section)
        self.assertNotIn("TRANSFORMERS_OFFLINE", agent_section)
        self.assertNotIn("/app/.cache/huggingface", agent_section)

    def test_environment_defaults_to_rules_and_pins_model_revision(self):
        environment = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("TREESEM_AGENT_ROUTING_MODE=rule", environment)
        self.assertIn(f"TREESEM_AGENT_ROUTING_MODEL={MODEL}", environment)
        self.assertIn(
            f"TREESEM_AGENT_ROUTING_MODEL_REVISION={REVISION}", environment)
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING=false", environment)
        self.assertIn(
            "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND=onnx_int8", environment)

    def test_operator_preparation_and_make_targets_are_reproducible(self):
        script = ROOT / "scripts/prepare-routing-model.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        self.assertIn(MODEL, content)
        self.assertIn(REVISION, content)
        self.assertIn("routing-export.Dockerfile", content)
        self.assertIn("TREESEM_ROUTING_EXPORT_BACKEND", content)
        self.assertIn("TREESEM_ROUTING_OUTPUT_ROOT", content)
        self.assertIn("HF_HUB_OFFLINE=1", content)
        self.assertNotIn("snapshot_download", content)
        prepare_demo = (ROOT / "scripts/prepare_demo.py").read_text(
            encoding="utf-8")
        self.assertIn("valid_routing_artifact", prepare_demo)
        self.assertIn("TREESEM_AGENT_ROUTING_ARTIFACT_DIR", prepare_demo)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target in (
                "routing-unit:", "routing-calibrate:",
                "routing-evaluate:", "routing-load-smoke:",
                "routing-export-fp32:", "routing-export-int8:",
                "routing-parity:", "routing-benchmark:"):
            self.assertIn(target, makefile)

    def test_model_cache_and_generated_weights_are_excluded(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        self.assertIn(".cache", dockerignore)
        self.assertIn("artifacts", dockerignore)
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/artifacts/", gitignore)
        if shutil.which("git") is not None:
            tracked = set(subprocess.run(
                ["git", "ls-files"], cwd=ROOT, check=True,
                text=True, capture_output=True).stdout.splitlines())
            self.assertFalse(any(
                path.endswith((".pt", ".pth", ".onnx", ".safetensors"))
                for path in tracked))

    def test_prepare_demo_fails_fast_for_invalid_hybrid_artifact(self):
        prepare_demo = load_prepare_demo_module()
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "artifact failed"):
                prepare_demo.prepare_routing({
                    "TREESEM_AGENT_ROUTING_MODE": "hybrid_required",
                    "TREESEM_INSTALL_SEMANTIC_ROUTING": "true",
                    "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND": "onnx_fp32",
                    "TREESEM_AGENT_ROUTING_ARTIFACT_DIR": directory,
                })

    def test_prepare_demo_rule_mode_does_not_require_artifact(self):
        prepare_demo = load_prepare_demo_module()
        with mock.patch.dict(os.environ, {}, clear=True):
            mode, directory = prepare_demo.prepare_routing({
                "TREESEM_AGENT_ROUTING_MODE": "rule",
            })
        self.assertEqual(mode, "rule")
        self.assertTrue(directory.is_dir())


if __name__ == "__main__":
    unittest.main()
