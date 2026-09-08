from __future__ import annotations

import unittest
import subprocess
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODEL = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


class RoutingPackagingTest(unittest.TestCase):
    def test_agent_image_keeps_semantic_dependencies_optional(self):
        dockerfile = (ROOT / "deploy/docker/agent.Dockerfile").read_text(
            encoding="utf-8")
        self.assertIn("ARG TREESEM_INSTALL_SEMANTIC_ROUTING=false", dockerfile)
        self.assertIn("requirements-routing.txt", dockerfile)
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING", dockerfile)
        self.assertNotIn("snapshot_download", dockerfile)
        requirements = (
            ROOT / "PythonServices/TreeSemAgent/requirements-routing.txt"
        ).read_text(encoding="utf-8")
        self.assertIn("torch==2.5.1+cpu", requirements)
        self.assertIn("https://download.pytorch.org/whl/cpu", requirements)

    def test_compose_uses_pinned_offline_read_only_routing_model(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING", compose)
        self.assertIn("TREESEM_AGENT_ROUTING_MODE", compose)
        self.assertIn("TREESEM_AGENT_ROUTING_MODEL_REVISION", compose)
        self.assertIn("HF_HUB_OFFLINE: \"1\"", compose)
        self.assertIn("TRANSFORMERS_OFFLINE: \"1\"", compose)
        self.assertIn("/app/.cache/huggingface:ro", compose)

    def test_environment_defaults_to_rules_and_pins_model_revision(self):
        environment = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("TREESEM_AGENT_ROUTING_MODE=rule", environment)
        self.assertIn(f"TREESEM_AGENT_ROUTING_MODEL={MODEL}", environment)
        self.assertIn(
            f"TREESEM_AGENT_ROUTING_MODEL_REVISION={REVISION}", environment)
        self.assertIn("TREESEM_INSTALL_SEMANTIC_ROUTING=false", environment)

    def test_operator_preparation_and_make_targets_are_reproducible(self):
        script = ROOT / "scripts/prepare-routing-model.sh"
        self.assertTrue(script.is_file())
        content = script.read_text(encoding="utf-8")
        self.assertIn(MODEL, content)
        self.assertIn(REVISION, content)
        self.assertIn("snapshot_download", content)
        self.assertIn("local_files_only=True", content)
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        for target in (
                "routing-unit:", "routing-calibrate:",
                "routing-evaluate:", "routing-load-smoke:"):
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


if __name__ == "__main__":
    unittest.main()
