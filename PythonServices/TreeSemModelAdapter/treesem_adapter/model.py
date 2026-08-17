from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from typing import Any

from .bundle import (
    FeaturePreprocessor,
    ServingBundle,
    TorchTreeSemEngine,
    TreeEvaluator,
)


class RequestValidationError(ValueError):
    """Raised when an inference request violates the adapter contract."""


class ServingBundlePredictor:
    """Self-contained golden implementation backed only by a serving bundle."""

    def __init__(self, bundle_path: str | Path) -> None:
        self._bundle = ServingBundle(bundle_path)
        self._preprocessor = FeaturePreprocessor(self._bundle)
        self._engine = TorchTreeSemEngine(self._bundle)
        self._tree = TreeEvaluator(self._bundle)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "service": "treeSem-model-adapter",
            "dataset": "pph",
            "input_dim": self._bundle.input_dim,
            "test_samples": (
                int(len(self._bundle.reference_inputs))
                if self._bundle.reference_inputs is not None
                else 0
            ),
            "device": "cpu",
            "model_version": self._bundle.model_version,
            "serving_backend": "python_reference",
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RequestValidationError("request body must be a JSON object")
        input_fields = [
            name
            for name in ("sample_index", "preprocessed_features", "raw_features")
            if name in payload
        ]
        if len(input_fields) != 1:
            raise RequestValidationError(
                "provide exactly one of sample_index, preprocessed_features or raw_features"
            )

        sample_index: int | None = None
        field = input_fields[0]
        try:
            if field == "sample_index":
                value = payload[field]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise RequestValidationError("sample_index must be an integer")
                reference = self._bundle.reference_inputs
                if reference is None:
                    raise RequestValidationError(
                        "sample_index is unavailable in this serving bundle"
                    )
                if value < 0 or value >= len(reference):
                    raise RequestValidationError("sample_index is out of range")
                sample_index = value
                standardized = reference[value]
                source = "pph_reference_dataset"
            elif field == "preprocessed_features":
                values = payload[field]
                if not isinstance(values, list):
                    raise RequestValidationError(
                        "preprocessed_features must be an array"
                    )
                standardized = self._preprocessor.validate_standardized(values)
                source = "request_preprocessed"
            else:
                standardized = self._preprocessor.raw_object_to_standardized(
                    payload[field]
                )
                source = "request_raw"
        except RequestValidationError:
            raise
        except (TypeError, ValueError) as error:
            raise RequestValidationError(str(error)) from error

        probabilities, cluster_logits = self._engine.predict(standardized)
        label = max(range(len(probabilities)), key=probabilities.__getitem__)
        cluster_id = max(range(len(cluster_logits)), key=cluster_logits.__getitem__)
        tree_probability, leaf_id, path = self._tree.evaluate(standardized)
        numeric_values = [*probabilities, *cluster_logits, tree_probability]
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise RuntimeError("model returned a non-finite prediction")

        return {
            "model": "treeSem",
            "model_version": self._bundle.model_version,
            "serving_backend": "python_reference",
            "dataset": "pph",
            "input_source": source,
            "sample_index": sample_index,
            "prediction": {
                "label": int(label),
                "positive_probability": float(probabilities[1]),
                "confidence": float(probabilities[label]),
                "cluster_id": int(cluster_id),
                "tree_probability": float(tree_probability),
                "tree_leaf_id": int(leaf_id),
            },
            "important_features": self._important_features(standardized),
            "decision_path": self._decision_path(path),
        }

    def _important_features(self, standardized: Any) -> list[dict[str, Any]]:
        result = []
        for index in self._tree.important_indices(limit=5):
            feature = self._bundle.feature(index)
            value = float(standardized[index])
            result.append(
                {
                    "index": index,
                    "name": feature.name,
                    "display_name": feature.display_name,
                    "standardized_value": value,
                    "original_value": self._preprocessor.original_value(index, value),
                    "unit": feature.unit,
                    "tree_importance": self._tree.importances[index],
                }
            )
        return result

    def _decision_path(self, path: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for step in path:
            if "leaf_id" in step:
                result.append(step)
                continue
            index = int(step["feature_index"])
            feature = self._bundle.feature(index)
            result.append(
                {
                    **step,
                    "feature_name": feature.name,
                    "feature_display_name": feature.display_name,
                    "threshold_original": self._preprocessor.original_value(
                        index, float(step["threshold_standardized"])
                    ),
                    "value_original": self._preprocessor.original_value(
                        index, float(step["value_standardized"])
                    ),
                    "unit": feature.unit,
                }
            )
        return result


class TreeSemPredictor:
    """Loads one trusted PPH artifact and serves deterministic predictions."""

    def __init__(
        self,
        *,
        artifact_path: str | Path,
        code_root: str | Path,
        raw_file: str | Path,
        device: str = "cpu",
    ) -> None:
        code_root = Path(code_root).resolve()
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))

        import numpy as np
        import torch

        # The research package keeps its historical import name. New service
        # code and external contracts use treeSem.
        from trivae.data.pph import load_pph
        from trivae.model import VAE

        artifact_path = Path(artifact_path).resolve()
        raw_file = Path(raw_file).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"artifact not found: {artifact_path}")
        if not raw_file.is_file():
            raise FileNotFoundError(f"PPH data not found: {raw_file}")

        self._np = np
        self._torch = torch
        self._lock = threading.Lock()
        self._artifact_path = artifact_path
        self._device = torch.device(device)

        # weights_only=False is required by the trusted local artifact because
        # it contains a sklearn tree. Never use this for untrusted uploads.
        artifact = torch.load(
            artifact_path,
            map_location=self._device,
            weights_only=False,
        )
        self._validate_artifact(artifact)
        state = artifact["model_state"]
        self._input_dim = int(artifact["model_config"]["input_dim"])

        model = VAE(
            input_dim=self._input_dim,
            hidden_dim=int(state["fc1.weight"].shape[0]),
            latent_dim=int(artifact["model_config"]["latent_dim"]),
            num_clusters=int(state["cluster_layer.weight"].shape[0]),
            num_classes=int(state["main.3.weight"].shape[0]),
        ).to(self._device)
        model.load_state_dict(state, strict=False)
        model.eval()
        self._model = model

        args = artifact.get("args", {})
        seed = int(args.get("seed", 42))
        bundle = load_pph(
            raw_file,
            random_state=seed,
            stratify=bool(args.get("stratify_split", False)),
            sampler_balance_power=float(args.get("sampler_balance_power", 1.0)),
        )
        self._test_data = np.asarray(bundle.test_data, dtype=np.float32)
        self._feature_names = list(bundle.feature_names or [])
        if self._test_data.ndim != 2 or self._test_data.shape[1] != self._input_dim:
            raise ValueError(
                "reconstructed PPH data does not match artifact input dimension"
            )

        self._tree = artifact["tree"]
        self._tree_features = np.asarray(
            artifact.get("t_star", artifact.get("u_star")), dtype=np.int64
        )
        if self._tree_features.ndim != 1 or len(self._tree_features) == 0:
            raise ValueError("artifact contains no tree feature selection")
        if int(self._tree_features.min()) < 0 or int(self._tree_features.max()) >= self._input_dim:
            raise ValueError("artifact tree feature index is out of range")

    @staticmethod
    def _validate_artifact(artifact: dict[str, Any]) -> None:
        required = {"model_state", "model_config", "tree"}
        missing = sorted(required.difference(artifact))
        if missing:
            raise ValueError(f"artifact is missing required fields: {missing}")
        if "t_star" not in artifact and "u_star" not in artifact:
            raise ValueError("artifact is missing t_star/u_star")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "service": "treeSem-model-adapter",
            "dataset": "pph",
            "input_dim": self._input_dim,
            "test_samples": int(len(self._test_data)),
            "device": str(self._device),
            "artifact": self._artifact_path.name,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise RequestValidationError("request body must be a JSON object")

        has_index = "sample_index" in payload
        has_features = "preprocessed_features" in payload
        if has_index == has_features:
            raise RequestValidationError(
                "provide exactly one of sample_index or preprocessed_features"
            )

        sample_index: int | None = None
        if has_index:
            value = payload["sample_index"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise RequestValidationError("sample_index must be an integer")
            if value < 0 or value >= len(self._test_data):
                raise RequestValidationError("sample_index is out of range")
            sample_index = value
            features = self._test_data[value]
            source = "pph_test_split"
        else:
            values = payload["preprocessed_features"]
            if not isinstance(values, list) or len(values) != self._input_dim:
                raise RequestValidationError(
                    f"preprocessed_features must contain {self._input_dim} values"
                )
            try:
                features = self._np.asarray(values, dtype=self._np.float32)
            except (TypeError, ValueError) as error:
                raise RequestValidationError(
                    "preprocessed_features must be numeric"
                ) from error
            if not self._np.isfinite(features).all():
                raise RequestValidationError(
                    "preprocessed_features must contain only finite values"
                )
            source = "request"

        with self._lock, self._torch.inference_mode():
            batch = self._torch.as_tensor(
                features[None, :], dtype=self._torch.float32, device=self._device
            )
            mu, _ = self._model.encode(batch)
            probabilities = self._model.main(mu).softmax(dim=1)[0]
            label = int(probabilities.argmax().item())
            positive_probability = float(probabilities[1].item())
            confidence = float(probabilities[label].item())
            cluster_id = int(self._model.cluster_layer(mu).argmax(dim=1).item())

            selected = features[self._tree_features]
            tree_probability = float(self._tree.predict(selected[None, :])[0])
            leaf_id = int(self._tree.apply(selected[None, :])[0])

        if not all(
            math.isfinite(value)
            for value in (positive_probability, confidence, tree_probability)
        ):
            raise RuntimeError("model returned a non-finite prediction")

        return {
            "model": "treeSem",
            "dataset": "pph",
            "input_source": source,
            "sample_index": sample_index,
            "prediction": {
                "label": label,
                "positive_probability": positive_probability,
                "confidence": confidence,
                "cluster_id": cluster_id,
                "tree_probability": tree_probability,
                "tree_leaf_id": leaf_id,
            },
            "important_features": self._important_features(features),
            "decision_path": self._decision_path(selected),
        }

    def _important_features(self, features: Any, limit: int = 5) -> list[dict[str, Any]]:
        importances = self._np.asarray(self._tree.feature_importances_, dtype=float)
        order = self._np.argsort(importances)[::-1][:limit]
        result = []
        for local_index in order:
            global_index = int(self._tree_features[int(local_index)])
            name = (
                self._feature_names[global_index]
                if global_index < len(self._feature_names)
                else f"feature_{global_index}"
            )
            result.append(
                {
                    "index": global_index,
                    "name": name,
                    "standardized_value": float(features[global_index]),
                    "tree_importance": float(importances[int(local_index)]),
                }
            )
        return result

    def _decision_path(self, selected: Any) -> list[dict[str, Any]]:
        tree = self._tree.tree_
        node = 0
        path: list[dict[str, Any]] = []
        while tree.children_left[node] != tree.children_right[node]:
            local_index = int(tree.feature[node])
            global_index = int(self._tree_features[local_index])
            value = float(selected[local_index])
            threshold = float(tree.threshold[node])
            go_left = value <= threshold
            name = (
                self._feature_names[global_index]
                if global_index < len(self._feature_names)
                else f"feature_{global_index}"
            )
            path.append(
                {
                    "node_id": int(node),
                    "feature_index": global_index,
                    "feature_name": name,
                    "operator": "<=" if go_left else ">",
                    "threshold_standardized": threshold,
                    "value_standardized": value,
                }
            )
            node = int(tree.children_left[node] if go_left else tree.children_right[node])
        path.append({"leaf_id": int(node)})
        return path
