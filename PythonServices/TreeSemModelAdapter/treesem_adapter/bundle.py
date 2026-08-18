from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BundleValidationError(ValueError):
    """Raised when a serving bundle is missing, corrupt, or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BundleValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FeatureDefinition:
    index: int
    name: str
    display_name: str
    value_kind: str
    source_dtype: str
    unit: str | None
    mean: float
    scale: float


class ServingBundle:
    """Immutable, checksummed treeSem inference assets."""

    def __init__(self, directory: str | Path) -> None:
        import numpy as np

        self.directory = Path(directory).resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise BundleValidationError("manifest.json is missing")
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BundleValidationError("manifest.json is invalid") from error
        self.schema_version = self.manifest.get("bundle_schema_version")
        if self.schema_version not in (1, 2):
            raise BundleValidationError("unsupported bundle schema version")
        if self.manifest.get("model") != "treeSem" or self.manifest.get("dataset") != "pph":
            raise BundleValidationError("bundle model or dataset is unsupported")
        self.model_version = self._required_string("model_version")
        self.input_dim = self._required_positive_int("input_dim")
        self.class_count = self._required_positive_int("class_count")
        self.cluster_count = self._required_positive_int("cluster_count")
        if self.input_dim != 49 or self.class_count != 2:
            raise BundleValidationError("PPH bundle dimensions are unsupported")

        files = self.manifest.get("files")
        if not isinstance(files, dict):
            raise BundleValidationError("manifest files must be an object")
        for logical_name, metadata in files.items():
            if not isinstance(metadata, dict):
                raise BundleValidationError(f"file metadata is invalid: {logical_name}")
            relative_name = metadata.get("name")
            expected_size = metadata.get("size")
            expected_sha = metadata.get("sha256")
            if not isinstance(relative_name, str) or Path(relative_name).name != relative_name:
                raise BundleValidationError(f"file name is invalid: {logical_name}")
            path = self.directory / relative_name
            if not path.is_file():
                raise BundleValidationError(f"bundle file is missing: {logical_name}")
            if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
                raise BundleValidationError(f"bundle checksum failed: {logical_name}")
        if self.schema_version == 2:
            self._validate_v2_manifest(files)

        feature_payload = self._read_json_file("feature_schema")
        preprocessing = self._read_json_file("preprocessing")
        feature_rows = feature_payload.get("features")
        means = preprocessing.get("mean")
        scales = preprocessing.get("scale")
        if not isinstance(feature_rows, list) or len(feature_rows) != self.input_dim:
            raise BundleValidationError("feature schema dimension is invalid")
        if not isinstance(means, list) or not isinstance(scales, list):
            raise BundleValidationError("preprocessing arrays are invalid")
        if len(means) != self.input_dim or len(scales) != self.input_dim:
            raise BundleValidationError("preprocessing dimension is invalid")

        features: list[FeatureDefinition] = []
        names: set[str] = set()
        for index, row in enumerate(feature_rows):
            if not isinstance(row, dict) or row.get("index") != index:
                raise BundleValidationError("feature indices must be ordered and contiguous")
            name = row.get("name")
            display_name = row.get("display_name")
            if not isinstance(name, str) or not name or name in names:
                raise BundleValidationError("feature names must be unique non-empty strings")
            if not isinstance(display_name, str) or not display_name:
                raise BundleValidationError("feature display names must be non-empty")
            if row.get("value_kind") != "numeric":
                raise BundleValidationError("only numeric PPH features are supported")
            unit = row.get("unit")
            if unit is not None and not isinstance(unit, str):
                raise BundleValidationError("feature unit must be a string or null")
            mean = _finite_number(means[index], f"mean[{index}]")
            scale = _finite_number(scales[index], f"scale[{index}]")
            if scale <= 0:
                raise BundleValidationError("feature scale must be positive")
            features.append(
                FeatureDefinition(
                    index=index,
                    name=name,
                    display_name=display_name,
                    value_kind="numeric",
                    source_dtype=str(row.get("source_dtype", "unknown")),
                    unit=unit,
                    mean=mean,
                    scale=scale,
                )
            )
            names.add(name)
        self.features = tuple(features)
        self.feature_names = tuple(feature.name for feature in self.features)
        self._feature_by_name = {feature.name: feature for feature in self.features}

        neural_path = self._file_path("neural_state")
        try:
            with np.load(neural_path, allow_pickle=False) as state:
                self.neural_state = {name: np.asarray(state[name]).copy() for name in state.files}
        except Exception as error:
            raise BundleValidationError("neural_state.npz is invalid") from error
        self._validate_neural_state()

        self.tree = self._read_json_file("tree")
        self._validate_tree()

        self.reference_inputs = None
        self.reference_labels = None
        if self.manifest.get("contains_reference_dataset"):
            reference_path = self._file_path("reference_inputs")
            rows = self._required_nonnegative_int("reference_rows")
            raw = np.fromfile(reference_path, dtype="<f4")
            if raw.size != rows * self.input_dim or not np.isfinite(raw).all():
                raise BundleValidationError("reference input matrix is invalid")
            self.reference_inputs = raw.reshape(rows, self.input_dim)
            if self.schema_version >= 2 and self.manifest.get("contains_reference_labels"):
                label_path = self._file_path("reference_labels")
                labels = np.fromfile(label_path, dtype="i1")
                if labels.size != rows or not np.isin(labels, [0, 1]).all():
                    raise BundleValidationError("reference label vector is invalid")
                self.reference_labels = labels.astype(np.int64)

    def _validate_v2_manifest(self, files: dict[str, Any]) -> None:
        def digest(value: Any, name: str) -> str:
            if (not isinstance(value, str) or len(value) != 64 or
                    any(ch not in "0123456789abcdef" for ch in value)):
                raise BundleValidationError(f"v2 {name} is not a SHA-256 digest")
            return value

        provenance = self.manifest.get("provenance")
        evaluation = self.manifest.get("evaluation")
        if not isinstance(provenance, dict) or not isinstance(evaluation, dict):
            raise BundleValidationError("v2 provenance and evaluation are required")
        artifact_sha = digest(self.manifest.get("artifact_sha256"), "artifact checksum")
        raw_sha = digest(self.manifest.get("raw_data_sha256"), "raw dataset checksum")
        if digest(provenance.get("artifact_sha256"), "provenance artifact checksum") != artifact_sha:
            raise BundleValidationError("v2 artifact provenance is inconsistent")
        if digest(provenance.get("raw_dataset_sha256"), "provenance dataset checksum") != raw_sha:
            raise BundleValidationError("v2 dataset provenance is inconsistent")
        if digest(provenance.get("feature_schema_sha256"), "feature schema checksum") != files.get("feature_schema", {}).get("sha256"):
            raise BundleValidationError("v2 feature schema provenance is inconsistent")
        if digest(provenance.get("preprocessing_sha256"), "preprocessing checksum") != files.get("preprocessing", {}).get("sha256"):
            raise BundleValidationError("v2 preprocessing provenance is inconsistent")
        fingerprint = provenance.get("split_fingerprint")
        if not isinstance(fingerprint, dict):
            raise BundleValidationError("v2 split fingerprint is required")
        for name in ("combined_sha256", "test_indices_sha256", "train_indices_sha256"):
            digest(fingerprint.get(name), f"split {name}")
        label_schema = provenance.get("label_schema")
        if label_schema != {"negative": 0, "positive": 1,
                            "source_negative_value": -1, "threshold": 0.5}:
            raise BundleValidationError("v2 label schema is unsupported")
        runtime = provenance.get("runtime_versions")
        if (not isinstance(runtime, dict) or
                not {"python", "torch", "sklearn"}.issubset(runtime) or
                any(not isinstance(value, str) or not value for value in runtime.values())):
            raise BundleValidationError("v2 runtime provenance is invalid")
        commit = provenance.get("training_code_commit")
        if commit is not None and (not isinstance(commit, str) or len(commit) != 40 or
                                   any(ch not in "0123456789abcdef" for ch in commit)):
            raise BundleValidationError("v2 training commit is invalid")
        for snapshot_name in ("artifact_metrics", "recomputed_metrics"):
            snapshot = evaluation.get(snapshot_name)
            if not isinstance(snapshot, dict):
                raise BundleValidationError(f"v2 {snapshot_name} is required")
            for metric in ("accuracy", "auc", "auprc", "balanced_accuracy",
                           "positive_f1"):
                _finite_number(snapshot.get(metric), f"v2 {snapshot_name}.{metric}")
        if bool(self.manifest.get("contains_reference_labels")) != bool(
                self.manifest.get("contains_reference_dataset")):
            raise BundleValidationError("v2 reference inputs and labels must be paired")

    def _required_string(self, name: str) -> str:
        value = self.manifest.get(name)
        if not isinstance(value, str) or not value:
            raise BundleValidationError(f"manifest {name} is invalid")
        return value

    def _required_positive_int(self, name: str) -> int:
        value = self.manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BundleValidationError(f"manifest {name} is invalid")
        return value

    def _required_nonnegative_int(self, name: str) -> int:
        value = self.manifest.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BundleValidationError(f"manifest {name} is invalid")
        return value

    def _file_path(self, logical_name: str) -> Path:
        try:
            relative_name = self.manifest["files"][logical_name]["name"]
        except (KeyError, TypeError) as error:
            raise BundleValidationError(f"manifest file is missing: {logical_name}") from error
        return self.directory / relative_name

    def _read_json_file(self, logical_name: str) -> dict[str, Any]:
        try:
            value = json.loads(self._file_path(logical_name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BundleValidationError(f"bundle JSON is invalid: {logical_name}") from error
        if not isinstance(value, dict):
            raise BundleValidationError(f"bundle JSON must be an object: {logical_name}")
        return value

    def _validate_neural_state(self) -> None:
        expected = {
            "fc1.weight": (64, 49),
            "fc1.bias": (64,),
            "fc21.weight": (64, 64),
            "fc21.bias": (64,),
            "main.0.weight": (32, 64),
            "main.0.bias": (32,),
            "main.3.weight": (self.class_count, 32),
            "main.3.bias": (self.class_count,),
            "cluster_layer.weight": (self.cluster_count, 64),
            "cluster_layer.bias": (self.cluster_count,),
        }
        if set(self.neural_state) != set(expected):
            raise BundleValidationError("neural state keys are invalid")
        for name, shape in expected.items():
            value = self.neural_state[name]
            if value.shape != shape or str(value.dtype) != "float32":
                raise BundleValidationError(f"neural state shape or dtype is invalid: {name}")
            if not __import__("numpy").isfinite(value).all():
                raise BundleValidationError(f"neural state is non-finite: {name}")

    def _validate_tree(self) -> None:
        nodes = self.tree.get("nodes")
        importances = self.tree.get("global_feature_importance")
        node_count = self.tree.get("node_count")
        if not isinstance(nodes, list) or len(nodes) != node_count or not nodes:
            raise BundleValidationError("tree nodes are invalid")
        if not isinstance(importances, list) or len(importances) != self.input_dim:
            raise BundleValidationError("tree feature importance is invalid")
        for value in importances:
            _finite_number(value, "tree feature importance")
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or node.get("id") != index:
                raise BundleValidationError("tree node ids must be contiguous")
            is_leaf = node.get("is_leaf")
            if not isinstance(is_leaf, bool):
                raise BundleValidationError("tree leaf marker is invalid")
            _finite_number(node.get("value"), "tree node value")
            if is_leaf:
                if node.get("left") is not None or node.get("right") is not None:
                    raise BundleValidationError("tree leaf has children")
            else:
                left, right = node.get("left"), node.get("right")
                feature_index = node.get("feature_index")
                if not all(isinstance(value, int) and not isinstance(value, bool) for value in (left, right, feature_index)):
                    raise BundleValidationError("tree branch indices are invalid")
                if not (0 <= left < node_count and 0 <= right < node_count):
                    raise BundleValidationError("tree child is out of range")
                if not 0 <= feature_index < self.input_dim:
                    raise BundleValidationError("tree feature is out of range")
                _finite_number(node.get("threshold_standardized"), "tree threshold")

    def feature(self, index: int) -> FeatureDefinition:
        return self.features[index]


class FeaturePreprocessor:
    def __init__(self, bundle: ServingBundle) -> None:
        import numpy as np

        self.bundle = bundle
        self._np = np
        self._mean = np.asarray([feature.mean for feature in bundle.features], dtype=np.float64)
        self._scale = np.asarray([feature.scale for feature in bundle.features], dtype=np.float64)

    def raw_object_to_standardized(self, values: Any) -> Any:
        if not isinstance(values, dict) or set(values) != set(self.bundle.feature_names):
            raise ValueError("raw_features must contain exactly the 49 configured fields")
        ordered = []
        for name in self.bundle.feature_names:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("raw_features values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("raw_features values must be finite")
            ordered.append(number)
        return ((self._np.asarray(ordered, dtype=self._np.float64) - self._mean) / self._scale).astype(self._np.float32)

    def validate_standardized(self, values: Any) -> Any:
        array = self._np.asarray(values, dtype=self._np.float32)
        if array.ndim != 1 or array.shape[0] != self.bundle.input_dim:
            raise ValueError("preprocessed_features has an invalid dimension")
        if not self._np.isfinite(array).all():
            raise ValueError("preprocessed_features must be finite")
        return array

    def original_value(self, index: int, standardized: float) -> float:
        result = float(standardized) * float(self._scale[index]) + float(self._mean[index])
        if not math.isfinite(result):
            raise RuntimeError("inverse preprocessing produced a non-finite value")
        return result


class TorchTreeSemEngine:
    """Deterministic PyTorch implementation used as the serving oracle."""

    def __init__(self, bundle: ServingBundle) -> None:
        import torch

        self._torch = torch
        self._lock = threading.Lock()
        self._state = {
            name: torch.from_numpy(value.copy()).to(dtype=torch.float32)
            for name, value in bundle.neural_state.items()
        }

    def predict(self, standardized: Any) -> tuple[list[float], list[float]]:
        torch = self._torch
        with self._lock, torch.inference_mode():
            x = torch.as_tensor(standardized[None, :], dtype=torch.float32)
            hidden = torch.relu(torch.nn.functional.linear(x, self._state["fc1.weight"], self._state["fc1.bias"]))
            mu = torch.nn.functional.linear(hidden, self._state["fc21.weight"], self._state["fc21.bias"])
            classifier_hidden = torch.relu(torch.nn.functional.linear(mu, self._state["main.0.weight"], self._state["main.0.bias"]))
            logits = torch.nn.functional.linear(classifier_hidden, self._state["main.3.weight"], self._state["main.3.bias"])
            probabilities = torch.softmax(logits, dim=1)[0]
            cluster_logits = torch.nn.functional.linear(mu, self._state["cluster_layer.weight"], self._state["cluster_layer.bias"])[0]
        return probabilities.tolist(), cluster_logits.tolist()


class TreeEvaluator:
    def __init__(self, bundle: ServingBundle) -> None:
        self.bundle = bundle
        self.nodes = bundle.tree["nodes"]
        self.importances = tuple(float(value) for value in bundle.tree["global_feature_importance"])

    def evaluate(self, standardized: Any) -> tuple[float, int, list[dict[str, Any]]]:
        node_id = int(self.bundle.tree.get("root_node", 0))
        path: list[dict[str, Any]] = []
        visited = 0
        while True:
            if visited > len(self.nodes):
                raise RuntimeError("tree contains a cycle")
            visited += 1
            node = self.nodes[node_id]
            if node["is_leaf"]:
                path.append({"leaf_id": node_id})
                return float(node["value"]), node_id, path
            feature_index = int(node["feature_index"])
            value = float(standardized[feature_index])
            threshold = float(node["threshold_standardized"])
            go_left = value <= threshold
            path.append(
                {
                    "node_id": node_id,
                    "feature_index": feature_index,
                    "operator": "<=" if go_left else ">",
                    "threshold_standardized": threshold,
                    "value_standardized": value,
                }
            )
            node_id = int(node["left"] if go_left else node["right"])

    def important_indices(self, limit: int = 5) -> list[int]:
        return sorted(range(len(self.importances)), key=lambda index: (-self.importances[index], index))[:limit]
